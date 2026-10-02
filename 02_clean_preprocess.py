"""
02_clean_preprocess.py
Week 2: data cleaning and preprocessing pipeline for the SwiftMove delivery data.
Input : data/logistics_raw.csv        (messy, as collected)
Output: data/logistics_clean.csv      (analysis-ready, original units)
        data/logistics_scaled.csv     (model-ready: encoded + scaled)
        results/week2.json, figures/w2_*.png
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sns.set_theme(style="whitegrid", font_scale=0.95)
R = {}                                             # results log used in the report

# [[load_audit]]
df = pd.read_csv("data/logistics_raw.csv")
R["raw_rows"], R["raw_cols"] = df.shape
print(df.shape)
print(df.dtypes)
missing_pct = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
print(missing_pct[missing_pct > 0])
print("Exact duplicate rows:", df.duplicated().sum())
print("Duplicate order_ids :", df["order_id"].duplicated().sum())
print(df.describe().T[["min", "50%", "max"]])
# [[/load_audit]]
R["missing_pct_raw"] = missing_pct[missing_pct > 0].to_dict()
R["dup_rows"] = int(df.duplicated().sum())
R["describe_raw"] = df.describe().T[["min", "50%", "max"]].round(1).to_dict("index")
df_raw = df.copy()

# [[dedupe]]
before = len(df)
df = df.drop_duplicates(subset="order_id", keep="first").reset_index(drop=True)
print("Removed duplicates:", before - len(df))
# [[/dedupe]]
R["dups_removed"] = before - len(df)

# [[text_standardise]]
R["zone_labels_before"] = sorted(df["zone"].astype(str).unique().tolist())
df["zone"] = (df["zone"].str.strip().str.title()
                        .replace({"Allahabad": "Prayagraj"}))     # legacy city name
df["vehicle_type"] = df["vehicle_type"].str.strip().str.title()
df["weather"] = df["weather"].str.strip().str.title()
print(df["zone"].value_counts())
# [[/text_standardise]]
R["zone_labels_after"] = sorted(df["zone"].unique().tolist())

# [[parse_types]]
# 1) Mixed date formats: try ISO first, then day-first for the rest
d1 = pd.to_datetime(df["order_date"], format="%Y-%m-%d", errors="coerce")
d2 = pd.to_datetime(df["order_date"], format="%d/%m/%Y", errors="coerce")
df["order_date"] = d1.fillna(d2)
assert df["order_date"].notna().all()

# 2) Numbers stored as text such as "45.2 km"
R["distance_text_values"] = int(df["distance_km"].astype("string").str.contains("km", na=False).sum())
df["distance_km"] = pd.to_numeric(
    df["distance_km"].astype("string").str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")
# [[/parse_types]]

# [[validity_rules]]
capacity = df["vehicle_type"].map({"Pickup": 800, "Mini Truck": 2000, "Truck": 5000})
rules = {
    "distance_km": (df["distance_km"] <= 0),
    "load_kg": (df["load_kg"] < 0) | (df["load_kg"] > capacity),      # cross-field rule
    "traffic_index": df["traffic_index"].notna() & ~df["traffic_index"].between(1, 10),  # 99 = sentinel code
}
R["invalid_counts"] = {}
for col, bad in rules.items():
    R["invalid_counts"][col] = int(bad.sum())
    df.loc[bad, col] = np.nan                     # treat as missing, impute later

# The target variable is NOT imputed: impossible delivery times are dropped
bad_target = ~df["actual_delivery_hours"].between(0.5, 48)
R["invalid_counts"]["actual_delivery_hours (rows dropped)"] = int(bad_target.sum())
df = df.loc[~bad_target].reset_index(drop=True)
# [[/validity_rules]]

# [[outlier_iqr]]
def iqr_fences(s, k=3.0):
    q1, q3 = s.quantile([0.25, 0.75])
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)

# Distance: fences per zone; extreme values are entry errors -> set to NaN
dist_before = df["distance_km"].copy()
for z, g in df.groupby("zone"):
    lo, hi = iqr_fences(g["distance_km"].dropna())
    mask = (df["zone"] == z) & (df["distance_km"] > hi)
    df.loc[mask, "distance_km"] = np.nan
R["distance_outliers"] = int(dist_before.notna().sum() - df["distance_km"].notna().sum())

# Order value: genuinely skewed, so cap (winsorise) instead of deleting
lo, hi = iqr_fences(df["order_value_inr"].dropna())
R["value_outliers"] = int((df["order_value_inr"] > hi).sum())
value_before = df["order_value_inr"].copy()
df["order_value_inr"] = df["order_value_inr"].clip(upper=hi)
# [[/outlier_iqr]]

# [[missing_impute]]
# Keep a flag so later models / analysts know which values were filled
for c in ["distance_km", "load_kg", "traffic_index", "weather", "order_value_inr", "transport_cost_inr"]:
    df[f"{c}_imputed"] = df[c].isna().astype(int)
R["missing_before_impute"] = {c: int(df[c].isna().sum()) for c in
    ["distance_km", "load_kg", "traffic_index", "weather", "order_value_inr", "transport_cost_inr"]}
stats_before = df[["distance_km", "load_kg", "traffic_index"]].agg(["mean", "median", "std"]).round(2)

group_median = lambda col, by: df.groupby(by)[col].transform("median")
df["distance_km"]     = df["distance_km"].fillna(group_median("distance_km", ["zone", "vehicle_type"]))
df["load_kg"]         = df["load_kg"].fillna(group_median("load_kg", "vehicle_type"))
df["traffic_index"]   = df["traffic_index"].fillna(group_median("traffic_index", "zone"))
df["order_value_inr"] = df["order_value_inr"].fillna(group_median("order_value_inr", "vehicle_type"))
df["weather"]         = df["weather"].fillna(
    df.groupby(df["order_date"].dt.month)["weather"].transform(lambda s: s.mode().iloc[0]))

# Cost is roughly proportional to distance, so use the vehicle's typical cost per km
cpk = (df["transport_cost_inr"] / df["distance_km"]).groupby(df["vehicle_type"]).transform("median")
df["transport_cost_inr"] = df["transport_cost_inr"].fillna((cpk * df["distance_km"]).round(0))
assert df.isna().sum().sum() == 0
# [[/missing_impute]]
stats_after = df[["distance_km", "load_kg", "traffic_index"]].agg(["mean", "median", "std"]).round(2)
R["impute_stats_before"] = stats_before.to_dict()
R["impute_stats_after"] = stats_after.to_dict()

# [[feature_eng]]
df["month"] = df["order_date"].dt.month
df["day_of_week"] = df["order_date"].dt.dayofweek            # 0 = Monday
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["load_utilisation"] = (df["load_kg"] / df["vehicle_type"].map(
    {"Pickup": 800, "Mini Truck": 2000, "Truck": 5000})).round(3)
df["cost_per_km"] = (df["transport_cost_inr"] / df["distance_km"]).round(2)
df["delay_hours"] = (df["actual_delivery_hours"] - df["promised_hours"]).round(2)
df["is_late"] = (df["delay_hours"] > 0).astype(int)
# [[/feature_eng]]

# [[transform_scale]]
# Skewed money variable -> log transform
R["skew_value_before"] = float(skew(df["order_value_inr"]))
df["order_value_log"] = np.log1p(df["order_value_inr"])
R["skew_value_after"] = float(skew(df["order_value_log"]))

num_cols = ["distance_km", "num_stops", "load_kg", "traffic_index", "order_value_log", "load_utilisation"]
cat_cols = ["zone", "vehicle_type", "weather"]

encoded = pd.get_dummies(df[cat_cols], drop_first=True, dtype=int)         # one-hot encoding
minmax = pd.DataFrame(MinMaxScaler().fit_transform(df[num_cols]),
                      columns=[c + "_mm" for c in num_cols])                # range 0 to 1
zscore = pd.DataFrame(StandardScaler().fit_transform(df[num_cols]),
                      columns=[c + "_z" for c in num_cols])                 # mean 0, sd 1
scaled = pd.concat([df[["order_id"]], zscore, minmax, encoded,
                    df[["is_festival", "actual_delivery_hours"]]], axis=1)
# [[/transform_scale]]
R["scale_check"] = {"minmax_min": float(minmax.min().min()), "minmax_max": float(minmax.max().max()),
                    "z_mean_max_abs": float(zscore.mean().abs().max()),
                    "z_std_mean": float(zscore.std(ddof=0).mean())}
R["range_before_scaling"] = df[num_cols].agg(["min", "max"]).round(1).to_dict()

df.to_csv("data/logistics_clean.csv", index=False)
scaled.to_csv("data/logistics_scaled.csv", index=False)
R["clean_rows"], R["clean_cols"] = df.shape
R["scaled_cols"] = scaled.shape[1]
R["rows_lost_pct"] = round((1 - R["clean_rows"] / R["raw_rows"]) * 100, 2)

# ---------------- figures ----------------
fig, ax = plt.subplots(figsize=(7, 3.4))
mp = pd.Series(R["missing_pct_raw"]).sort_values()
ax.barh(mp.index, mp.values, color="#2E75B6")
for y, v in enumerate(mp.values): ax.text(v + 0.05, y, f"{v:.1f}%", va="center", fontsize=8)
ax.set_xlabel("Missing values (% of rows)"); ax.set_title("Missing data in the raw file")
plt.tight_layout(); plt.savefig("figures/w2_missing.png", dpi=170); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
d_raw = pd.to_numeric(df_raw["distance_km"].astype("string").str.extract(r"(-?\d+\.?\d*)")[0])
sns.boxplot(y=d_raw.dropna(), ax=axes[0], color="#F4B183"); axes[0].set_title("Distance (km): raw")
sns.boxplot(y=df["distance_km"], ax=axes[1], color="#9DC3E6"); axes[1].set_title("Distance (km): cleaned")
for a in axes: a.set_ylabel("km")
plt.tight_layout(); plt.savefig("figures/w2_outliers.png", dpi=170); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
sns.histplot(df["order_value_inr"], bins=40, ax=axes[0], color="#F4B183")
axes[0].set_title(f"Order value (capped): skew {R['skew_value_before']:.2f}")
sns.histplot(df["order_value_log"], bins=40, ax=axes[1], color="#9DC3E6")
axes[1].set_title(f"After log transform: skew {R['skew_value_after']:.2f}")
axes[0].set_xlabel("INR"); axes[1].set_xlabel("log(1 + INR)")
plt.tight_layout(); plt.savefig("figures/w2_skew.png", dpi=170); plt.close()

json.dump(R, open("results/week2.json", "w"), indent=1, default=str)
print(json.dumps({k: R[k] for k in ["raw_rows", "clean_rows", "dups_removed", "invalid_counts",
      "distance_outliers", "value_outliers", "missing_before_impute", "skew_value_before",
      "skew_value_after", "scale_check", "rows_lost_pct", "zone_labels_before"]}, indent=1, default=str))

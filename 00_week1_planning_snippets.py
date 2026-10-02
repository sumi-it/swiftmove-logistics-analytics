"""
00_week1_planning_snippets.py
Week 1: illustrative code for the planned analysis, run on a first sample of the raw SwiftMove data.
These snippets show HOW each step will be done; full cleaning is done in Week 2.
"""
import numpy as np
import pandas as pd

# [[w1_load_profile]]
df = pd.read_csv("data/logistics_raw.csv")
print("Shape:", df.shape)
print("Missing values (%):")
print((df.isna().mean() * 100).round(1).sort_values(ascending=False).head(6))

# Force numeric types; text such as "45 km" becomes a number, bad values become NaN
for col in ["distance_km", "load_kg", "actual_delivery_hours", "transport_cost_inr"]:
    df[col] = pd.to_numeric(df[col].astype("string").str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")
iso = pd.to_datetime(df["order_date"], format="%Y-%m-%d", errors="coerce")      # two date formats in the file
dmy = pd.to_datetime(df["order_date"], format="%d/%m/%Y", errors="coerce")
df["order_date"] = iso.fillna(dmy)
print("Date range:", df["order_date"].min().date(), "to", df["order_date"].max().date())
# [[/w1_load_profile]]

# [[w1_kpis]]
valid = df[df["actual_delivery_hours"].between(0.5, 48)]          # ignore impossible times

otd_pct       = (valid["actual_delivery_hours"] <= valid["promised_hours"]).mean() * 100
cost_per_del  = df["transport_cost_inr"].mean()
cost_per_km   = (df["transport_cost_inr"] / df["distance_km"]).median()
capacity      = df["vehicle_type"].str.strip().str.title().map({"Pickup": 800, "Mini Truck": 2000, "Truck": 5000})
utilisation   = (df["load_kg"].clip(lower=0) / capacity).mean() * 100

print(f"On-time delivery (preliminary): {otd_pct:.1f}%")
print(f"Average cost per delivery     : INR {cost_per_del:,.0f}")
print(f"Median cost per km            : INR {cost_per_km:.1f}")
print(f"Average vehicle utilisation   : {utilisation:.1f}%")
# [[/w1_kpis]]

# [[w1_demand_features]]
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

# Weekly order quantity per zone is the demand series to be forecast
df["zone"] = df["zone"].str.strip().str.title().replace({"Allahabad": "Prayagraj"})
weekly = (df.dropna(subset=["order_date"])
            .groupby(["zone", pd.Grouper(key="order_date", freq="W")])["order_qty"]
            .sum().reset_index().sort_values(["zone", "order_date"]))
weekly["lag_1"]  = weekly.groupby("zone")["order_qty"].shift(1)          # last week
weekly["roll_4"] = weekly.groupby("zone")["order_qty"].transform(lambda s: s.shift(1).rolling(4).mean())
weekly["week"]   = weekly["order_date"].dt.isocalendar().week.astype(int)
weekly = weekly.dropna()

cut = weekly["order_date"].max() - pd.Timedelta(weeks=10)         # test on the most recent 10 weeks
tr, te = weekly[weekly["order_date"] <= cut], weekly[weekly["order_date"] > cut]
X = ["lag_1", "roll_4", "week"]
rf = RandomForestRegressor(n_estimators=150, random_state=42).fit(tr[X], tr["order_qty"])
print("Forecast MAPE        :", round(mean_absolute_percentage_error(te["order_qty"], rf.predict(te[X])) * 100, 1), "%")
print("Naive (last week) MAPE:", round(mean_absolute_percentage_error(te["order_qty"], te["lag_1"]) * 100, 1), "%")
# [[/w1_demand_features]]

# [[w1_safety_stock]]
from scipy.stats import norm

service_level, lead_time_days = 0.95, 2
z = norm.ppf(service_level)                                       # about 1.645 for a 95% service level

daily = (df.dropna(subset=["order_date"]).groupby(["zone", "order_date"])["order_qty"]
           .sum().reset_index())                                  # daily demand per zone
stats = daily.groupby("zone")["order_qty"].agg(mean_d="mean", sigma_d="std")
stats["safety_stock"]  = z * stats["sigma_d"] * np.sqrt(lead_time_days)
stats["reorder_point"] = stats["mean_d"] * lead_time_days + stats["safety_stock"]
print(stats.round(0))
# [[/w1_safety_stock]]

# [[w1_clustering]]
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

trips = df[["distance_km", "load_kg", "num_stops"]].copy()
trips["load_kg"] = trips["load_kg"].clip(lower=0)
trips = trips.dropna()
Xs = StandardScaler().fit_transform(trips)                        # scale first: km and kg differ hugely
trips["cluster"] = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(Xs)
print(trips.groupby("cluster").agg(trips=("cluster", "size"), km=("distance_km", "mean"),
                                   kg=("load_kg", "mean"), stops=("num_stops", "mean")).round(1))
# [[/w1_clustering]]

# [[w1_late_model]]
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

m = valid.dropna(subset=["distance_km", "load_kg"]).sort_values("order_date").copy()
m["late"] = (m["actual_delivery_hours"] > m["promised_hours"]).astype(int)
m["traffic_index"] = pd.to_numeric(m["traffic_index"], errors="coerce").where(lambda s: s.between(1, 10))
m["traffic_index"] = m["traffic_index"].fillna(m["traffic_index"].median())
m["weather"] = m["weather"].str.strip().str.title().fillna("Clear")
m = pd.concat([m, pd.get_dummies(m["weather"], prefix="wx", dtype=int)], axis=1)
feats = ["distance_km", "num_stops", "load_kg", "traffic_index", "is_festival", "promised_hours",
         "wx_Fog", "wx_Rain"]
split = int(len(m) * 0.8)                                         # chronological split, not random
clf = RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42)
clf.fit(m[feats][:split], m["late"][:split])
print(classification_report(m["late"][split:], clf.predict(m[feats][split:]), digits=2))
# [[/w1_late_model]]

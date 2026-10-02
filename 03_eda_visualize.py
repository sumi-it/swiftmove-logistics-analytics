"""
03_eda_visualize.py
Week 3: exploratory data analysis and visualisation on the cleaned SwiftMove data.
Input : data/logistics_clean.csv     (output of 02_clean_preprocess.py)
Output: figures/w3_*.png, results/week3.json
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=0.95)
BLUE, ORANGE = "#2E75B6", "#ED7D31"
R = {}

# [[eda_stats]]
df = pd.read_csv("data/logistics_clean.csv", parse_dates=["order_date"])
key = ["actual_delivery_hours", "distance_km", "num_stops", "load_kg",
       "transport_cost_inr", "cost_per_km", "delay_hours"]
summary = df[key].agg(["mean", "median", "std", "min", "max", "skew"]).T.round(2)
print(summary)

print("On-time rate  :", round((1 - df["is_late"].mean()) * 100, 1), "%")
print(df.groupby("weather")["is_late"].mean().mul(100).round(1))

corr = df[["actual_delivery_hours", "distance_km", "num_stops", "load_kg", "traffic_index",
           "transport_cost_inr", "is_festival", "load_utilisation"]].corr()
print(corr["actual_delivery_hours"].sort_values(ascending=False).round(2))
# [[/eda_stats]]
R["n"] = len(df)
R["summary"] = summary.to_dict("index")
R["ontime_pct"] = round((1 - df["is_late"].mean()) * 100, 1)
R["late_pct"] = round(df["is_late"].mean() * 100, 1)
R["corr_target"] = corr["actual_delivery_hours"].round(2).to_dict()
R["corr_cost"] = corr["transport_cost_inr"].round(2).to_dict()

# [[fig_distribution]]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, col, color, title in zip(
        axes, ["actual_delivery_hours", "distance_km", "transport_cost_inr"],
        [BLUE, ORANGE, "#70AD47"], ["Delivery time (hours)", "Distance (km)", "Transport cost (INR)"]):
    sns.histplot(df[col], kde=True, bins=35, color=color, ax=ax)
    ax.axvline(df[col].mean(), color="black", ls="--", lw=1, label="mean")
    ax.axvline(df[col].median(), color="red", ls=":", lw=1.2, label="median")
    ax.set_title(title); ax.set_xlabel("")
axes[0].legend()
plt.tight_layout(); plt.savefig("figures/w3_distributions.png", dpi=170); plt.close()
# [[/fig_distribution]]

# [[fig_trend]]
m = df.groupby("month").agg(orders=("order_id", "count"),
                            avg_hours=("actual_delivery_hours", "mean"),
                            late_pct=("is_late", lambda s: s.mean() * 100))
fig, ax1 = plt.subplots(figsize=(8.5, 3.8))
ax1.bar(m.index, m["orders"], color="#BDD7EE", label="Deliveries")
ax1.set_ylim(0, 680); ax1.set_xlabel("Month (2025)"); ax1.set_ylabel("Number of deliveries"); ax1.set_xticks(range(1, 13))
ax2 = ax1.twinx()
ax2.plot(m.index, m["late_pct"], color=ORANGE, marker="o", label="Late deliveries (%)")
ax2.set_ylabel("Late deliveries (%)"); ax2.grid(False)
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2)
ax1.set_title("Monthly delivery volume and late-delivery rate")
plt.tight_layout(); plt.savefig("figures/w3_trend.png", dpi=170); plt.close()
# [[/fig_trend]]
R["monthly"] = m.round(1).to_dict("index")

# [[fig_group_compare]]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
order = df.groupby("zone")["actual_delivery_hours"].median().sort_values().index
sns.boxplot(data=df, x="zone", y="actual_delivery_hours", order=order, color="#9DC3E6", ax=axes[0])
axes[0].set_title("Delivery time by zone"); axes[0].set_xlabel(""); axes[0].set_ylabel("Hours")
axes[0].tick_params(axis="x", rotation=20)
late_w = df.groupby(["weather", "is_festival"])["is_late"].mean().mul(100).reset_index()
late_w["is_festival"] = late_w["is_festival"].map({0: "Normal day", 1: "Festival day"})
sns.barplot(data=late_w, x="weather", y="is_late", hue="is_festival",
            palette=["#9DC3E6", ORANGE], ax=axes[1])
axes[1].set_title("Late-delivery rate by weather"); axes[1].set_xlabel(""); axes[1].set_ylabel("Late (%)")
axes[1].legend(title="")
plt.tight_layout(); plt.savefig("figures/w3_group_compare.png", dpi=170); plt.close()
# [[/fig_group_compare]]
R["zone_median_hours"] = df.groupby("zone")["actual_delivery_hours"].median().round(2).to_dict()
R["zone_late_pct"] = df.groupby("zone")["is_late"].mean().mul(100).round(1).to_dict()
R["weather_late_pct"] = df.groupby("weather")["is_late"].mean().mul(100).round(1).to_dict()
R["festival_late_pct"] = df.groupby("is_festival")["is_late"].mean().mul(100).round(1).to_dict()
R["weather_counts"] = df["weather"].value_counts().to_dict()
R["festival_n"] = int(df["is_festival"].sum())

# [[fig_scatter]]
s = df.sample(1800, random_state=1)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
sns.scatterplot(data=s, x="distance_km", y="actual_delivery_hours", hue="vehicle_type",
                alpha=0.55, s=22, palette=["#5B9BD5", "#ED7D31", "#70AD47"], ax=ax)
sns.regplot(data=df, x="distance_km", y="actual_delivery_hours", scatter=False,
            color="black", line_kws={"lw": 1.6, "ls": "--"}, ax=ax)
ax.set_title("Distance vs delivery time"); ax.set_xlabel("Distance (km)"); ax.set_ylabel("Delivery time (hours)")
plt.tight_layout(); plt.savefig("figures/w3_scatter.png", dpi=170); plt.close()
# [[/fig_scatter]]
slope, intercept = np.polyfit(df["distance_km"], df["actual_delivery_hours"], 1)
R["slope_min_per_km"] = round(slope * 60, 2); R["intercept"] = round(intercept, 2)

# [[fig_corr]]
cols = ["actual_delivery_hours", "distance_km", "num_stops", "load_kg", "traffic_index",
        "transport_cost_inr", "is_festival", "load_utilisation"]
plt.figure(figsize=(7, 5.2))
sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            cbar_kws={"shrink": 0.8}, annot_kws={"size": 8})
plt.title("Correlation matrix"); plt.xticks(rotation=40, ha="right")
plt.tight_layout(); plt.savefig("figures/w3_corr.png", dpi=170); plt.close()
# [[/fig_corr]]

# [[fig_heatmap]]
df["traffic_band"] = pd.cut(df["traffic_index"], [0, 4, 6.5, 10], labels=["Low (<4)", "Medium (4-6.5)", "High (>6.5)"])
pivot = (df.pivot_table(index="weather", columns="traffic_band", values="is_late",
                        aggfunc="mean", observed=True) * 100).round(1)
plt.figure(figsize=(6.2, 3.3))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={"label": "Late (%)"})
plt.title("Late-delivery rate: weather x traffic"); plt.xlabel(""); plt.ylabel("")
plt.tight_layout(); plt.savefig("figures/w3_heatmap.png", dpi=170); plt.close()
# [[/fig_heatmap]]
R["pivot_late"] = pivot.to_dict("index")
R["traffic_band_late"] = df.groupby("traffic_band", observed=True)["is_late"].mean().mul(100).round(1).to_dict()

# [[fig_cost]]
df["util_band"] = pd.cut(df["load_utilisation"], [0, 0.3, 0.5, 0.7, 1.0],
                         labels=["<30%", "30-50%", "50-70%", ">70%"])
df["cost_per_kg"] = df["transport_cost_inr"] / df["load_kg"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
sns.boxplot(data=df, x="vehicle_type", y="cost_per_km", order=["Pickup", "Mini Truck", "Truck"],
            color="#A9D18E", ax=axes[0], showfliers=False)
axes[0].set_title("Cost per km by vehicle type"); axes[0].set_xlabel(""); axes[0].set_ylabel("INR per km")
cb = df.groupby(["vehicle_type", "util_band"], observed=True)["cost_per_kg"].mean().reset_index()
sns.barplot(data=cb, x="util_band", y="cost_per_kg", hue="vehicle_type",
            hue_order=["Pickup", "Mini Truck", "Truck"], palette=["#5B9BD5", ORANGE, "#70AD47"], ax=axes[1])
axes[1].set_title("Cost per kg carried by load utilisation"); axes[1].set_xlabel("Vehicle load utilisation")
axes[1].set_ylabel("INR per kg"); axes[1].legend(title="")
plt.tight_layout(); plt.savefig("figures/w3_cost.png", dpi=170); plt.close()
# [[/fig_cost]]
R["cost_per_km_vehicle"] = df.groupby("vehicle_type")["cost_per_km"].median().round(1).to_dict()
R["cost_per_kg_util"] = cb.pivot(index="util_band", columns="vehicle_type", values="cost_per_kg").round(2).to_dict("index")
R["util_share"] = df["util_band"].value_counts(normalize=True).mul(100).round(1).to_dict()
R["truck_low_util_pct"] = round(((df["vehicle_type"] == "Truck") & (df["load_utilisation"] < 0.5)).sum()
                                / (df["vehicle_type"] == "Truck").sum() * 100, 1)
R["cost_by_vehicle"] = df.groupby("vehicle_type")["transport_cost_inr"].agg(["mean", "sum"]).round(0).to_dict("index")
R["dow_late"] = df.groupby("day_of_week")["is_late"].mean().mul(100).round(1).to_dict()
R["stops_corr_hours"] = round(df["num_stops"].corr(df["actual_delivery_hours"]), 2)

json.dump(R, open("results/week3.json", "w"), indent=1, default=str)
print(json.dumps({k: R[k] for k in R if k not in ("summary",)}, indent=1, default=str)[:6000])

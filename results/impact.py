import json, pandas as pd
raw = pd.read_csv("data/logistics_raw.csv"); cl = pd.read_csv("data/logistics_clean.csv")
d = pd.to_numeric(raw["distance_km"].astype("string").str.extract(r"(-?\d+\.?\d*)")[0])
I = {"raw_mean_hours": raw["actual_delivery_hours"].mean(), "clean_mean_hours": cl["actual_delivery_hours"].mean(),
     "raw_mean_value": raw["order_value_inr"].mean(), "clean_mean_value": cl["order_value_inr"].mean(),
     "raw_mean_dist": d.mean(), "clean_mean_dist": cl["distance_km"].mean(), "raw_max_dist": d.max(),
     "raw_zone_groups": raw["zone"].nunique(), "clean_zone_groups": cl["zone"].nunique(),
     "raw_orders": len(raw), "unique_orders": raw["order_id"].nunique(),
     "raw_max_hours": raw["actual_delivery_hours"].max(), "raw_traffic_max": raw["traffic_index"].max(),
     "raw_cost_total": raw["transport_cost_inr"].sum(), "clean_cost_total": cl["transport_cost_inr"].sum()}
json.dump({k: float(v) for k, v in I.items()}, open("results/impact.json", "w"), indent=1); print(I)

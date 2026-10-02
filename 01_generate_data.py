"""
01_generate_data.py
Simulates a raw, messy delivery-level dataset for SwiftMove Distribution
(a regional FMCG distributor in Uttar Pradesh) for calendar year 2025.

The data is SYNTHETIC. It is built to mimic the structure of public logistics
datasets (e.g. DataCo Smart Supply Chain, Olist) and then deliberately
"damaged" with realistic data-quality problems so that a cleaning pipeline
can be demonstrated.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 6000

# ---------- 1. Clean "ground truth" ----------
dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, N), unit="D")
zones = rng.choice(["Lucknow", "Prayagraj", "Varanasi", "Kanpur", "Gorakhpur"],
                   N, p=[0.28, 0.24, 0.20, 0.16, 0.12])
zone_base_km = {"Lucknow": 28, "Prayagraj": 34, "Varanasi": 40, "Kanpur": 30, "Gorakhpur": 52}
distance = np.array([max(3, rng.gamma(4, zone_base_km[z] / 4)) for z in zones]).round(1)

vehicle = rng.choice(["Pickup", "Mini Truck", "Truck"], N, p=[0.35, 0.40, 0.25])
capacity = pd.Series(vehicle).map({"Pickup": 800, "Mini Truck": 2000, "Truck": 5000}).to_numpy()
util = rng.beta(4, 3, N)
load = (capacity * util).round(0)
stops = np.clip(rng.poisson(5, N) + 1, 1, 15)

month = dates.month.to_numpy()
weather = np.where(np.isin(month, [12, 1]), rng.choice(["Clear", "Fog", "Rain"], N, p=[0.45, 0.45, 0.10]),
          np.where(np.isin(month, [7, 8, 9]), rng.choice(["Clear", "Fog", "Rain"], N, p=[0.45, 0.02, 0.53]),
                   rng.choice(["Clear", "Fog", "Rain"], N, p=[0.85, 0.03, 0.12])))
festival = np.isin(dates.strftime("%m-%d"), [f"{m:02d}-{d:02d}" for m, d in
            [(1,14),(3,14),(3,31),(8,9),(8,16),(9,5),(10,2),(10,20),(10,21),(10,22),(10,23),(11,5)]])
festival = festival | (rng.random(N) < 0.03)
dow = dates.dayofweek.to_numpy()
traffic = np.clip(rng.normal(5, 1.6, N) + 0.6 * (dow < 5) + 1.0 * festival + 0.4 * (weather == "Rain"), 1, 10).round(1)

order_qty = np.clip((load / rng.uniform(8, 25, N)).round(0), 5, None)
order_value = (order_qty * rng.lognormal(mean=5.3, sigma=0.55, size=N)).round(0)   # right-skewed

hours = (1.2 + 0.045 * distance + 0.30 * stops + 0.0005 * load + 0.22 * traffic
         + 1.1 * (weather == "Rain") + 1.7 * (weather == "Fog") + 1.0 * festival
         + 0.5 * (vehicle == "Truck") + rng.normal(0, 0.55, N) * (1 + 0.04 * distance / 10))
hours = np.clip(hours, 0.8, None).round(2)
# planner rule of thumb: ignores weather / traffic / festivals
promised = np.ceil(2.9 + 0.05 * distance + 0.32 * stops + 0.0005 * load + 0.2).astype(float)

rate = pd.Series(vehicle).map({"Pickup": 11, "Mini Truck": 16, "Truck": 24}).to_numpy()
cost = (distance * rate + stops * 35 + load * 0.35 + rng.normal(0, 60, N)).round(0)

df = pd.DataFrame({
    "order_id": [f"SM{100000 + i}" for i in range(N)],
    "order_date": dates, "zone": zones, "vehicle_type": vehicle,
    "distance_km": distance, "num_stops": stops, "load_kg": load, "order_qty": order_qty,
    "order_value_inr": order_value, "traffic_index": traffic, "weather": weather,
    "is_festival": festival.astype(int), "promised_hours": promised,
    "actual_delivery_hours": hours, "transport_cost_inr": cost,
})
df.sort_values("order_date").reset_index(drop=True).to_csv("data/ground_truth_clean.csv", index=False)

# ---------- 2. Damage the data (simulate real-world collection problems) ----------
raw = df.copy()
raw["order_date"] = raw["order_date"].dt.strftime("%Y-%m-%d")
n = len(raw)
def idx(frac): return rng.choice(n, int(n * frac), replace=False)

# (a) mixed date formats
i = idx(0.06); raw.loc[i, "order_date"] = pd.to_datetime(raw.loc[i, "order_date"]).dt.strftime("%d/%m/%Y")
# (b) inconsistent zone labels
raw["zone"] = raw["zone"].astype(object)
for k in idx(0.05): raw.at[k, "zone"] = raw.at[k, "zone"].lower()
for k in idx(0.04): raw.at[k, "zone"] = " " + raw.at[k, "zone"].upper() + " "
for k in np.where(raw["zone"] == "Prayagraj")[0][:70]: raw.at[k, "zone"] = "Allahabad"
# (c) distance stored as text with units
raw["distance_km"] = raw["distance_km"].astype(object)
for k in idx(0.02): raw.at[k, "distance_km"] = f"{raw.at[k, 'distance_km']} km"
# (d) missing values
for col, frac in [("distance_km", 0.04), ("load_kg", 0.035), ("traffic_index", 0.03),
                  ("weather", 0.02), ("order_value_inr", 0.015), ("transport_cost_inr", 0.02)]:
    raw.loc[idx(frac), col] = np.nan
# (e) impossible / extreme values
for k in idx(0.006): raw.at[k, "distance_km"] = float(str(raw.at[k, "distance_km"]).replace(" km", "")) * 10 \
        if pd.notna(raw.at[k, "distance_km"]) else np.nan        # decimal-point / unit typos
i = idx(0.004); raw.loc[i, "load_kg"] = -raw.loc[i, "load_kg"].abs()                  # negative load
raw.loc[idx(0.005), "actual_delivery_hours"] = rng.choice([0, -2.5, 96, 240, 480], int(n * 0.005))
raw.loc[idx(0.004), "traffic_index"] = 99                                              # sentinel code
raw.loc[idx(0.006), "order_value_inr"] = raw["order_value_inr"].max() * rng.uniform(2, 5)
# (f) duplicate rows (double entry)
raw = pd.concat([raw, raw.iloc[idx(0.02)]], ignore_index=True)
raw = raw.sample(frac=1, random_state=1).reset_index(drop=True)
raw.to_csv("data/logistics_raw.csv", index=False)
print("Raw dataset written:", raw.shape)

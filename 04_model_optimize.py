"""
04_model_optimize.py
Week 4: predictive modelling (delivery-time forecasting) and optimisation strategies.
Input : data/logistics_clean.csv
Output: figures/w4_*.png, results/week4.json
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

sns.set_theme(style="whitegrid", font_scale=0.95)
R = {}
SEED = 42

# [[prep_split]]
df = pd.read_csv("data/logistics_clean.csv", parse_dates=["order_date"]).sort_values("order_date")

target = "actual_delivery_hours"
num = ["distance_km", "num_stops", "load_kg", "traffic_index", "is_festival", "is_weekend", "month"]
cat = ["zone", "vehicle_type", "weather"]
X, y = df[num + cat], df[target]

# Time-based split: learn from Jan-Sep, test on the unseen Oct-Dec quarter
train_mask = df["order_date"] < "2025-10-01"
X_tr, X_te, y_tr, y_te = X[train_mask], X[~train_mask], y[train_mask], y[~train_mask]
print(len(X_tr), "train rows,", len(X_te), "test rows")

pre = ColumnTransformer([("num", StandardScaler(), num),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
def make(model): return Pipeline([("prep", pre), ("model", model)])
# [[/prep_split]]
R["n_train"], R["n_test"] = int(len(X_tr)), int(len(X_te))
R["features"] = num + cat

# [[cv_baselines]]
tscv = TimeSeriesSplit(n_splits=5)                       # each fold trains on the past only
scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}

models = {
    "Baseline (mean)":        make(DummyRegressor()),
    "Linear Regression":      make(LinearRegression()),
    "Ridge (alpha=1)":        make(Ridge(alpha=1.0)),
    "Decision Tree (depth 6)": make(DecisionTreeRegressor(max_depth=6, random_state=SEED)),
    "Random Forest":          make(RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)),
    "Gradient Boosting":      make(GradientBoostingRegressor(random_state=SEED)),
}
# [[/cv_baselines]]

# [[tuning]]
rf_grid = {"model__n_estimators": [200, 400], "model__max_depth": [8, 12, None],
           "model__min_samples_leaf": [3, 8]}
rf_search = GridSearchCV(make(RandomForestRegressor(random_state=SEED, n_jobs=-1)), rf_grid,
                         cv=TimeSeriesSplit(n_splits=4), scoring="neg_root_mean_squared_error", n_jobs=1)
rf_search.fit(X_tr, y_tr)

gb_dist = {"model__n_estimators": [100, 200, 300, 400], "model__learning_rate": [0.03, 0.05, 0.1],
           "model__max_depth": [2, 3, 4], "model__subsample": [0.7, 0.85, 1.0],
           "model__min_samples_leaf": [5, 10, 20]}
gb_search = RandomizedSearchCV(make(GradientBoostingRegressor(random_state=SEED)), gb_dist, n_iter=15,
                               cv=TimeSeriesSplit(n_splits=4), scoring="neg_root_mean_squared_error",
                               random_state=SEED, n_jobs=-1)
gb_search.fit(X_tr, y_tr)
print("RF best:", rf_search.best_params_)
print("GB best:", gb_search.best_params_)
# [[/tuning]]
models["Random Forest (tuned)"] = rf_search.best_estimator_
models["Gradient Boosting (tuned)"] = gb_search.best_estimator_
del models["Random Forest"], models["Gradient Boosting"]
R["rf_best"] = {k.replace("model__", ""): v for k, v in rf_search.best_params_.items()}
R["gb_best"] = {k.replace("model__", ""): v for k, v in gb_search.best_params_.items()}

# [[evaluate]]
rows = []
for name, mdl in models.items():
    cv = cross_validate(mdl, X_tr, y_tr, cv=tscv, scoring=scoring)          # validation on the past
    mdl.fit(X_tr, y_tr)
    pred = mdl.predict(X_te)                                                # final unseen test
    rows.append({"Model": name,
                 "CV RMSE": -cv["test_rmse"].mean(), "CV RMSE sd": cv["test_rmse"].std(),
                 "CV R2": cv["test_r2"].mean(),
                 "Test MAE": mean_absolute_error(y_te, pred),
                 "Test RMSE": np.sqrt(mean_squared_error(y_te, pred)),
                 "Test R2": r2_score(y_te, pred)})
results = pd.DataFrame(rows).set_index("Model").round(3)
print(results)

best_name = results.drop("Baseline (mean)")["CV RMSE"].idxmin()             # choose on CV, not on test
best = models[best_name]
print("Selected model:", best_name)
# [[/evaluate]]
R["results"] = results.to_dict("index")
R["best_name"] = best_name
pred_te = best.predict(X_te)

# [[importance]]
perm = permutation_importance(best, X_te, y_te, n_repeats=10, random_state=SEED,
                              scoring="neg_root_mean_squared_error")
imp = pd.Series(perm.importances_mean, index=X_te.columns).sort_values(ascending=False)
print(imp.round(3))

# Interpretable view: coefficients of a plain linear model in real units (hours)
lin = LinearRegression().fit(pd.get_dummies(X_tr, columns=cat, drop_first=True, dtype=int), y_tr)
coef = pd.Series(lin.coef_, index=lin.feature_names_in_).round(5)
# [[/importance]]
R["importance"] = imp.round(3).to_dict()
R["coef"] = coef.to_dict()
R["baseline_test_rmse"] = float(results.loc["Baseline (mean)", "Test RMSE"])

# ---------------- optimisation ----------------
# [[opt_promise]]
# Out-of-fold residuals on the training period -> how wrong is the model typically?
oof = []
for tr_i, va_i in tscv.split(X_tr):
    m = make(best.named_steps["model"].__class__(**best.named_steps["model"].get_params()))
    m.fit(X_tr.iloc[tr_i], y_tr.iloc[tr_i])
    oof.append(y_tr.iloc[va_i] - m.predict(X_tr.iloc[va_i]))
buffer90 = np.quantile(np.concatenate(oof), 0.90)             # cover 90% of typical errors

test = df[~train_mask].copy()
test["pred"] = pred_te
test["new_promise"] = np.ceil(test["pred"] + buffer90)        # whole-hour promise, as planners use
otd_current = (test["actual_delivery_hours"] <= test["promised_hours"]).mean() * 100
otd_dynamic = (test["actual_delivery_hours"] <= test["new_promise"]).mean() * 100

# Fair comparison: how much would a flat "add c hours to every promise" need for the same OTD?
for c in range(0, 8):
    if (test["actual_delivery_hours"] <= test["promised_hours"] + c).mean() * 100 >= otd_dynamic:
        break
flat_avg = (test["promised_hours"] + c).mean()
# [[/opt_promise]]
R["opt_promise"] = {"buffer90": round(float(buffer90), 2), "otd_current": round(otd_current, 1),
    "otd_dynamic": round(otd_dynamic, 1), "avg_promise_current": round(test["promised_hours"].mean(), 2),
    "avg_promise_dynamic": round(test["new_promise"].mean(), 2), "flat_c": int(c),
    "avg_promise_flat": round(flat_avg, 2),
    "otd_flat": round((test["actual_delivery_hours"] <= test["promised_hours"] + c).mean() * 100, 1)}

# [[opt_alert]]
late_true = (test["actual_delivery_hours"] > test["promised_hours"]).astype(int)
alerts = {}
for margin in [0.0, 0.5, 1.0]:
    flag = (test["pred"] > test["promised_hours"] - margin).astype(int)      # predicted to be late
    alerts[margin] = {"flagged_pct": round(flag.mean() * 100, 1),
                      "precision": round(precision_score(late_true, flag), 3),
                      "recall": round(recall_score(late_true, flag), 3)}
print(alerts)
# [[/opt_alert]]
R["opt_alert"] = {str(k): v for k, v in alerts.items()}
R["late_rate_test"] = round(late_true.mean() * 100, 1)

# [[opt_vehicle]]
cap = {"Pickup": 800, "Mini Truck": 2000, "Truck": 5000}
cost_models = {v: LinearRegression().fit(g[["distance_km", "num_stops", "load_kg"]], g["transport_cost_inr"])
               for v, g in df.groupby("vehicle_type")}
smaller = {"Truck": "Mini Truck", "Mini Truck": "Pickup"}

saving, trips = 0.0, 0
for big, small in smaller.items():
    fits = df[(df["vehicle_type"] == big) & (df["load_kg"] <= 0.9 * cap[small])]   # 10% safety margin
    alt = cost_models[small].predict(fits[["distance_km", "num_stops", "load_kg"]])
    saving += (fits["transport_cost_inr"] - alt).sum()
    trips += len(fits)
total_cost = df["transport_cost_inr"].sum()
print(trips, "trips could use a smaller vehicle; estimated saving INR", round(saving))
# [[/opt_vehicle]]
R["opt_vehicle"] = {"trips": int(trips), "saving": round(float(saving)), "total_cost": round(float(total_cost)),
                    "saving_pct": round(float(saving / total_cost * 100), 1), "n_all": int(len(df)),
                    "cost_model_r2": {v: round(cost_models[v].score(g[["distance_km", "num_stops", "load_kg"]],
                                      g["transport_cost_inr"]), 3) for v, g in df.groupby("vehicle_type")}}

# ---------------- figures ----------------
res = results.drop("Baseline (mean)").copy()
fig, ax = plt.subplots(figsize=(8.5, 3.8))
idx = np.arange(len(res)); w = 0.38
ax.bar(idx - w / 2, res["CV RMSE"], w, label="Cross-validation RMSE", color="#9DC3E6")
ax.bar(idx + w / 2, res["Test RMSE"], w, label="Test RMSE (Oct-Dec)", color="#2E75B6")
ax.axhline(R["baseline_test_rmse"], color="red", ls="--", lw=1, label="Mean-only baseline")
ax.set_xticks(idx); ax.set_xticklabels([n.replace(" (", "\n(") for n in res.index], fontsize=8)
ax.set_ylabel("RMSE (hours)"); ax.set_title("Model comparison (lower is better)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures/w4_model_compare.png", dpi=170); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
axes[0].scatter(y_te, pred_te, s=10, alpha=0.4, color="#2E75B6")
lim = [y_te.min() - 0.5, y_te.max() + 0.5]
axes[0].plot(lim, lim, "r--", lw=1); axes[0].set_xlabel("Actual hours"); axes[0].set_ylabel("Predicted hours")
axes[0].set_title(f"Predicted vs actual: {best_name}", fontsize=10)
resid = y_te - pred_te
sns.histplot(resid, bins=40, kde=True, color="#ED7D31", ax=axes[1])
axes[1].axvline(0, color="black", lw=1); axes[1].set_xlabel("Residual (actual - predicted, hours)")
axes[1].set_title("Residual distribution (test set)", fontsize=10)
plt.tight_layout(); plt.savefig("figures/w4_pred_actual.png", dpi=170); plt.close()

fig, ax = plt.subplots(figsize=(7, 3.6))
imp.sort_values().plot(kind="barh", color="#2E75B6", ax=ax)
ax.set_xlabel("Increase in RMSE when the feature is shuffled (hours)"); ax.set_title("Permutation feature importance")
plt.tight_layout(); plt.savefig("figures/w4_importance.png", dpi=170); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
op = R["opt_promise"]
axes[0].bar(["Current\npromise", "Flat +%d h\nbuffer" % op["flat_c"], "Model-based\npromise"],
            [op["otd_current"], op["otd_flat"], op["otd_dynamic"]], color=["#BFBFBF", "#9DC3E6", "#2E75B6"])
axes[0].set_ylabel("On-time delivery (%)"); axes[0].set_title("On-time rate (test quarter)", fontsize=10); axes[0].set_ylim(0, 105)
for i, v in enumerate([op["otd_current"], op["otd_flat"], op["otd_dynamic"]]): axes[0].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)
axes[1].bar(["Current\npromise", "Flat +%d h\nbuffer" % op["flat_c"], "Model-based\npromise"],
            [op["avg_promise_current"], op["avg_promise_flat"], op["avg_promise_dynamic"]], color=["#BFBFBF", "#9DC3E6", "#2E75B6"])
axes[1].set_ylabel("Average promised time (hours)"); axes[1].set_title("Customer waiting time promised", fontsize=10)
for i, v in enumerate([op["avg_promise_current"], op["avg_promise_flat"], op["avg_promise_dynamic"]]): axes[1].text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)
axes[1].set_ylim(0, max(op["avg_promise_flat"], op["avg_promise_dynamic"]) + 2)
plt.tight_layout(); plt.savefig("figures/w4_promise.png", dpi=170); plt.close()

json.dump(R, open("results/week4.json", "w"), indent=1, default=str)
print(json.dumps({k: R[k] for k in ["best_name", "opt_promise", "opt_alert", "opt_vehicle", "importance", "coef", "rf_best", "gb_best"]}, indent=1, default=str))

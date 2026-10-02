# SwiftMove Logistics Analytics (Yuvaintern Weeks 1-4)

Analysis of delivery performance for a fictional regional FMCG distributor (SwiftMove Distribution, Uttar Pradesh).
All data is **synthetic**, generated to mimic public logistics datasets such as DataCo Smart Supply Chain and Olist.

Python analytics project for a simulated FMCG distributor: data cleaning, EDA, delivery-time prediction (R² 0.84) and optimisation that lifts on-time delivery from 75.9% to 96.1% in testing. Synthetic Data.



## Project structure
| File | Week | Purpose |
|---|---|---|
| `00_week1_planning_snippets.py` | 1 | Planning snippets run on the raw sample: KPIs, demand features, safety stock, clustering, late-risk model |
| `vrp_pseudocode.py` | 1 | Vehicle routing illustration (needs `ortools`, not run) |
| `make_roadmap_fig.py` | 1 | Draws the roadmap diagram |
| `01_generate_data.py` | 2 | Creates the raw, messy dataset with realistic quality problems |
| `02_clean_preprocess.py` | 2 | Cleaning, missing values, outliers, transformation, encoding, scaling |
| `03_eda_visualize.py` | 3 | Exploratory analysis and 7 visualisations |
| `04_model_optimize.py` | 4 | Delivery-time prediction, validation, tuning and optimisation strategies |
| `results/impact.py` | 2 | Before/after impact of cleaning |
| `data/`, `figures/`, `results/` | | Inputs, generated charts and JSON results |

## How to run
```bash
pip install -r requirements.txt
python 01_generate_data.py   # creates data/logistics_raw.csv first
python 02_clean_preprocess.py
python 03_eda_visualize.py
python 04_model_optimize.py
```
Scripts must be run from this folder in the order above (each uses the previous output).

## Key results (simulated data)
- Cleaning: 6,120 raw rows -> 5,970 clean rows; all missing values, duplicates and impossible values resolved
- Best model: Linear Regression, test MAE 0.52 h, RMSE 0.67 h, R2 0.84 (time-based split, Oct-Dec test)
- Model-based promise times raise on-time delivery from 75.9% to 96.1% on the test quarter

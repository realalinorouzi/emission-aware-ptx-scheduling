# Emission-aware Power-to-X scheduling and sold-product carbon intensity

**Research code accompanying:** Ali Norouzi and Erdal Aydin, *Hourly Carbon Intensity of Sold Hydrogen, Ammonia, and Methanol under Emission-Aware Scheduling*.

This repository provides the computational workflow used to examine how day-ahead electricity-emission signals affect the operation of a grid-connected Power-to-X (PtX) plant and the carbon intensity (CI) of hydrogen (H₂), ammonia (NH₃), and methanol (MeOH) **actually withdrawn for sale**. The analysis links electricity-generation forecasts and emission signals to rolling-horizon scheduling, followed by ex-post carbon accounting through synthesis, labeled inventories, ammonia cracking, and product withdrawal.

The case study covers Spain in 2024: **366 daily scheduling horizons and 8,784 hourly intervals (UTC)**. The manuscript, Supplementary Information (SI), and Appendices A–E describe the mathematical formulation, parameters, accounting conventions, and scientific results.
Please contact for any questions or request for files: realalinorouzi@gmail.com

## Start here

The code supports two complementary routes. **You do not need to retrain the forecasting models or rerun optimization to calculate product CI from the supplied trajectories.**

| Route | Starting point | What you can reproduce | Main entry point |
| --- | --- | --- | --- |
| **A — Scheduling and carbon accounting** | Supplied model-ready 2024 inputs, or archived optimization trajectories | M0/M1/M2 scheduling and/or hourly and annual sold-product CI | `emission_aware_ptx_notebook.ipynb` |
| **B — Complete upstream workflow** | Processed ENTSO-E, MIBGAS, Ember, and GFS data; supplied ANN models where applicable | Generation forecasts, emission signals, renewable availability, model-ready inputs, and then Route A | `notebooks/01_forecasting.ipynb` → `02_emission_signals.ipynb` → `03_renewable_inputs.ipynb` |

For the quickest reproduction of the product-CI calculations, use **Route A with the archived schedules** in the next section. Optimization is disabled by default in the principal notebook.

## Repository structure

| Path | Purpose |
| --- | --- |
| `emission_aware_ptx_notebook.ipynb` | Short, configuration-driven notebook for scheduling, generating M2 reference bounds, and running product-CI accounting. |
| `ptx_optimization.py` | M0 cost baseline, signal-specific M1 emission reference, and emission-capped M2; rolling-horizon state propagation and essential result exports. |
| `product_ci.py` | Ex-post labeled carbon ledger, hourly and annual sold-product CI, and optional in-memory hourly statistics. |
| `notebooks/01_forecasting.ipynb` | Tier 1 ENTSO-E processing and eleven technology-specific ANN forecasting models. |
| `notebooks/02_emission_signals.ipynb` | Tier 2 AEF, UpRamp MEF, and MSDR Soft MEF construction, including the original DLR comparison. |
| `notebooks/03_renewable_inputs.ipynb` | GFS reconstruction, local PV/wind calculations, demand preparation, and assembly of model-ready inputs. |
| `archive/emission_aware_ptx_notebook_original.ipynb` | **Original combined research notebook**, retained as a historical record; not the recommended execution entry point. |
| `Csv files of input and output/` | Supplied datasets, trained ANN files, model-ready inputs, and original optimization outputs. **Do not overwrite these files.** |
| `generated/` | Destination for new results. Created when needed; kept separate from the supplied data. |
| `requirements*.txt` | Separate dependency specifications for core accounting, optimization, and upstream processing. |

The archived notebook preserves the original integrated code, including earlier supplementary and exploratory processing. The reorganized files provide a clearer route for new executions without replacing the original scientific record.

## Route A — Scheduling and carbon accounting

### Requirements

Run the notebooks from the **repository root** (the directory containing `README.md`). The core environment uses Python 3.12:

```bash
python -m pip install -r requirements.txt
jupyter lab
```

Install the optimization dependencies only when solving new MIQCP models:

```bash
python -m pip install -r requirements-optimization.txt
```

Optimization requires **Gurobi 13.0.1 and an appropriate license**. The original study used a 350 s limit per daily solve, a target second-priority MIP gap of 1%, and lexicographic objectives that prioritize product-demand fulfillment. A complete model trajectory involves 366 sequential daily solves. Restricted Gurobi licenses may not support the model size.

### A1. Calculate CI from the supplied optimization results — no solver required

1. Open `emission_aware_ptx_notebook.ipynb` and run its **setup** cell.
2. Leave `RUN_OPTIMIZATION=False` and `RUN_NEW_REFERENCES=False`. You can skip the scheduling cells.
3. Run **Sections 3 and 4** to calculate the hourly carbon ledger and annual CI, followed by optional in-memory hourly statistics.

By default, the notebook reads the archived M0 and the three central M2 trajectories from:

`Csv files of input and output/results/Optimizations/baseline models/`

The emission signals are read from `Csv files of input and output/optimization input-baseline/`. To assess only one signal, set `CI_SIGNALS = ["mef_upramp"]` (or `"aef"` / `"mef_soft"`).

The same calculation can be called directly from Python:

```python
from pathlib import Path
from product_ci import calculate_product_ci

root = Path.cwd()  # repository root
inputs = root / "Csv files of input and output/optimization input-baseline"
base = root / "Csv files of input and output/results/Optimizations/baseline models"

hourly, annual = calculate_product_ci(
    {
        "M0": base / "M0/post/hourly_timeseries_all_days.csv",
        "M2-UpRamp MEF0.5": base / "M2 - mef upramp 0.5/post/hourly_timeseries_all_days.csv",
    },
    inputs,
    root / "generated/ci_upramp",
    signals=["mef_upramp"],
)
```

The accounting writes **two files** to the selected output folder:

- `hourly_carbon_ledger.csv`: hourly electricity/process carbon assignments, labeled stock burdens, sales, and sold-product CI.
- `annual_product_ci.csv`: delivered-product carbon intensities and their differences relative to the matched M0 evaluation.

Choose an unused output directory for each new accounting execution; existing generated outputs are not silently replaced.

### A2. Run M0, M1, and M2 from the supplied inputs

In the principal notebook, set `MODEL` to `"M0"`, `"M1"`, or `"M2"`; choose `SIGNAL` from `"aef"`, `"mef_soft"`, or `"mef_upramp"`; and set `ALPHA`, `N_DAYS`, and `RUN_OPTIMIZATION=True`. `N_DAYS=1` is a January 1 test run; `N_DAYS=366` reproduces the full-year horizon. M0 scheduling does not depend on the chosen emission signal.

For the **unchanged baseline inputs and plant configuration**, M2 can use the supplied daily reference file `M2_S1_emission_bounds_input_file.csv`. If inputs or physical parameters change, solve **M0 → corresponding M1 → regenerate daily references → M2**. M0 and M1 have independent rolling-horizon inventory and commitment histories; their daily emissions are operational references for the M2 cap, rather than guaranteed bounds over one common feasible set.

Example for a complete UpRamp MEF run (this example **launches 1,098 daily solves**):

```python
from pathlib import Path
from ptx_optimization import run_optimization, make_emission_bounds

root = Path.cwd()
inputs = root / "Csv files of input and output/optimization input-baseline"
output = root / "generated/optimization"

m0 = run_optimization("M0", inputs, output)
m1 = run_optimization("M1", inputs, output, signal="mef_upramp")
bounds = make_emission_bounds(
    m0 / "post/daily_summary.csv",
    {"mef_upramp": m1 / "post/daily_summary.csv"},
    root / "generated/upramp_emission_bounds.csv",
)
m2 = run_optimization("M2", inputs, output,
                      signal="mef_upramp", alpha=0.5, bounds=bounds)
```

Each new run has its own timestamped folder under `generated/optimization/`. Its `post/` directory contains:

| Output | Description |
| --- | --- |
| `hourly_timeseries_all_days.csv` | Hourly dispatch, power and process flows, labeled production and withdrawal, opening inventory states, and direct cracking/recovery exports. |
| `daily_summary.csv` | Daily cost, sales, emissions, and other operating totals used also to construct M2 reference bounds. |
| `annual_summary.csv` | Aggregated results for the executed horizon, with a field identifying partial runs. |
| `state_trace.csv` | Closing material, battery, and commitment states carried between consecutive daily horizons. |
| `solver_log_daily.csv` | Daily solver status, duration, and objective information. |

To calculate CI for new solutions, point `MODEL_FILES` in Section 3 of the principal notebook to their `hourly_timeseries_all_days.csv` files. Use the **same model-ready inputs** used for those schedules. For partial runs, set `FULL_YEAR=False` in the CI section. Do not reuse baseline M2 bounds when the inputs or plant configuration have changed.

## Route B — Forecasts, emission signals, and renewable-input preparation

The upstream notebooks retain the code used to generate the model-ready scheduling inputs. Install their dependencies separately:

```bash
python -m pip install -r requirements-upstream.txt
```

This environment includes Keras 3.10.0 (TensorFlow backend), scikit-learn, statsmodels, `entsoe-py`, and supporting scientific packages. The supplied `.keras` files allow ANN inference without retraining; `TRAIN_ANN=True` runs the original training procedures. Source downloads require an ENTSO-E API key only when the supplied processed caches are insufficient.

1. **Tier 1 — `notebooks/01_forecasting.ipynb`:** Prepare chronological ENTSO-E features, scalers, and 24-hour input/24-hour forecast windows. Train or load the eleven technology-specific ANNs and produce the 2024 forecast series. Generated outputs go to `generated/upstream/tier1/`.
2. **Tier 2 — `notebooks/02_emission_signals.ipynb`:** Construct AEF, UpRamp MEF, and MSDR Soft MEF signals. Set `USE_NEW_TIER1` to select newly generated or supplied Tier 1 forecasts. The original UpRamp parameter screening and MSDR/DLR calculations are retained. The selected signals are written to `generated/upstream/signals/`.
3. **Local renewable and input assembly — `notebooks/03_renewable_inputs.ipynb`:** Reconstruct hourly GFS inputs, calculate PV/wind availability, and assemble demands, electricity prices, and emission signals. Set `USE_NEW_SIGNALS` as appropriate. New model-ready inputs are written under `generated/upstream/model_ready/`.
4. **Scheduling and CI:** Point `MODEL_READY` in the principal notebook to these new inputs, solve the required M0/M1/M2 trajectories, and perform the accounting described in Route A. Generate new M2 references for the corresponding inputs.

The original signal selection is **UpRamp `k3_q20`** and **MSDR Soft MEF with `K=2` and a 336-hour rolling window**. The notebooks provide the processing details and retain the existing forecast and signal evaluation routines. Fully repeating the upstream stages may require substantial additional computation.

## Supplied data and results: folder map

All paths in this section are relative to `Csv files of input and output/`. Folder spellings, including `migbas` and `other renawables`, are preserved from the original research files.

| Folder | Contents |
| --- | --- |
| `colab folders data to upload/data_entsoe_es_processed/` | Processed 2022–2024 ENTSO-E hourly generation, load, day-ahead forecasts, cross-border exchanges, and electricity-price CSVs; used as input caches for Tier 1. |
| `colab folders data to upload/ember_co2/` | Annual Ember-derived technology emission-factor tables. |
| `colab folders data to upload/migbas/` | Processed MIBGAS day-ahead gas-price workbook. |
| `gas price-mibgas/` | Original 2022–2024 MIBGAS workbooks and the processed gas-price file. |
| `emission tier2/` | Technology emission-factor mappings and CSVs used by the Tier 2 signals. |
| `results/ML/tier1/` | Eleven trained `.keras` models, 2024 generation forecasts, model-performance CSVs, and evaluation figures. |
| `results/ML/tier2/2.1/` | AEF preparation, forecast/observed evaluation series, and comparison outputs. |
| `results/ML/tier2/2.2/` | UpRamp MEF inputs, configuration sweeps, weight diagnostics, and supporting references. The selected forecast signal comes from `tier2_2_mef_vnew2_outputs/vnew2_k3_q20/`. |
| `results/ML/tier2/2.3/` | MSDR Soft MEF inputs, prior-year history, rolling-window configurations, and archived selected outputs under `v6/`. |
| `results/ML/GFS for local RES/` | Extracted GFS and reconstructed hourly temperature (`t2m`), wind (`u100`, `v100`), and downward shortwave radiation (`dswrf`) CSVs. |
| `optimization input-baseline/` | Final model-ready 2024 CSVs and daily M0/M1 emission references; **start here for scheduling**. |
| `results/Optimizations/baseline models/` | Original M0, three signal-specific M1, and nine M2 trajectories; **start here for CI replay**. |

### Trained ANN models and final forecasts

The following folders are under `results/ML/tier1/`. Each also contains the associated evaluation metrics and figures.

| Technology folder | Trained model | Hourly 2024 forecast |
| --- | --- | --- |
| `coaloil/` | `best_mlp_coal_oil.keras` | `gen_fossil_coal_oil_mw_hourly_predictions_2024.csv` |
| `biomass/` | `best_mlp_biomass.keras` | `gen_biomass_mw_hourly_predictions_2024.csv` |
| `wind/` | `best_mlp_wind_onshore.keras` | `gen_wind_onshore_mw_hourly_predictions_2024.csv` |
| `fossil gas/` | `best_mlp_fossil_gas.keras` | `gen_fossil_gas_mw_hourly_predictions_2024.csv` |
| `nuclear/` | `best_mlp_nuclear.keras` | `gen_nuclear_mw_hourly_predictions_2024.csv` |
| `solar/` | `best_mlp_solar.keras` | `gen_solar_mw_hourly_predictions_2024.csv` |
| `waste/` | `best_mlp_waste.keras` | `gen_waste_mw_hourly_predictions_2024.csv` |
| `other renawables/` | `best_mlp_other_renewables.keras` | `gen_other_renewable_mw_hourly_predictions_2024.csv` |
| `hydro run of river/` | `best_mlp_hydro_run_of_river.keras` | `gen_hydro_run_of_river_mw_hourly_predictions_2024.csv` |
| `other/` | `best_mlp_other.keras` | `gen_other_mw_hourly_predictions_2024.csv` |
| `hydro dispatchable/` | `best_mlp_hydro_dispatchable.keras` | `gen_hydro_dispatchable_mw_hourly_predictions_2024.csv` |

The processed input caches include the 2022, 2023, and 2024 `gen_actual_ES_YYYY.csv`, `load_actual_ES_YYYY.csv`, `da_total_load_ES_YYYY.csv`, `da_gen_total_ES_YYYY.csv`, `da_wind_solar_ES_YYYY.csv`, `da_exchanges_ES_FR_YYYY.csv`, `da_exchanges_ES_PT_YYYY.csv`, and `da_elec_price_ES_YYYY.csv` series. The GFS folders retain their original three-hour source/derived CSVs and reconstructed hourly series; the raw GRIB downloads are not required for Route A.

### Final model-ready inputs

The folder `optimization input-baseline/` contains these files:

| File | Role |
| --- | --- |
| `demand_hourly_2024.csv` | Hourly H₂, NH₃, and MeOH demands (kg/h). |
| `res_hourly_2024.csv` | PV, wind, and total available renewable power (MW). |
| `grid_price_hourly_2024.csv` | Day-ahead electricity price (EUR/MWh). |
| `grid_aef_hourly_2024.csv` | Forecast AEF, `grid_aef_predicted_kg_per_kwh`. |
| `grid_mef_soft_hourly_2024.csv` | Soft MEF, `grid_mef_soft_kg_per_kwh`. |
| `grid_mef_upramp_hourly_2024.csv` | UpRamp MEF, `grid_mef_upramp_pred_kg_per_kwh`. |
| `params_scenario_all.json` | Input-preparation metadata; effective optimization coefficients are specified in the model source. |
| `M2_S1_emission_bounds_input_file.csv` | Signal-specific daily M0/M1 emissions used to construct the baseline M2 caps. |

Hourly inputs use `datetime_utc`. The source emission-factor units are kgCO₂/kWh, with power quantities in MW. The accounting converts MW to kW for hourly electricity-carbon assignment.

### Original optimization outputs

Saved cases are under `results/Optimizations/baseline models/`:

- `M0/`; `M1-aef/`, `M1-mef-soft/`, `M1-mef-upramp/`.
- `M2 - aef 0.2/`, `M2 - aef 0.5/`, `M2 - aef 0.8/`.
- `M2 - mef soft 0.2/`, `M2 - mef soft 0.5/`, `M2 - mef soft 0.8/`.
- `M2 - mef upramp 0.2/`, `M2 - mef upramp 0.5/`, `M2 - mef upramp 0.8/`.

Each case contains a `post/` directory with `hourly_timeseries_all_days.csv`, `daily_summary.csv`, `annual_summary.csv`, `state_trace.csv`, and `solver_log_daily.csv`. Additional original diagnostic exports remain available but are not needed by the simplified CI workflow.

## Carbon-accounting conventions

Carbon accounting is performed **after optimization** and does not modify dispatch decisions. A daily horizon has flow indices 0–23 and inventory-state indices 0–24. Closing material, battery, and operating states carry to the following day; carbon stocks propagate continuously without a midnight reset.

The initial product inventories equal **15% of their respective capacities**, split equally between green and non-green accounting labels. Their **initial assigned carbon burden is zero**; this is not an assumption that the physical tanks are empty. The two labels are separate accounting pools within shared physical storage, with perfect mixing *within each label*. Product withdrawals and internal transfers inherit the opening-pool CI; newly produced or recovered material updates subsequent carbon stocks.

New optimization exports include labeled NH₃ cracker-feed variables (`crk_nh3_in_g_kgph`, `crk_nh3_in_ng_kgph`), recovered streams, relevant H₂ transfers, and inventory boundary states. The CI code uses these direct flows when present; for the archived trajectories, it retains the historical balance-based reconstruction. The original proportional grid/battery electricity attribution and cracker-fuel burden convention are retained.

Hourly sold-product CI equals assigned carbon burden withdrawn during that hour divided by the corresponding sold mass. Hours without meaningful sales have undefined CI. Annual CI uses **total annual sold burden / total annual sold mass**, not the arithmetic mean of hourly CI. Optional distribution and paired-hour comparison functions in `product_ci.py` return DataFrames without creating further files. Paired comparisons use matched positive-sale hours under the same emission signal; a positive or negative CI difference is counted when its magnitude exceeds `1e-12 kgCO2/kg product`.

For the complete model equations and parameter definitions, see the manuscript Methods, SI Sections S1–S2.3 and S7, and Appendices A–E.

## Data sources and citation

The processed datasets originate from the **ENTSO-E Transparency Platform** (electricity-system observations and forecasts), **MIBGAS** (natural-gas prices), **Ember** (technology emission factors), and the **Global Forecast System** (local meteorological inputs). The source references, preprocessing assumptions, and limitations are documented in the accompanying manuscript and SI.

When using this repository, please cite the accompanying manuscript. The publication DOI can be added here when available.

# Context

> **Protocol note:** This file is a live snapshot, not a log. Every entry reflects the *current* state of the project only. When updating, overwrite stale content — do not append a new dated section on top of the old one. If a fact changes, replace it; if a fact becomes irrelevant, delete it.

---

## 1. Project Overview

- **Project name:** `Project-B-Star`
- **Purpose:** An experimental astrodynamics research platform and flight planning engine investigating the predictive ceiling of publicly available tracking data (Two-Line Elements from Space-Track.org) and planetary space-weather indices ($K_p$, $F_{10.7}$) for Low Earth Orbit (LEO, $\le 1,000\text{ km}$) satellite orbital decay. Benchmarks machine learning architectures against simple persistence, provides 1-day ahead decay innovation forecasts ($\log(y_t / y_{t-1})$), and powers an autonomous Model Predictive Control (MPC) mission control console.
- **Current status:** Standard Repository Structure Implemented —
  - **Modular Architecture:** All Python scripts migrated to [`scripts/`](file:///z:/Project-B%20Star/Project-B-Star-main/Scripts), all primary dataset archives to [`datafiles/`](file:///z:/Project-B%20Star/Project-B-Star-main/datafiles), models in [`3 Models/`](file:///z:/Project-B%20Star/Project-B-Star-main/3%20Models/), and diagrams in [`assets/`](file:///z:/Project-B%20Star/Project-B-Star-main/assets).
  - **Zero Path Conflicts:** Centralized [`scripts/path_config.py`](file:///z:/Project-B%20Star/Project-B-Star-main/Scripts/path_config.py) provides dynamic root resolution, ensuring seamless execution whether scripts are invoked from the repository root, within `scripts/`, or by external tooling.
  - **Dual Launch Capability:** Interactive mission control app can be run via `streamlit run app.py` (root launcher) or `streamlit run scripts/app.py`.
- **Entry point(s):**
  ```
  app.py                                — Root Streamlit entrypoint launcher
  scripts/app.py                        — Flight dynamics mission control UI console (Streamlit HUD)
  scripts/train_clean_b_star_model.py   — Clean training & 2025-2026 evaluation pipeline for Model "B *"
  scripts/run_three_decisive_tests.py   — The 3 Decisive Tests audit suite (As-of join, Permutation, Bootstrap)
  scripts/evaluate_forecast_skill.py    — Benchmark against Altitude Mean, Ridge, and Persistence
  scripts/step7_maneuver_optimizer.py   — 7-day MPC autonomous maneuver & attitude optimization solver
  scripts/stratified_model.py           — StratifiedLEOPredictor: 5-seed uncertainty ensemble, SGP4 geometry engine
  datafiles/2014-2025.csv               — Final processed 12-year astrodynamic training dataset (153k rows)
  3 Models/                             — Top 3 model architectures, weights, metadata, and test gap tables
  README.md                             — Primary project documentation and scientific report
  ```

---

## 2. Recent Changes

> Keep this to what's true *right now*. When a change is superseded by a later one, remove the earlier entry rather than stacking both.

- **Change:** Repository Standard Reorganization & Dynamic Path Resolver:
  - **Data Files Consolidation:** Created [`datafiles/`](file:///z:/Project-B%20Star/Project-B-Star-main/datafiles) and moved primary datasets (`2014-2025.csv` and `satellite_data.db`).
  - **Scripts Consolidation:** Moved all operational, simulation, and training modules to [`scripts/`](file:///z:/Project-B%20Star/Project-B-Star-main/Scripts).
  - **Path Conflict Elimination:** Implemented [`scripts/path_config.py`](file:///z:/Project-B%20Star/Project-B-Star-main/Scripts/path_config.py) to dynamically locate `PROJECT_ROOT`, `DATAFILES_DIR`, and `MODELS_DIR`, guaranteeing clean path and module resolution across all scripts.
  - **Root Launcher:** Maintained root [`app.py`](file:///z:/Project-B%20Star/Project-B-Star-main/app.py) as a forwarding entrypoint so `streamlit run app.py` executes without directory changes.
  - **Git & Asset Standardization:** Moved visualization images to [`assets/`](file:///z:/Project-B%20Star/Project-B-Star-main/assets). Updated [`.gitignore`](file:///z:/Project-B%20Star/Project-B-Star-main/.gitignore) to explicitly whitelist `datafiles/2014-2025.csv` while ignoring SQLite databases.
  - **Documentation Sync:** Updated [`README.md`](file:///z:/Project-B%20Star/Project-B-Star-main/README.md) directory structure and quickstart commands to match the new layout.
  **Files:**
  ```
  scripts/path_config.py
  scripts/app.py
  scripts/stratified_model.py
  scripts/train_clean_b_star_model.py
  scripts/run_three_decisive_tests.py
  scripts/evaluate_forecast_skill.py
  scripts/physics_engine.py
  scripts/etl_pipeline.py
  app.py
  datafiles/2014-2025.csv
  datafiles/satellite_data.db
  .gitignore
  README.md
  context.md
  ```
  **Rationale:** Fulfills user request to maintain repository standard by grouping scripts into `scripts/` and data into `datafiles/` with zero path conflicts.

---

## 3. Active Architecture

- **Stack:** Python 3.13, Streamlit ≥1.30, XGBoost ≥2.0, Plotly ≥5.18, SciPy (SLSQP), SGP4 ≥2.22, Skyfield ≥1.46, Pandas, NumPy, scikit-learn, Requests, SQLite3
- **Hardware/Runtime:** CPU/NPU execution (vectorized array operations with NumPy/Pandas).
- **Physical Operating Domain:** Low Earth Orbit (LEO, $\le 1,000\text{ km}$).

### Core Repository Layout
```
Project-B-Star/
├── datafiles/                        # Primary observation archives & datasets
│   ├── 2014-2025.csv                 # Processed 12-year dataset (153k rows, 65 satellites)
│   └── satellite_data.db             # SQLite database: TLE history, space weather, solar flux
├── scripts/                          # Operational, training, and simulation modules
│   ├── app.py                        # Flight dynamics mission control UI console (Streamlit HUD)
│   ├── path_config.py                # Centralized environment-agnostic path resolver
│   ├── train_clean_b_star_model.py   # Clean training & evaluation pipeline for Model "B *"
│   ├── run_three_decisive_tests.py   # Decisive audit: As-of join, 30-run permutation, bootstrap
│   ├── evaluate_forecast_skill.py    # Benchmark against Altitude Mean, Ridge, Persistence
│   ├── step7_maneuver_optimizer.py   # 7-day Model Predictive Control (Vis-Viva, SLSQP solver)
│   ├── stratified_model.py           # Stratified dual-regime predictor & SGP4 geometry engine
│   ├── physics_engine.py             # Vectorized SGP4 orbital propagator & solar beta engine
│   ├── feature_engineering.py        # Astrodynamic & space-weather feature transforms
│   ├── etl_pipeline.py               # Automated ingestion & sync for Space-Track and GFZ archives
│   └── run_golden_epoch_ingestion.py # Raw TLE ingestion driver
├── 3 Models/                         # Top 3 model weights, metadata & gap tables
│   ├── Model_1_Single_Global_XGBoost/
│   ├── Model_2_Stratified_Dual_Regime/
│   └── Model_3_Causal_Log_Ratio_Innovation/
├── assets/                           # Project diagrams & benchmark visualizations
├── .streamlit/                       # Streamlit UI theme configuration
├── app.py                            # Root launcher entrypoint (runs scripts/app.py)
├── requirements.txt                  # Python dependencies
├── LICENSE                           # MIT License
└── README.md                         # Primary project documentation
```

---

## 4. Pending Tasks / Notes

- [x] Create `scripts/` and `datafiles/` folders.
- [x] Move all python scripts into `scripts/` and data files into `datafiles/`.
- [x] Implement `path_config.py` for deterministic path resolution.
- [x] Create root `app.py` launcher forwarding to `scripts/app.py`.
- [x] Verify that scripts resolve data files and model weights from both root and `scripts/` directory.
- [x] Update `README.md`, `.gitignore`, and `context.md`.
- **Repository Readiness:** Clean, organized standard repository layout with verified zero path conflicts.

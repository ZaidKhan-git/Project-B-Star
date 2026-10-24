# AGENT SKILL CONSTITUTION: Satellite Drag Forecasting System

## 1. ROLE & PERSONA
**Role:** Principal Astrodynamics & Machine Learning Engineer.
**Objective:** Architect and deploy an "Early Warning System" for Low Earth Orbit (LEO) satellite decay caused by space weather.
**Operational Environment:** Intel Core Ultra 5 (CPU/NPU-heavy). No CUDA/NVIDIA GPUs available. Code must be optimized for modern CPU vectorization (AVX-512) and memory efficiency.

---

## 2. TECHNICAL STACK & STANDARDS
All generated code must adhere to this specific stack to ensure compatibility and performance.

### Core Libraries
* **Physics:** `sgp4` (specifically `sgp4.api.Satrec` wrapper for C++ speed), `skyfield` (coordinate frames).
* **Data Processing:** `pandas` (strictly vectorized), `numpy` (array operations), `sqlite3` (local caching).
* **Networking:** `requests` (with `Session` and `Retry` logic), `spacetrack` (official client).
* **ML/AI:** `scikit-learn` (Random Forest, XGBoost), `polars` (if pandas exceeds 1GB RAM).

### Coding Standards (The "Golden Rules")
1.  **Type Hinting:** Mandatory for all function signatures (e.g., `def propagate(tle: str) -> float:`).
2.  **Logging > Print:** Never use `print()` for status. Use `logging.info()` for operations and `logging.error()` for failures.
3.  **Modular Design:** Classes must have single responsibilities (Ingestor vs. Propagator vs. Trainer).
4.  **Error Handling:** All external calls (API, DB, Physics) must be wrapped in `try/except` blocks with specific exception catching (not bare `except Exception`).

---

## 3. DOMAIN-SPECIFIC PROCEDURES (The "Playbook")

### A. Orbital Propagation (SGP4)
* **Constraint:** Never implement SGP4 math manually. Always use the library.
* **Workflow:**
    1.  Parse TLE lines $\rightarrow$ `Satrec` object.
    2.  Convert UTC target time $\rightarrow$ Julian Date (`jday`).
    3.  Propagate $\rightarrow$ TEME (True Equator Mean Equinox) coordinates.
    4.  Transform TEME $\rightarrow$ ITRS (WGS84) Geodetic (Lat, Lon, Alt).
* **Optimization:** Use `Satrec.sgp4_array()` when processing >10,000 points to use C++ vectorization instead of Python loops.

### B. Space Weather Correlation
* **The Physics:** Drag lag is **not instantaneous**.
* **Requirement:** Feature engineering **must** include lagged variables.
    * `Kp_Lag_3h` (Immediate Ionosphere response)
    * `Kp_Lag_6h` (Thermosphere heating)
    * `F10.7_81d_Avg` (Solar cycle baseline)
* **Data alignment:** Always merge datasets on `UTC DateTime`, rounded to the nearest hour.

### C. Data Ingestion & Rate Limits
* **Rule:** "Cache-First, Ask-Later."
* **Logic:**
    ```python
    if database.has_data(id):
        return database.load(id)
    else:
        api.fetch(id) # STRICTLY respect 1/lifetime limit for GP_HISTORY
        database.save(id)
    ```

---

## 4. PERFORMANCE OPTIMIZATION (Speed & Simplicity)

### Pattern 1: Vectorization Over Loops (Pandas)
**BAD (Slow Python Loop):**
```python
# Do NOT do this
for index, row in df.iterrows():
    df.at[index, 'drop'] = row['alt_yesterday'] - row['alt_today']
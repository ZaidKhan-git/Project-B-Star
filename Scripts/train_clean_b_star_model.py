"""
train_clean_b_star_model.py
===========================
Clean, leakage-free training of Model "B *" (Multiplicative Causal Innovation Model).
- Dataset: 2014-2024 for training (2014-2023 train, 2024 val for early stopping & calibration)
- Holdout Test: 2025-2026 (Completely unseen, zero peeking)
- Strictly causal as-of join on space weather (timestamp <= prev_epoch, zero overlap)
- Static Ballistic Coefficient calculated strictly on pre-2025 data
- Strong tree regularization (L1, L2, min_child_weight, subsampling) to prevent memorization
- Per-satellite comparison table on 2025-2026 test set showing predictions vs real data
"""

import os
import sys

try:
    from path_config import get_data_file, get_db_path, get_model_path, PROJECT_ROOT, DATAFILES_DIR, MODELS_DIR
except ImportError:
    _cur_dir = os.path.dirname(os.path.abspath(__file__))
    if _cur_dir not in sys.path:
        sys.path.insert(0, _cur_dir)
    from path_config import get_data_file, get_db_path, get_model_path, PROJECT_ROOT, DATAFILES_DIR, MODELS_DIR
import json
import sqlite3
import shutil
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

DB_NAME = get_db_path("satellite_data.db")
OUTPUT_DIR = get_model_path("Model_3_Causal_Log_Ratio_Innovation")
PROD_MODELS_DIR = "models"
RNG_SEED = 42

def rmse(y_true, y_pred=None):
    if y_pred is None:
        return float(np.sqrt(np.mean(np.square(y_true))))
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def main():
    print("=" * 80)
    print("TRAINING CLEAN, LEAK-FREE MODEL 'B *' (2014-2024 TRAIN | 2025-2026 TEST)")
    print("=" * 80)

    # 1. Clean out previous model artifacts
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning out old artifacts in {OUTPUT_DIR}...")
        for f in os.listdir(OUTPUT_DIR):
            f_path = os.path.join(OUTPUT_DIR, f)
            if os.path.isfile(f_path):
                os.remove(f_path)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load space weather & solar flux from database
    print("\n[Step 1/6] Loading & engineering strictly causal space weather (As-Of Join)...")
    conn = sqlite3.connect(DB_NAME)

    sw = pd.read_sql("SELECT time_tag, kp FROM space_weather ORDER BY time_tag", conn)
    sw['time'] = pd.to_datetime(sw['time_tag'], format='ISO8601', utc=True)
    sw = sw.sort_values('time').reset_index(drop=True).drop_duplicates('time')

    sw['Kp_Current'] = sw['kp']
    sw['Kp_roll_24h_mean'] = sw['kp'].rolling(8, min_periods=1).mean()
    sw['Kp_roll_24h_max'] = sw['kp'].rolling(8, min_periods=1).max()
    sw['Kp_roll_72h_mean'] = sw['kp'].rolling(24, min_periods=1).mean()
    sw['Kp_roll_24h_std'] = sw['kp'].rolling(8, min_periods=1).std().fillna(0.0)
    sw['delta_Kp_24h'] = sw['Kp_roll_24h_mean'] - sw['Kp_roll_24h_mean'].shift(8)
    sw['delta_Kp_72h'] = sw['Kp_roll_24h_mean'] - sw['Kp_roll_24h_mean'].shift(24)

    sf = pd.read_sql("SELECT date_tag, f107_obs FROM solar_flux ORDER BY date_tag", conn)
    sf['date'] = pd.to_datetime(sf['date_tag'], utc=True)
    sf = sf.sort_values('date').reset_index(drop=True).drop_duplicates('date')
    # Penticton noon observation is taken at 20:00 UTC
    sf['time'] = sf['date'] + pd.Timedelta(hours=20)
    sf['f107_81d_avg'] = sf['f107_obs'].rolling(81, min_periods=1).mean()
    sf['delta_f107_24h'] = sf['f107_obs'] - sf['f107_obs'].shift(1)
    sf['f107_ratio'] = sf['f107_obs'] / (sf['f107_81d_avg'] + 1e-4)

    # As-of merge solar flux into 3-hourly weather grid
    sw = pd.merge_asof(sw, sf[['time', 'f107_obs', 'f107_81d_avg', 'delta_f107_24h', 'f107_ratio']],
                       on='time', direction='backward')

    # 3. Load TLE history to match exact previous epoch
    print("[Step 2/6] Matching exact TLE previous epoch timestamps (prev_epoch)...")
    tle = pd.read_sql("SELECT norad_id, epoch_datetime FROM tle_history WHERE epoch_datetime >= '2014-01-01' ORDER BY norad_id, epoch_datetime", conn)
    conn.close()

    tle['epoch'] = pd.to_datetime(tle['epoch_datetime'], format='ISO8601', utc=True)
    tle['date_only'] = tle['epoch'].dt.date
    tle = tle.drop_duplicates(subset=['norad_id', 'date_only'], keep='first')
    tle['prev_epoch'] = tle.groupby('norad_id')['epoch'].shift(1)
    tle['prev_date'] = tle.groupby('norad_id')['date_only'].shift(1)
    tle['days_diff'] = (pd.to_datetime(tle['date_only']) - pd.to_datetime(tle['prev_date'])).dt.days

    tle_consec = tle[tle['days_diff'] == 1][['norad_id', 'date_only', 'prev_epoch', 'epoch']].copy()

    # Load geometric dataset
    df_geom = pd.read_csv("training_set_geometric_expanded.csv")
    df_geom['date_dt'] = pd.to_datetime(df_geom['date'])
    df_geom['date_only'] = df_geom['date_dt'].dt.date

    df = pd.merge(df_geom, tle_consec, on=['norad_id', 'date_only'], how='inner')

    # Clean filters: exclude active payload ISS / Tiangong, maneuvers, extreme altitudes
    ACTIVE_PAYLOAD_IDS = [25544, 48274]
    df = df[~df['norad_id'].isin(ACTIVE_PAYLOAD_IDS)]
    if 'is_maneuver' in df.columns:
        df = df[df['is_maneuver'] == False]
    df = df[(df['perigee_alt_km'] >= 200.0) & (df['decay_rate_m'] > 0.01) & (df['decay_rate_m'] < 500.0)]
    df = df.sort_values('prev_epoch').reset_index(drop=True)

    # 4. Anti-cheating: As-Of merge on weather strictly <= prev_epoch
    weather_cols_sw = [
        'Kp_Current', 'Kp_roll_24h_mean', 'Kp_roll_24h_max', 'Kp_roll_72h_mean', 'Kp_roll_24h_std',
        'delta_Kp_24h', 'delta_Kp_72h',
        'f107_obs', 'f107_81d_avg', 'delta_f107_24h', 'f107_ratio'
    ]
    old_cols = [c for c in df.columns if c in weather_cols_sw or 'Kp_' in c or 'f107_' in c]
    df = df.drop(columns=old_cols)

    asof_df = pd.merge_asof(df, sw[['time'] + weather_cols_sw],
                            left_on='prev_epoch', right_on='time',
                            direction='backward')

    # 5. Anti-cheating: Compute static Ballistic Coefficient (B*) STRICTLY on pre-2025 training data
    print("[Step 3/6] Computing Ballistic Coefficients strictly on pre-2025 training data...")
    asof_df['date'] = pd.to_datetime(asof_df['date'])
    pre_2025_mask = asof_df['date'] < '2025-01-01'

    asof_df['density_proxy'] = (
        asof_df['f107_obs'] * asof_df['Kp_roll_24h_mean'] /
        (asof_df['altitude_km'] ** 3)
    )
    asof_df['obs_bc'] = asof_df['decay_rate_m'] / np.maximum(1e-12, asof_df['density_proxy'])

    # Median B* computed ONLY on pre-2025 rows!
    train_bc_sub = asof_df[pre_2025_mask & (asof_df['obs_bc'] > 0) & (asof_df['obs_bc'] < 1e10)]
    bc_by_sat = train_bc_sub.groupby('norad_id')['obs_bc'].median().to_dict()
    global_median_bc = float(train_bc_sub['obs_bc'].median())

    # Map pre-2025 B* to all rows (no future peeking)
    asof_df['static_bc_est'] = asof_df['norad_id'].map(bc_by_sat).fillna(global_median_bc)

    # 6. Reindex to 1-day calendar daily grid to guarantee causal lag structure
    print("[Step 4/6] Reindexing to daily calendar grid & generating strictly past features...")
    asof_df = asof_df.sort_values(['norad_id', 'date']).reset_index(drop=True)

    grid_list = []
    for nid, g in asof_df.groupby('norad_id'):
        sat_name = g['satellite_name'].iloc[0] if 'satellite_name' in g.columns else str(nid)
        g_reindexed = g.set_index('date').resample('1D').asfreq()
        g_reindexed['norad_id'] = nid
        g_reindexed['satellite_name'] = sat_name
        for c in ['semi_major_axis_km', 'altitude_km', 'perigee_alt_km', 'eccentricity', 'inclination_deg', 'static_bc_est']:
            if c in g_reindexed.columns:
                g_reindexed[c] = g_reindexed[c].ffill()
        grid_list.append(g_reindexed)

    full = pd.concat(grid_list).reset_index()

    # Past decay history strictly from t-1 or earlier
    full['y_t1'] = full.groupby('norad_id')['decay_rate_m'].shift(1)
    full['y_t2'] = full.groupby('norad_id')['decay_rate_m'].shift(2)
    full['y_t3'] = full.groupby('norad_id')['decay_rate_m'].shift(3)
    full['y_t7'] = full.groupby('norad_id')['decay_rate_m'].shift(7)

    full['decay_roll_3d_mean'] = full.groupby('norad_id')['decay_rate_m'].apply(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    ).reset_index(level=0, drop=True)

    full['decay_roll_7d_mean'] = full.groupby('norad_id')['decay_rate_m'].apply(
        lambda s: s.shift(1).rolling(7, min_periods=2).mean()
    ).reset_index(level=0, drop=True)

    full['decay_roll_7d_std'] = full.groupby('norad_id')['decay_rate_m'].apply(
        lambda s: s.shift(1).rolling(7, min_periods=2).std()
    ).reset_index(level=0, drop=True).fillna(0.0)

    full['decay_momentum_1d'] = full['y_t1'] - full['y_t2']
    full['decay_ratio_1d_7d'] = full['y_t1'] / (full['decay_roll_7d_mean'] + 1e-4)

    # Orbital / geometry at t-1
    full['alt_t1'] = full.groupby('norad_id')['altitude_km'].shift(1)
    full['perigee_t1'] = full.groupby('norad_id')['perigee_alt_km'].shift(1)
    full['ecc_t1'] = full.groupby('norad_id')['eccentricity'].shift(1)
    full['inc_t1'] = full.groupby('norad_id')['inclination_deg'].shift(1)
    full['bc_t1'] = full.groupby('norad_id')['static_bc_est'].shift(1)
    full['sef_t1'] = full.groupby('norad_id')['sun_exposure_factor'].shift(1)
    full['beta_t1'] = full.groupby('norad_id')['beta_angle_deg'].shift(1)

    full['sin_doy'] = np.sin(2 * np.pi * full['date'].dt.dayofyear / 365.25)
    full['cos_doy'] = np.cos(2 * np.pi * full['date'].dt.dayofyear / 365.25)

    # Physical interaction terms: dynamic drag expansion proportional to past decay level
    full['interact_drag_delta_Kp'] = full['y_t1'] * full['delta_Kp_24h']
    full['interact_drag_Kp'] = full['y_t1'] * np.maximum(0.0, full['Kp_roll_24h_mean'] - 2.0)

    # Target: Multiplicative Log-Ratio Innovation
    full['target_log_ratio'] = np.log(full['decay_rate_m'] / full['y_t1'])

    HISTORY_FEATURES = [
        'y_t1', 'y_t2', 'y_t3', 'y_t7',
        'decay_roll_3d_mean', 'decay_roll_7d_mean', 'decay_roll_7d_std',
        'decay_momentum_1d', 'decay_ratio_1d_7d',
        'alt_t1', 'perigee_t1', 'ecc_t1', 'inc_t1', 'bc_t1',
        'sef_t1', 'beta_t1', 'sin_doy', 'cos_doy'
    ]

    WEATHER_FEATURES = [
        'Kp_Current', 'Kp_roll_24h_mean', 'Kp_roll_24h_max', 'Kp_roll_72h_mean', 'Kp_roll_24h_std',
        'delta_Kp_24h', 'delta_Kp_72h',
        'f107_obs', 'f107_81d_avg', 'delta_f107_24h', 'f107_ratio',
        'interact_drag_delta_Kp', 'interact_drag_Kp'
    ]

    ALL_FEATURES = HISTORY_FEATURES + WEATHER_FEATURES

    clean = full.dropna(subset=['decay_rate_m', 'y_t1', 'target_log_ratio'] + ALL_FEATURES).copy()
    clean = clean.sort_values(['norad_id', 'date']).reset_index(drop=True)

    # 7. Strictly enforced temporal splits
    # Train: 2014-2023 (10 years)
    # Val: 2024 (1 year, inside training budget, used for early stopping and scalar calibration)
    # Test: 2025-2026 (2 years, completely unseen holdout)
    train_df = clean[clean['date'] < '2024-01-01'].copy()
    val_df = clean[(clean['date'] >= '2024-01-01') & (clean['date'] < '2025-01-01')].copy()
    test_df = clean[clean['date'] >= '2025-01-01'].copy()

    print(f"\n[Step 5/6] Final Split Breakdown:")
    print(f"  Training Set   (2014-2023): {len(train_df):,} rows across {train_df['norad_id'].nunique()} satellites")
    print(f"  Validation Set (2024):      {len(val_df):,} rows across {val_df['norad_id'].nunique()} satellites (for early stopping)")
    print(f"  Test Set       (2025-2026): {len(test_df):,} rows across {test_df['norad_id'].nunique()} satellites (Completely Unseen)")

    # 8. Train Model "B *" with strong anti-memorization regularization
    print("\n[Step 6/6] Fitting Model 'B *' with anti-memorization regularization...")
    xgb_params = {
        'objective': 'reg:squarederror',
        'n_estimators': 2000,
        'learning_rate': 0.015,
        'max_depth': 5,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'reg_alpha': 0.2,       # L1 regularization
        'reg_lambda': 2.0,      # L2 regularization
        'min_child_weight': 15, # Minimum sample weight in a leaf node
        'early_stopping_rounds': 80,
        'n_jobs': 4,
        'random_state': RNG_SEED,
        'verbosity': 0
    }

    b_star_model = XGBRegressor(**xgb_params)
    b_star_model.fit(
        train_df[ALL_FEATURES], train_df['target_log_ratio'],
        eval_set=[(val_df[ALL_FEATURES], val_df['target_log_ratio'])],
        verbose=False
    )

    print(f"  Training stopped at iteration: {b_star_model.best_iteration}")

    # Scalar calibration strictly on 2024 validation data
    def fit_scalar(y_true, y_pred):
        denom = np.sum(y_pred ** 2)
        return float(np.sum(y_true * y_pred) / denom) if denom > 1e-9 else 1.0

    val_pred_raw = b_star_model.predict(val_df[ALL_FEATURES])
    cal_scalar = fit_scalar(val_df['target_log_ratio'], val_pred_raw)
    print(f"  Validation Calibration Scalar (fitted on 2024): {cal_scalar:.4f}")

    # Save model weights & metadata
    model_file_target = os.path.join(OUTPUT_DIR, "b_star_model.json")
    b_star_model.save_model(model_file_target)

    # Also copy to root models/ directory for system availability
    if os.path.exists(PROD_MODELS_DIR):
        shutil.copy(model_file_target, os.path.join(PROD_MODELS_DIR, "b_star_model.json"))

    meta = {
        "model_name": "B *",
        "description": "Multiplicative Causal Log-Ratio Innovation Model",
        "training_period": "2014-01-01 to 2024-12-31 (Train: 2014-2023, Val: 2024)",
        "test_period": "2025-01-01 to 2026-10-19",
        "best_iteration": int(b_star_model.best_iteration),
        "calibration_scalar": float(cal_scalar),
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "unique_satellites_train": int(train_df['norad_id'].nunique()),
        "unique_satellites_test": int(test_df['norad_id'].nunique()),
        "features": ALL_FEATURES,
        "anti_cheating_guarantees": [
            "Strict As-Of join on weather: timestamp <= prev_epoch (zero lookahead)",
            "Decay history lags strictly up to t-1",
            "Static B* calculated strictly on pre-2025 training data",
            "Early stopping and calibration fitted on 2024, zero peeking into 2025-2026",
            "L1 and L2 regularization to prevent memorization"
        ]
    }
    with open(os.path.join(OUTPUT_DIR, "b_star_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 9. Evaluate on completely unseen 2025-2026 Test Set
    print("\n" + "=" * 80)
    print("EVALUATION ON UNSEEN 2025-2026 TEST SET")
    print("=" * 80)

    test_df['y_pers'] = test_df['y_t1']
    test_pred_ratio = b_star_model.predict(test_df[ALL_FEATURES])
    test_df['y_pred'] = np.maximum(0.01, test_df['y_t1'] * np.exp(cal_scalar * test_pred_ratio))

    test_df['abs_err_model'] = np.abs(test_df['decay_rate_m'] - test_df['y_pred'])
    test_df['abs_err_pers'] = np.abs(test_df['decay_rate_m'] - test_df['y_pers'])

    overall_rmse_model = rmse(test_df['decay_rate_m'], test_df['y_pred'])
    overall_rmse_pers = rmse(test_df['decay_rate_m'], test_df['y_pers'])
    overall_skill = 1.0 - overall_rmse_model / overall_rmse_pers
    print(f"Overall Test Set (2025-2026, {len(test_df):,} rows across {test_df['norad_id'].nunique()} sats):")
    print(f"  Model 'B *' Pooled RMSE:     {overall_rmse_model:.3f} m/day")
    print(f"  1-Day Persistence RMSE:      {overall_rmse_pers:.3f} m/day")
    print(f"  Pooled Forecast Skill Score: {overall_skill:+.3f}")

    # Build per-satellite comparison table
    sat_summary = []
    for nid in sorted(test_df['norad_id'].unique()):
        sub = test_df[test_df['norad_id'] == nid]
        if len(sub) < 15:
            continue
        sat_name = sub['satellite_name'].iloc[0] if 'satellite_name' in sub.columns else str(nid)
        mean_alt = float(sub['altitude_km'].mean())
        regime = "Deep LEO" if mean_alt < 550.0 else "Upper LEO"

        y_actual = sub['decay_rate_m'].values
        y_model = sub['y_pred'].values
        y_p = sub['y_pers'].values

        r_m = rmse(y_actual, y_model)
        r_p = rmse(y_actual, y_p)
        mae_m = float(np.mean(np.abs(y_actual - y_model)))
        bias_gap = float(np.mean(y_model - y_actual))
        skill = 1.0 - r_m / r_p

        sat_summary.append({
            'norad_id': int(nid),
            'satellite_name': sat_name,
            'altitude_km': round(mean_alt, 1),
            'regime': regime,
            'test_days_2025_2026': len(sub),
            'actual_mean_m_day': round(float(np.mean(y_actual)), 2),
            'predicted_mean_m_day': round(float(np.mean(y_model)), 2),
            'gap_bias_m_day': round(bias_gap, 2),
            'model_mae_m_day': round(mae_m, 2),
            'model_rmse_m_day': round(r_m, 2),
            'pers_rmse_m_day': round(r_p, 2),
            'skill_vs_pers': round(skill, 3),
            'beats_persistence': "YES" if skill > 0 else "NO"
        })

    sat_table_df = pd.DataFrame(sat_summary)
    csv_table_path = os.path.join(OUTPUT_DIR, "b_star_test_vs_actual_gap_2025_2026.csv")
    sat_table_df.to_csv(csv_table_path, index=False)

    print(f"\nSaved Per-Satellite Comparison Table to: {csv_table_path}")
    print(f"Total evaluated satellites with >=15 test days: {len(sat_table_df)}")
    win_rate = float(np.mean(sat_table_df['skill_vs_pers'] > 0) * 100)
    median_skill = float(sat_table_df['skill_vs_pers'].median())
    print(f"Median Skill vs Persistence across Satellites: {median_skill:+.3f}")
    print(f"Satellite Win Rate vs Persistence:              {win_rate:.1f}% ({int(np.sum(sat_table_df['skill_vs_pers'] > 0))}/{len(sat_table_df)})")

    print("\nComparison Table Preview:")
    print(sat_table_df[['norad_id', 'satellite_name', 'altitude_km', 'regime', 'test_days_2025_2026',
                        'actual_mean_m_day', 'predicted_mean_m_day', 'gap_bias_m_day',
                        'model_rmse_m_day', 'pers_rmse_m_day', 'skill_vs_pers', 'beats_persistence']].to_string())

if __name__ == "__main__":
    main()

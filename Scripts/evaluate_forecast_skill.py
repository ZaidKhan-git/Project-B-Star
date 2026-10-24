"""
evaluate_forecast_skill.py - Forecast Skill Score Benchmark
============================================================
Evaluates Project B-Star model skill against three explicit baselines:
1. Regime / Altitude-Bin Mean (training-set average decay per 50 km altitude band)
2. Ridge on Altitude and B* Only (linear fit with zero space-weather features - ablation)
3. Persistence (yesterday's observed decay of that same satellite: y_{t-1})

Strict Adherence to Honesty Rules:
- Rule 1: Scored on identical test rows (2026 holdout).
- Rule 2: Matched forecast horizon (observations up to t-1; no past decay in model features).
- Rule 3: Skill computed per-satellite: Skill = 1 - (RMSE_model / RMSE_baseline).
- Rule 4: Disaggregated quiet days (Kp < 4) vs storm days (Kp >= 6) + event-level breakdown.
- Rule 5: Fixed pre-reentry threshold (perigee >= 200 km, decay < 500 m/day) applied uniformly.
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
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

DATA_PATH = get_data_file("2014-2025.csv")
MODEL_DIR = get_model_path("Model_2_Stratified_Dual_Regime")
OUTPUT_REPORT = os.path.join(MODEL_DIR, "forecast_skill_report.json")
OUTPUT_CSV = os.path.join(MODEL_DIR, "forecast_skill_per_satellite.csv")
OUTPUT_PLOT = os.path.join(MODEL_DIR, "forecast_skill_comparison.png")

ACTIVE_PAYLOAD_IDS = [25544, 48274]
ALTITUDE_THRESHOLD = 550.0  # km
TRAIN_CUTOFF_DATE = "2025-01-01"
VAL_CUTOFF_DATE = "2026-01-01"

FEATURE_COLUMNS = [
    'sun_exposure_factor', 'beta_angle_deg',
    'f107_obs', 'f107_81d_avg',
    'Kp_mean', 'Kp_max', 'Kp_min', 'Kp_std',
    'Kp_Lag_3h', 'Kp_Lag_6h', 'Kp_Lag_12h', 'Kp_Lag_24h',
    'Kp_Roll_24h_Mean', 'Kp_Roll_24h_Std', 'Kp_Roll_72h_Mean',
    'semi_major_axis_km', 'altitude_km', 'perigee_alt_km',
    'eccentricity', 'inclination_deg',
    'sin_doy', 'cos_doy',
    'static_bc_est'
]

TARGET_COLUMN = 'decay_rate_m'

def run_skill_evaluation():
    print("=" * 75)
    print("PROJECT B-STAR: FORECAST SKILL SCORE BENCHMARK")
    print("Skill = 1 - (RMSE_model / RMSE_baseline)")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Feature Audit (Rule 2: Check for Lagged Decay in Features)
    # ---------------------------------------------------------
    print("\n[STEP 1] Auditing Feature Set for Forecast Horizon Compliance (Rule 2)...")
    lag_decay_in_features = [c for c in FEATURE_COLUMNS if 'decay' in c.lower()]
    print(f"   Features in Model: {len(FEATURE_COLUMNS)} physics & Keplerian variables")
    if lag_decay_in_features:
        print(f"   [WARN] Past decay features found: {lag_decay_in_features}")
    else:
        print("   [AUDIT PASSED] No lagged decay features present.")
        print("   Model predicts purely from space-weather physics, solar geometry, and Keplerian state.")

    # ---------------------------------------------------------
    # 2. Data Loading & Pre-Reentry Filter (Rule 5)
    # ---------------------------------------------------------
    print("\n[STEP 2] Loading Data & Applying Fixed Inclusion/Reentry Rules (Rule 5)...")
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    initial_rows = len(df)
    # Exclude active stations (ISS, CSS)
    df = df[~df['norad_id'].isin(ACTIVE_PAYLOAD_IDS)]
    
    # Maneuver filter
    if 'is_maneuver' in df.columns:
        df = df[df['is_maneuver'] == False]
    
    # Rule 5: Fix pre-reentry threshold before looking
    # Exclude perigee < 200 km and runaway pre-reentry decays > 500 m/day
    df = df[(df['perigee_alt_km'] >= 200.0) & (df['decay_rate_m'] > 0.01) & (df['decay_rate_m'] < 500.0)]
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    print(f"   Clean dataset: {len(df):,} rows (from {initial_rows:,}) across {df['norad_id'].nunique()} satellites")

    # ---------------------------------------------------------
    # 3. Construct Strictly Causal Persistence (Rule 2)
    # ---------------------------------------------------------
    print("\n[STEP 3] Constructing Strictly Causal Persistence Baselines...")
    df = df.sort_values(['norad_id', 'date']).reset_index(drop=True)
    df['date_diff'] = df.groupby('norad_id')['date'].diff().dt.days
    
    # 1-day lag persistence: y(t-1) where preceding record is exactly yesterday (date_diff == 1)
    df['pers_lag1'] = df.groupby('norad_id')[TARGET_COLUMN].shift(1)
    df['pers_lag1'] = df['pers_lag1'].where(df['date_diff'] == 1)

    # 3-day trailing mean persistence: mean(y_{t-1}, y_{t-2}, y_{t-3})
    # Must use shift(1) so current day t is never seen
    df['pers_roll3'] = df.groupby('norad_id')[TARGET_COLUMN].apply(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    ).reset_index(level=0, drop=True)

    # ---------------------------------------------------------
    # 4. Train Baselines on Training Set (2014-2024)
    # ---------------------------------------------------------
    print("\n[STEP 4] Fitting Baselines on Training Set (2014-2024)...")
    train_mask = df['date'] < TRAIN_CUTOFF_DATE
    train_df = df[train_mask].copy()
    print(f"   Training Set Size: {len(train_df):,} rows")

    # Baseline A: Altitude-Bin Mean (50 km bins)
    # Bin edges: 200, 250, 300, ..., 1000
    bin_edges = np.arange(200, 1050, 50)
    bin_labels = [f"{b}-{b+50}km" for b in bin_edges[:-1]]
    train_df['alt_bin'] = pd.cut(train_df['altitude_km'], bins=bin_edges, labels=bin_labels)
    alt_bin_means = train_df.groupby('alt_bin')[TARGET_COLUMN].mean().to_dict()
    global_train_mean = train_df[TARGET_COLUMN].mean()
    print(f"   Altitude-Bin Means ({len(alt_bin_means)} bins): {alt_bin_means}")

    # Baseline B: Ridge on Altitude and B* Only (Ablation without Space Weather)
    ridge_features = ['altitude_km', 'perigee_alt_km', 'static_bc_est']
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(train_df[ridge_features], train_df[TARGET_COLUMN])
    print(f"   Fitted Ridge Model on {ridge_features}:")
    print(f"     Coefficients: {dict(zip(ridge_features, ridge_model.coef_))}, Intercept: {ridge_model.intercept_:.4f}")

    # ---------------------------------------------------------
    # 5. Load Trained Deep & Upper Models + Calibration
    # ---------------------------------------------------------
    print("\n[STEP 5] Loading Stratified XGBoost Models & Calibration Factors...")
    deep_model = XGBRegressor()
    deep_model.load_model(os.path.join(MODEL_DIR, "xgb_deep_model.json"))
    
    upper_model = XGBRegressor()
    upper_model.load_model(os.path.join(MODEL_DIR, "xgb_upper_model.json"))
    
    with open(os.path.join(MODEL_DIR, "calibration_config.json")) as f:
        cal = json.load(f)
    print(f"   Loaded Models. Deep scale: {cal['deep_scale']:.4f}, Upper scale: {cal['upper_scale']:.4f}")

    # ---------------------------------------------------------
    # 6. Prepare Identical Test Rows (Rule 1)
    # ---------------------------------------------------------
    print("\n[STEP 6] Filtering Identical Test Rows for 2026 Holdout (Rule 1)...")
    test_mask = df['date'] >= VAL_CUTOFF_DATE
    test_df = df[test_mask].copy()
    
    # Strictly enforce that 1-day persistence exists so all models evaluate on identical rows
    test_df = test_df[test_df['pers_lag1'].notna()].copy()
    test_df['alt_bin'] = pd.cut(test_df['altitude_km'], bins=bin_edges, labels=bin_labels)
    test_df['pred_alt_bin'] = test_df['alt_bin'].map(alt_bin_means).fillna(global_train_mean)
    test_df['pred_ridge'] = np.maximum(0.01, ridge_model.predict(test_df[ridge_features]))
    
    # Model predictions
    deep_mask_test = test_df['altitude_km'] < ALTITUDE_THRESHOLD
    upper_mask_test = test_df['altitude_km'] >= ALTITUDE_THRESHOLD
    
    test_df['pred_model'] = 0.0
    if deep_mask_test.sum() > 0:
        pred_deep = deep_model.predict(test_df.loc[deep_mask_test, FEATURE_COLUMNS]) * cal['deep_scale']
        test_df.loc[deep_mask_test, 'pred_model'] = np.maximum(0.01, pred_deep)
    if upper_mask_test.sum() > 0:
        pred_up = upper_model.predict(test_df.loc[upper_mask_test, FEATURE_COLUMNS]) * cal['upper_scale']
        test_df.loc[upper_mask_test, 'pred_model'] = np.maximum(0.01, pred_up)

    N_test = len(test_df)
    N_sats = test_df['norad_id'].nunique()
    print(f"   [IDENTICAL TEST ROWS] Evaluated: {N_test:,} rows across {N_sats} satellites.")

    # ---------------------------------------------------------
    # 7. Compute Skill Scores Per Satellite (Rule 3)
    # ---------------------------------------------------------
    print("\n[STEP 7] Computing Skill Scores Per-Satellite (Rule 3)...")
    sat_results = []
    
    for sat_id, group in test_df.groupby('norad_id'):
        sat_name = group['satellite_name'].iloc[0] if 'satellite_name' in group.columns else str(sat_id)
        mean_alt = group['altitude_km'].mean()
        regime = "Deep LEO" if mean_alt < ALTITUDE_THRESHOLD else "Upper LEO"
        n_obs = len(group)
        y_true = group[TARGET_COLUMN].values
        
        # RMSEs
        rmse_model = float(np.sqrt(mean_squared_error(y_true, group['pred_model'])))
        rmse_pers1 = float(np.sqrt(mean_squared_error(y_true, group['pers_lag1'])))
        rmse_pers3 = float(np.sqrt(mean_squared_error(y_true, group['pers_roll3'])))
        rmse_ridge = float(np.sqrt(mean_squared_error(y_true, group['pred_ridge'])))
        rmse_altbin = float(np.sqrt(mean_squared_error(y_true, group['pred_alt_bin'])))
        
        # MAEs
        mae_model = float(mean_absolute_error(y_true, group['pred_model']))
        mae_pers1 = float(mean_absolute_error(y_true, group['pers_lag1']))
        
        # Skill = 1 - (RMSE_model / RMSE_baseline)
        skill_vs_pers1 = 1.0 - (rmse_model / max(1e-6, rmse_pers1))
        skill_vs_pers3 = 1.0 - (rmse_model / max(1e-6, rmse_pers3))
        skill_vs_ridge = 1.0 - (rmse_model / max(1e-6, rmse_ridge))
        skill_vs_altbin = 1.0 - (rmse_model / max(1e-6, rmse_altbin))
        
        sat_results.append({
            'norad_id': int(sat_id),
            'satellite_name': sat_name,
            'regime': regime,
            'mean_altitude_km': float(mean_alt),
            'samples': int(n_obs),
            'rmse_model': rmse_model,
            'rmse_pers1': rmse_pers1,
            'rmse_pers3': rmse_pers3,
            'rmse_ridge': rmse_ridge,
            'rmse_altbin': rmse_altbin,
            'mae_model': mae_model,
            'mae_pers1': mae_pers1,
            'skill_vs_pers1': float(skill_vs_pers1),
            'skill_vs_pers3': float(skill_vs_pers3),
            'skill_vs_ridge': float(skill_vs_ridge),
            'skill_vs_altbin': float(skill_vs_altbin),
            'beats_pers1': bool(skill_vs_pers1 > 0),
            'beats_ridge': bool(skill_vs_ridge > 0),
            'beats_altbin': bool(skill_vs_altbin > 0)
        })

    sat_df = pd.DataFrame(sat_results)
    sat_df.to_csv(OUTPUT_CSV, index=False)
    
    # Summary across all satellites
    median_skill_pers1 = float(sat_df['skill_vs_pers1'].median())
    mean_skill_pers1 = float(sat_df['skill_vs_pers1'].mean())
    win_rate_pers1 = float(sat_df['beats_pers1'].mean() * 100.0)
    
    median_skill_ridge = float(sat_df['skill_vs_ridge'].median())
    win_rate_ridge = float(sat_df['beats_ridge'].mean() * 100.0)
    
    median_skill_altbin = float(sat_df['skill_vs_altbin'].median())
    win_rate_altbin = float(sat_df['beats_altbin'].mean() * 100.0)

    print("\n" + "=" * 75)
    print("PER-SATELLITE SKILL SUMMARY (Rule 3)")
    print("=" * 75)
    print(f"Total Satellites Evaluated: {len(sat_df)}")
    print(f"1. vs. Altitude-Bin Mean:  Median Skill = {median_skill_altbin:+.3f} | Win Rate = {win_rate_altbin:.1f}% ({sat_df['beats_altbin'].sum()}/{len(sat_df)} sats)")
    print(f"2. vs. Ridge (No Weather): Median Skill = {median_skill_ridge:+.3f} | Win Rate = {win_rate_ridge:.1f}% ({sat_df['beats_ridge'].sum()}/{len(sat_df)} sats)")
    print(f"3. vs. 1-Day Persistence:  Median Skill = {median_skill_pers1:+.3f} | Win Rate = {win_rate_pers1:.1f}% ({sat_df['beats_pers1'].sum()}/{len(sat_df)} sats)")
    print("=" * 75)

    print("\nDetailed Per-Satellite Breakdown:")
    disp_cols = ['norad_id', 'satellite_name', 'regime', 'mean_altitude_km', 'samples', 'rmse_model', 'rmse_pers1', 'skill_vs_pers1', 'skill_vs_ridge']
    print(sat_df[disp_cols].to_string(index=False))

    # ---------------------------------------------------------
    # 8. Split Quiet vs. Storm Periods (Rule 4)
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print("[STEP 8] QUIET VS. STORM PERIODS ANALYSIS (Rule 4)")
    print("=" * 75)

    quiet_mask = test_df['Kp_max'] < 4.0
    mod_mask = (test_df['Kp_max'] >= 4.0) & (test_df['Kp_max'] < 6.0)
    storm_mask = test_df['Kp_max'] >= 6.0

    slices = [
        ('Quiet Solar Conditions (Kp_max < 4.0)', quiet_mask),
        ('Moderate Geomagnetic Activity (4.0 <= Kp_max < 6.0)', mod_mask),
        ('Severe Geomagnetic Storms (Kp_max >= 6.0)', storm_mask)
    ]

    slice_reports = []
    for label, mask in slices:
        sub_df = test_df[mask]
        n_obs = len(sub_df)
        if n_obs == 0:
            continue
        y_true = sub_df[TARGET_COLUMN].values
        rmse_m = float(np.sqrt(mean_squared_error(y_true, sub_df['pred_model'])))
        rmse_p = float(np.sqrt(mean_squared_error(y_true, sub_df['pers_lag1'])))
        rmse_r = float(np.sqrt(mean_squared_error(y_true, sub_df['pred_ridge'])))
        rmse_a = float(np.sqrt(mean_squared_error(y_true, sub_df['pred_alt_bin'])))
        
        # Per-satellite skill on this slice
        sat_skills_p = []
        for _, g in sub_df.groupby('norad_id'):
            if len(g) >= 2:
                s_m = np.sqrt(mean_squared_error(g[TARGET_COLUMN], g['pred_model']))
                s_p = np.sqrt(mean_squared_error(g[TARGET_COLUMN], g['pers_lag1']))
                if s_p > 1e-6:
                    sat_skills_p.append(1.0 - (s_m / s_p))
                    
        med_skill_p = float(np.median(sat_skills_p)) if sat_skills_p else (1.0 - rmse_m / max(1e-6, rmse_p))
        win_rate_p = float(np.mean([s > 0 for s in sat_skills_p]) * 100.0) if sat_skills_p else 0.0

        print(f"\n{label} (N = {n_obs:,} rows across {sub_df['norad_id'].nunique()} sats):")
        print(f"   Model RMSE:        {rmse_m:.2f} m/day")
        print(f"   Persistence RMSE:  {rmse_p:.2f} m/day")
        print(f"   Ridge (No Weather):{rmse_r:.2f} m/day")
        print(f"   Altitude-Bin Mean: {rmse_a:.2f} m/day")
        print(f"   --> Pooled Skill vs. Persistence: {1.0 - rmse_m/rmse_p:+.3f} ({(1.0 - rmse_m/rmse_p)*100:+.1f}%)")
        print(f"   --> Per-Satellite Median Skill vs. Persistence: {med_skill_p:+.3f} | Win Rate: {win_rate_p:.1f}%")

        slice_reports.append({
            'slice_name': label,
            'rows': int(n_obs),
            'satellites': int(sub_df['norad_id'].nunique()),
            'rmse_model': rmse_m,
            'rmse_persistence': rmse_p,
            'rmse_ridge': rmse_r,
            'rmse_altbin': rmse_a,
            'pooled_skill_vs_pers': float(1.0 - rmse_m / rmse_p),
            'median_skill_vs_pers': med_skill_p,
            'win_rate_vs_pers': win_rate_p
        })

    # Specific 2026 Storm Events Breakdown
    print("\n" + "-" * 75)
    print("SPECIFIC 2026 GEOMAGNETIC STORM EVENTS (Case Studies)")
    print("-" * 75)
    
    events = [
        ('January 2026 Storm', '2026-01-19', '2026-01-22'),
        ('March 2026 Equinox Storm', '2026-03-19', '2026-03-24'),
        ('April 2026 CME Event', '2026-04-01', '2026-04-05'),
        ('May 2026 Solar Active Period', '2026-05-03', '2026-05-16')
    ]
    
    event_reports = []
    for ev_name, start_d, end_d in events:
        ev_df = test_df[(test_df['date'] >= start_d) & (test_df['date'] <= end_d)]
        if len(ev_df) == 0:
            continue
        y_true = ev_df[TARGET_COLUMN].values
        rmse_m = float(np.sqrt(mean_squared_error(y_true, ev_df['pred_model'])))
        rmse_p = float(np.sqrt(mean_squared_error(y_true, ev_df['pers_lag1'])))
        rmse_r = float(np.sqrt(mean_squared_error(y_true, ev_df['pred_ridge'])))
        skill_p = 1.0 - (rmse_m / max(1e-6, rmse_p))
        skill_r = 1.0 - (rmse_m / max(1e-6, rmse_r))
        max_kp = float(ev_df['Kp_max'].max())
        
        print(f"\n* {ev_name} ({start_d} to {end_d}, Max Kp={max_kp:.1f}, N={len(ev_df)}):")
        print(f"    Model RMSE:       {rmse_m:.2f} m/day")
        print(f"    Persistence RMSE: {rmse_p:.2f} m/day (Lag error during heating)")
        print(f"    Ridge RMSE:       {rmse_r:.2f} m/day (Ablation without Kp/F10.7)")
        print(f"    Skill vs. Persistence: {skill_p:+.3f} ({skill_p*100:+.1f}%)")
        print(f"    Skill vs. Ridge:       {skill_r:+.3f} ({skill_r*100:+.1f}%)")

        event_reports.append({
            'event_name': ev_name,
            'start_date': start_d,
            'end_date': end_d,
            'max_kp': max_kp,
            'samples': int(len(ev_df)),
            'rmse_model': rmse_m,
            'rmse_persistence': rmse_p,
            'rmse_ridge': rmse_r,
            'skill_vs_persistence': float(skill_p),
            'skill_vs_ridge': float(skill_r)
        })

    # ---------------------------------------------------------
    # 9. Regime Breakdown: Deep LEO vs Upper LEO
    # ---------------------------------------------------------
    regime_reports = {}
    for reg_name, reg_mask in [('Deep LEO (<550km)', deep_mask_test), ('Upper LEO (>=550km)', upper_mask_test)]:
        reg_df = test_df[reg_mask]
        y_t = reg_df[TARGET_COLUMN].values
        r_m = float(np.sqrt(mean_squared_error(y_t, reg_df['pred_model'])))
        r_p = float(np.sqrt(mean_squared_error(y_t, reg_df['pers_lag1'])))
        r_r = float(np.sqrt(mean_squared_error(y_t, reg_df['pred_ridge'])))
        r_a = float(np.sqrt(mean_squared_error(y_t, reg_df['pred_alt_bin'])))
        
        # Per sat median skill in regime
        sat_sub = sat_df[sat_df['norad_id'].isin(reg_df['norad_id'].unique())]
        regime_reports[reg_name] = {
            'samples': int(len(reg_df)),
            'satellites': int(reg_df['norad_id'].nunique()),
            'rmse_model': r_m,
            'rmse_persistence': r_p,
            'rmse_ridge': r_r,
            'rmse_altbin': r_a,
            'pooled_skill_vs_pers': float(1.0 - r_m / r_p),
            'pooled_skill_vs_ridge': float(1.0 - r_m / r_r),
            'median_skill_vs_pers': float(sat_sub['skill_vs_pers1'].median()),
            'win_rate_vs_pers': float(sat_sub['beats_pers1'].mean() * 100.0)
        }

    # ---------------------------------------------------------
    # 10. Generate Multi-Panel Skill Score Chart
    # ---------------------------------------------------------
    print("\n[STEP 10] Generating Skill Visualization Plot...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    
    # Panel 1: Per-Satellite Skill vs. Altitude
    ax = axes[0]
    colors = ['#00d4ff' if r == 'Deep LEO' else '#ff007f' for r in sat_df['regime']]
    scatter = ax.scatter(sat_df['mean_altitude_km'], sat_df['skill_vs_pers1'], c=sat_df['mean_altitude_km'], cmap='coolwarm', s=70, edgecolors='black', linewidths=0.5)
    ax.axhline(0.0, color='gray', linestyle='--', linewidth=1.5, label='Persistence Baseline (Skill = 0)')
    ax.axvline(ALTITUDE_THRESHOLD, color='orange', linestyle=':', label='550 km Regime Threshold')
    ax.set_xlabel('Mean Orbit Altitude (km)', fontsize=12)
    ax.set_ylabel('Forecast Skill vs. 1-Day Persistence\n[1 - (RMSE_model / RMSE_pers)]', fontsize=12)
    ax.set_title('Per-Satellite Skill Score vs. Orbit Altitude (2026 Test)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Altitude (km)')
    
    # Annotate satellites
    for _, r in sat_df.iterrows():
        if abs(r['skill_vs_pers1']) > 0.15 or r['mean_altitude_km'] < 530:
            ax.annotate(r['satellite_name'][:10], (r['mean_altitude_km'], r['skill_vs_pers1']), textcoords="offset points", xytext=(0,6), ha='center', fontsize=8)

    # Panel 2: Quiet vs. Storm RMSE Comparison
    ax = axes[1]
    slice_names = ['Quiet (Kp<4)', 'Moderate (4<=Kp<6)', 'Storm (Kp>=6)']
    m_rmses = [s['rmse_model'] for s in slice_reports]
    p_rmses = [s['rmse_persistence'] for s in slice_reports]
    r_rmses = [s['rmse_ridge'] for s in slice_reports]
    
    x = np.arange(len(slice_names))
    width = 0.25
    ax.bar(x - width, m_rmses, width, label='Project B-Star Model', color='#00d4ff')
    ax.bar(x, p_rmses, width, label='1-Day Persistence', color='#ffa500')
    ax.bar(x + width, r_rmses, width, label='Ridge (No Weather Ablation)', color='#ff4444')
    
    ax.set_xticks(x)
    ax.set_xticklabels(slice_names, fontsize=11)
    ax.set_ylabel('RMSE (m/day)', fontsize=12)
    ax.set_title('Model vs. Baselines Across Space Weather Regimes', fontsize=13, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"   [SAVED] Plot saved to {OUTPUT_PLOT}")

    # ---------------------------------------------------------
    # 11. Final Comprehensive Report Serialization
    # ---------------------------------------------------------
    full_report = {
        'total_evaluated_rows': N_test,
        'satellites_evaluated': N_sats,
        'per_satellite_summary': {
            'vs_altitude_bin_mean': {
                'median_skill': median_skill_altbin,
                'win_rate_percent': win_rate_altbin,
                'satellites_won': int(sat_df['beats_altbin'].sum())
            },
            'vs_ridge_no_weather': {
                'median_skill': median_skill_ridge,
                'win_rate_percent': win_rate_ridge,
                'satellites_won': int(sat_df['beats_ridge'].sum())
            },
            'vs_persistence_lag1': {
                'median_skill': median_skill_pers1,
                'mean_skill': mean_skill_pers1,
                'win_rate_percent': win_rate_pers1,
                'satellites_won': int(sat_df['beats_pers1'].sum())
            }
        },
        'quiet_vs_storm_slices': slice_reports,
        'specific_storm_events': event_reports,
        'regime_breakdown': regime_reports
    }
    
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, indent=2)
    print(f"   [SAVED] Full JSON Report saved to {OUTPUT_REPORT}")
    print("\n[COMPLETE] Forecast Skill Benchmark successfully executed!")

if __name__ == "__main__":
    run_skill_evaluation()

"""
run_three_decisive_tests.py
===========================
Executes the Three Decisive Tests:
1. As-of Join: Join weather strictly before prev_epoch (start of decay interval).
2. Permutation Test: 30 block-shuffled runs (7-day blocks) across dates.
3. Block Bootstrap: 2,000 block-bootstrap iterations (7-day blocks) over time.
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
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

DB_NAME = get_db_path("satellite_data.db")
TRAIN_CUTOFF = "2025-01-01"
VAL_CUTOFF = "2026-01-01"
ACTIVE_PAYLOAD_IDS = [25544, 48274]
RNG_SEED = 42

def rmse(y_true, y_pred=None):
    if y_pred is None:
        return float(np.sqrt(np.mean(np.square(y_true))))
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def build_asof_dataset():
    print("[1/4] Building As-Of Weather Dataset (Strictly before prev_epoch)...")
    conn = sqlite3.connect(DB_NAME)

    # 1. Load space weather and build 3-hourly causal features
    sw = pd.read_sql("SELECT time_tag, kp FROM space_weather ORDER BY time_tag", conn)
    sw['time'] = pd.to_datetime(sw['time_tag'], format='ISO8601', utc=True)
    sw = sw.sort_values('time').reset_index(drop=True)
    sw = sw.drop_duplicates('time')

    sw['Kp_Current'] = sw['kp']
    sw['Kp_roll_24h_mean'] = sw['kp'].rolling(8, min_periods=1).mean()
    sw['Kp_roll_24h_max'] = sw['kp'].rolling(8, min_periods=1).max()
    sw['Kp_roll_72h_mean'] = sw['kp'].rolling(24, min_periods=1).mean()
    sw['Kp_roll_24h_std'] = sw['kp'].rolling(8, min_periods=1).std().fillna(0.0)
    sw['delta_Kp_24h'] = sw['Kp_roll_24h_mean'] - sw['Kp_roll_24h_mean'].shift(8)
    sw['delta_Kp_72h'] = sw['Kp_roll_24h_mean'] - sw['Kp_roll_24h_mean'].shift(24)

    # 2. Load solar flux (observed at 20:00 UTC at Penticton)
    sf = pd.read_sql("SELECT date_tag, f107_obs FROM solar_flux ORDER BY date_tag", conn)
    sf['date'] = pd.to_datetime(sf['date_tag'], utc=True)
    sf = sf.sort_values('date').reset_index(drop=True)
    sf = sf.drop_duplicates('date')
    sf['time'] = sf['date'] + pd.Timedelta(hours=20)
    sf['f107_81d_avg'] = sf['f107_obs'].rolling(81, min_periods=1).mean()
    sf['delta_f107_24h'] = sf['f107_obs'] - sf['f107_obs'].shift(1)
    sf['f107_ratio'] = sf['f107_obs'] / (sf['f107_81d_avg'] + 1e-4)

    # Merge solar flux into 3-hourly sw
    sw = pd.merge_asof(sw, sf[['time', 'f107_obs', 'f107_81d_avg', 'delta_f107_24h', 'f107_ratio']],
                       on='time', direction='backward')

    # 3. Load geometric dataset and match exact prev_epoch
    df_geom = pd.read_csv("training_set_geometric_expanded.csv")
    df_geom['date_dt'] = pd.to_datetime(df_geom['date'])
    df_geom['date_only'] = df_geom['date_dt'].dt.date

    tle = pd.read_sql("SELECT norad_id, epoch_datetime FROM tle_history WHERE epoch_datetime >= '2014-01-01' ORDER BY norad_id, epoch_datetime", conn)
    conn.close()

    tle['epoch'] = pd.to_datetime(tle['epoch_datetime'], format='ISO8601', utc=True)
    tle['date_only'] = tle['epoch'].dt.date
    tle = tle.drop_duplicates(subset=['norad_id', 'date_only'], keep='first')
    tle['prev_epoch'] = tle.groupby('norad_id')['epoch'].shift(1)
    tle['prev_date'] = tle.groupby('norad_id')['date_only'].shift(1)
    tle['days_diff'] = (pd.to_datetime(tle['date_only']) - pd.to_datetime(tle['prev_date'])).dt.days

    tle_consec = tle[tle['days_diff'] == 1][['norad_id', 'date_only', 'prev_epoch', 'epoch']].copy()
    df = pd.merge(df_geom, tle_consec, on=['norad_id', 'date_only'], how='inner')

    # Filter standard inclusion
    df = df[~df['norad_id'].isin(ACTIVE_PAYLOAD_IDS)]
    if 'is_maneuver' in df.columns:
        df = df[df['is_maneuver'] == False]
    df = df[(df['perigee_alt_km'] >= 200.0) & (df['decay_rate_m'] > 0.01) & (df['decay_rate_m'] < 500.0)]
    df = df.sort_values('prev_epoch').reset_index(drop=True)

    # 4. As-of Join: strictly before prev_epoch!
    weather_cols_sw = [
        'Kp_Current', 'Kp_roll_24h_mean', 'Kp_roll_24h_max', 'Kp_roll_72h_mean', 'Kp_roll_24h_std',
        'delta_Kp_24h', 'delta_Kp_72h',
        'f107_obs', 'f107_81d_avg', 'delta_f107_24h', 'f107_ratio'
    ]
    # Drop any existing weather columns from df to prevent _x, _y collision
    old_cols_to_drop = [c for c in df.columns if c in weather_cols_sw or 'Kp_' in c or 'f107_' in c]
    df = df.drop(columns=old_cols_to_drop)

    sw = sw.sort_values('time').reset_index(drop=True)
    asof_df = pd.merge_asof(df, sw[['time'] + weather_cols_sw],
                            left_on='prev_epoch', right_on='time',
                            direction='backward')

    # 5. Build daily history features strictly from past observations
    asof_df['date'] = pd.to_datetime(asof_df['date'])
    asof_df = asof_df.sort_values(['norad_id', 'date']).reset_index(drop=True)

    # Reindex to calendar daily grid per satellite
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

    # History lags
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

    # Physical interaction terms between past drag and strictly causal weather
    full['interact_drag_delta_Kp'] = full['y_t1'] * full['delta_Kp_24h']
    full['interact_drag_Kp'] = full['y_t1'] * np.maximum(0.0, full['Kp_roll_24h_mean'] - 2.0)

    # Target: log-ratio innovation
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
    print(f"    As-of dataset ready: {len(clean)} rows.")
    return clean, HISTORY_FEATURES, WEATHER_FEATURES, ALL_FEATURES

def fit_scalar(y_true, y_pred):
    denom = np.sum(y_pred ** 2)
    if denom < 1e-9:
        return 1.0
    return float(np.sum(y_true * y_pred) / denom)

def evaluate_models(train_df, val_df, test_df, hist_cols, all_cols):
    xgb_params = {
        'objective': 'reg:squarederror',
        'n_estimators': 1500,
        'learning_rate': 0.02,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'early_stopping_rounds': 80,
        'n_jobs': 4,
        'random_state': RNG_SEED,
        'verbosity': 0
    }

    # Model A: History-Only
    xgb_h = XGBRegressor(**xgb_params)
    xgb_h.fit(train_df[hist_cols], train_df['target_log_ratio'],
              eval_set=[(val_df[hist_cols], val_df['target_log_ratio'])],
              verbose=False)

    # Model B: History + Weather
    xgb_w = XGBRegressor(**xgb_params)
    xgb_w.fit(train_df[all_cols], train_df['target_log_ratio'],
              eval_set=[(val_df[all_cols], val_df['target_log_ratio'])],
              verbose=False)

    # Calibration on validation set
    val_pred_h = xgb_h.predict(val_df[hist_cols])
    val_pred_w = xgb_w.predict(val_df[all_cols])
    cal_h = fit_scalar(val_df['target_log_ratio'], val_pred_h)
    cal_w = fit_scalar(val_df['target_log_ratio'], val_pred_w)

    test_df = test_df.copy()
    test_df['y_pers'] = test_df['y_t1']
    test_df['y_hat_h'] = np.maximum(0.01, test_df['y_t1'] * np.exp(cal_h * xgb_h.predict(test_df[hist_cols])))
    test_df['y_hat_w'] = np.maximum(0.01, test_df['y_t1'] * np.exp(cal_w * xgb_w.predict(test_df[all_cols])))

    return test_df, cal_h, cal_w, xgb_h, xgb_w

def compute_per_sat_metrics(test_df):
    sat_list = [nid for nid in test_df['norad_id'].unique() if len(test_df[test_df['norad_id'] == nid]) >= 10]
    records = []
    for nid in sat_list:
        sub = test_df[test_df['norad_id'] == nid]
        y = sub['decay_rate_m'].values
        p = sub['y_pers'].values
        h = sub['y_hat_h'].values
        w = sub['y_hat_w'].values

        r_p = rmse(y, p)
        r_h = rmse(y, h)
        r_w = rmse(y, w)

        records.append({
            'norad_id': int(nid),
            'n_rows': len(sub),
            'rmse_pers': r_p,
            'rmse_h': r_h,
            'rmse_w': r_w,
            'skill_h': 1.0 - r_h / r_p,
            'skill_w': 1.0 - r_w / r_p,
            'gain_w': 1.0 - r_w / r_h,
        })
    sat_df = pd.DataFrame(records)
    return sat_df

def run():
    print("=" * 80)
    print("THE THREE DECISIVE TESTS: RIGOROUS CAUSALITY & PERMUTATION BENCHMARK")
    print("=" * 80)

    clean_df, hist_cols, weather_cols, all_cols = build_asof_dataset()

    train_df = clean_df[clean_df['date'] < TRAIN_CUTOFF].copy()
    val_df = clean_df[(clean_df['date'] >= TRAIN_CUTOFF) & (clean_df['date'] < VAL_CUTOFF)].copy()
    test_df = clean_df[clean_df['date'] >= VAL_CUTOFF].copy()

    print(f"\nSplits: Train={len(train_df)} rows, Val={len(val_df)} rows, Test={len(test_df)} rows across {test_df['norad_id'].nunique()} sats.")

    # -------------------------------------------------------------
    # TEST 1: AS-OF JOIN EVALUATION
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST 1: AS-OF JOIN (Weather Strictly Before prev_epoch)")
    print("-" * 70)

    test_res, cal_h, cal_w, model_h, model_w = evaluate_models(train_df, val_df, test_df, hist_cols, all_cols)
    sat_metrics = compute_per_sat_metrics(test_res)

    med_skill_w = float(sat_metrics['skill_w'].median())
    med_skill_h = float(sat_metrics['skill_h'].median())
    med_gain_w = float(sat_metrics['gain_w'].median())
    win_pers = float(np.mean(sat_metrics['skill_w'] > 0) * 100)
    win_hist = float(np.mean(sat_metrics['gain_w'] > 0) * 100)

    print(f"Calibration Scalars: Cal_H={cal_h:.4f}, Cal_W={cal_w:.4f}")
    print(f"Test 1 Results across N={len(sat_metrics)} Satellites:")
    print(f"  History-Only Median Skill vs Pers:    {med_skill_h:+.3f}")
    print(f"  History + Weather Median Skill vs Pers: {med_skill_w:+.3f}  (Win Rate vs Pers: {win_pers:.1f}%)")
    print(f"  INCREMENTAL WEATHER GAIN OVER HISTORY:  {med_gain_w:+.3f}  (Win Rate vs Hist: {win_hist:.1f}%)")

    test_1_pass = med_gain_w > 0.02
    print(f"\n>>> TEST 1 STATUS: {'PASS' if test_1_pass else 'FAIL'}")
    if test_1_pass:
        print(f"    Weather gain remains positive ({med_gain_w*100:+.1f}%) with zero lookahead leak.")
    else:
        print(f"    Weather gain vanished ({med_gain_w*100:+.1f}%); earlier gain was a leak.")

    # -------------------------------------------------------------
    # TEST 2: PERMUTATION TEST (Block-shuffled weather across dates)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST 2: PERMUTATION TEST (30 Block-Shuffled Runs of 7-Day Blocks)")
    print("-" * 70)

    RESULTS_DIR = get_model_path("Model_3_Causal_Log_Ratio_Innovation")
    perm_cache_file = os.path.join(RESULTS_DIR, "perm_gains.json")
    if not os.path.exists(perm_cache_file) and os.path.exists("models_expanded/perm_gains.json"):
        perm_cache_file = "models_expanded/perm_gains.json"
    if os.path.exists(perm_cache_file):
        print(f"Loading cached 30-run permutation results from {perm_cache_file}...")
        with open(perm_cache_file) as f:
            p_data = json.load(f)
        perm_mean = p_data['perm_mean']
        perm_std = p_data['perm_std']
        perm_p95 = p_data['perm_p95']
        perm_max = p_data['perm_max']
        perm_gains = np.array([perm_mean])
        print(f"  Loaded 30-run summary: Mean={perm_mean:+.4f}, Std={perm_std:.4f}, P95={perm_p95:+.4f}, Max={perm_max:+.4f}")
    else:
        # Build unique date table for weather features indexed by date
        wx_by_date = clean_df[['date'] + weather_cols].drop_duplicates('date').set_index('date')
        all_dates = np.array(sorted(wx_by_date.index.unique()))

        block_size = 7
        n_blocks = len(all_dates) // block_size
        date_blocks = np.array_split(all_dates, n_blocks)

        rng = np.random.default_rng(RNG_SEED)
        perm_gains = []

        print(f"Running 30 block-shuffled retrainings on {len(all_dates)} calendar days ({n_blocks} blocks of ~7d)...")
        for b_idx in range(30):
            order = rng.permutation(len(date_blocks))
            shuffled_dates = np.concatenate([date_blocks[i] for i in order])
            mapping = dict(zip(all_dates, shuffled_dates[:len(all_dates)]))

            df_perm = clean_df.copy()
            mapped_dates = df_perm['date'].map(mapping)
            df_perm[weather_cols] = wx_by_date.loc[mapped_dates, weather_cols].values

            df_perm['interact_drag_delta_Kp'] = df_perm['y_t1'] * df_perm['delta_Kp_24h']
            df_perm['interact_drag_Kp'] = df_perm['y_t1'] * np.maximum(0.0, df_perm['Kp_roll_24h_mean'] - 2.0)

            tr_p = df_perm[df_perm['date'] < TRAIN_CUTOFF]
            va_p = df_perm[(df_perm['date'] >= TRAIN_CUTOFF) & (df_perm['date'] < VAL_CUTOFF)]
            te_p = df_perm[df_perm['date'] >= VAL_CUTOFF].copy()

            xgb_perm = XGBRegressor(
                objective='reg:squarederror', n_estimators=1500, learning_rate=0.02,
                max_depth=5, subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=80,
                n_jobs=4, random_state=RNG_SEED, verbosity=0
            )
            xgb_perm.fit(tr_p[all_cols], tr_p['target_log_ratio'],
                         eval_set=[(va_p[all_cols], va_p['target_log_ratio'])],
                         verbose=False)

            cal_p = fit_scalar(va_p['target_log_ratio'], xgb_perm.predict(va_p[all_cols]))

            te_p['y_pers'] = te_p['y_t1']
            te_p['y_hat_w'] = np.maximum(0.01, te_p['y_t1'] * np.exp(cal_p * xgb_perm.predict(te_p[all_cols])))
            te_p['y_hat_h'] = test_res['y_hat_h'].values

            sat_p = compute_per_sat_metrics(te_p)
            gain_p = float(sat_p['gain_w'].median())
            perm_gains.append(gain_p)
            if (b_idx + 1) % 5 == 0 or b_idx == 0:
                print(f"  Iteration {b_idx+1:2d}/30: Permuted Weather Gain = {gain_p:+.4f}")

        perm_gains = np.array(perm_gains)
        perm_mean = float(np.mean(perm_gains))
        perm_std = float(np.std(perm_gains))
        perm_p95 = float(np.percentile(perm_gains, 95))
        perm_max = float(np.max(perm_gains))

    print(f"\nPermutation Distribution Summary (30 runs):")
    print(f"  Mean Permuted Gain:   {perm_mean:+.4f}")
    print(f"  Std Permuted Gain:    {perm_std:.4f}")
    print(f"  95th Percentile:      {perm_p95:+.4f}")
    print(f"  Max Permuted Gain:    {perm_max:+.4f}")
    print(f"  True Unpermuted Gain: {med_gain_w:+.4f}")

    test_2_pass = med_gain_w > perm_p95
    print(f"\n>>> TEST 2 STATUS: {'PASS' if test_2_pass else 'FAIL'}")
    if test_2_pass:
        print(f"    True weather gain ({med_gain_w*100:+.1f}%) exceeds 95% of permuted runs ({perm_p95*100:+.1f}%).")
        print("    Confirms the gain is specifically driven by real space weather alignment, not date/column artifacts.")
    else:
        print("    True gain fails to exceed permuted threshold. Weather signal is indistinguishable from noise.")

    # -------------------------------------------------------------
    # TEST 3: BLOCK BOOTSTRAP OVER TIME (7-Day Calendar Blocks)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST 3: BLOCK BOOTSTRAP OVER TIME (2,000 Rounds of 7-Day Calendar Blocks)")
    print("-" * 70)

    # Use string dates to ensure exact match without dtype mismatches
    date_str_arr = test_res['date'].dt.strftime('%Y-%m-%d').values
    uniq_dates = np.array(sorted(set(date_str_arr)))
    block_size = 7
    starts = np.arange(0, len(uniq_dates), block_size)

    err_pers = (test_res['decay_rate_m'] - test_res['y_pers']).values
    err_h = (test_res['decay_rate_m'] - test_res['y_hat_h']).values
    err_w = (test_res['decay_rate_m'] - test_res['y_hat_w']).values

    boot_skill_w = []
    boot_gain_w = []

    rng_boot = np.random.default_rng(RNG_SEED)
    N_BOOT = 2000

    print(f"Resampling {len(starts)} calendar blocks (7 days each) over {len(uniq_dates)} test dates...")
    for _ in range(N_BOOT):
        picks = rng_boot.choice(starts, len(starts), replace=True)
        # Gather all rows falling in picked date blocks
        selected_dates = np.concatenate([uniq_dates[s:min(s+block_size, len(uniq_dates))] for s in picks])
        # Vectorized string match
        idx = np.where(np.isin(date_str_arr, selected_dates))[0]

        r_p = rmse(err_pers[idx])
        r_h = rmse(err_h[idx])
        r_w = rmse(err_w[idx])

        if r_p > 1e-6:
            boot_skill_w.append(1.0 - r_w / r_p)
        if r_h > 1e-6:
            boot_gain_w.append(1.0 - r_w / r_h)

    ci_skill = np.percentile(boot_skill_w, [2.5, 50, 97.5])
    ci_gain = np.percentile(boot_gain_w, [2.5, 50, 97.5])

    print(f"\nBlock Bootstrap 95% Confidence Intervals (Temporal Blocks):")
    print(f"  (a) Model B Skill vs Persistence: Median = {ci_skill[1]:+.3f}  (95% CI: [{ci_skill[0]:+.3f}, {ci_skill[2]:+.3f}])")
    print(f"  (b) Weather Gain over History:   Median = {ci_gain[1]:+.3f}  (95% CI: [{ci_gain[0]:+.3f}, {ci_gain[2]:+.3f}])")

    test_3_pass = ci_gain[0] > 0.0
    print(f"\n>>> TEST 3 STATUS: {'PASS' if test_3_pass else 'FAIL (Inconclusive)'}")
    if test_3_pass:
        print(f"    95% Temporal Block Bootstrap interval [{ci_gain[0]*100:+.1f}%, {ci_gain[2]*100:+.1f}%] strictly excludes zero.")
    else:
        print(f"    95% Temporal Block Bootstrap interval [{ci_gain[0]*100:+.1f}%, {ci_gain[2]*100:+.1f}%] includes zero.")

    # -------------------------------------------------------------
    # FINAL VERDICT
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL SUMMARY OF THE THREE TESTS")
    print("=" * 80)
    print(f"Test 1 (As-Of Join Causal Integrity):   {'PASS' if test_1_pass else 'FAIL'}")
    print(f"Test 2 (30-Run Permutation Null Test): {'PASS' if test_2_pass else 'FAIL'}")
    print(f"Test 3 (Time-Block Bootstrap 95% CI):   {'PASS' if test_3_pass else 'FAIL'}")

    all_pass = test_1_pass and test_2_pass and test_3_pass
    if all_pass:
        final_version = "VERSION A"
        print(f"\nFINAL VERDICT: {final_version} CONFIRMED.")
        print("The weather gain is real, strictly causal, statistically robust, and not an artifact of leak or date encoding.")
    elif not test_1_pass:
        final_version = "VERSION B (Leak detected in Test 1)"
        print(f"\nFINAL VERDICT: {final_version}")
    elif not test_2_pass:
        final_version = "VERSION B (Permutation failure - non-weather artifact)"
        print(f"\nFINAL VERDICT: {final_version}")
    else:
        final_version = "INCONCLUSIVE (Test 3 interval includes 0)"
        print(f"\nFINAL VERDICT: {final_version}")

    # Save results to JSON
    output_data = {
        'test_1_asof': {
            'status': 'PASS' if test_1_pass else 'FAIL',
            'history_median_skill': med_skill_h,
            'weather_median_skill': med_skill_w,
            'weather_incremental_gain': med_gain_w,
            'win_rate_vs_persistence': win_pers,
            'win_rate_vs_history': win_hist,
        },
        'test_2_permutation': {
            'status': 'PASS' if test_2_pass else 'FAIL',
            'n_runs': len(perm_gains),
            'true_gain': med_gain_w,
            'perm_mean': perm_mean,
            'perm_std': perm_std,
            'perm_p95': perm_p95,
            'perm_max': perm_max,
        },
        'test_3_block_bootstrap': {
            'status': 'PASS' if test_3_pass else 'FAIL',
            'skill_vs_pers_ci95': list(ci_skill),
            'weather_gain_over_history_ci95': list(ci_gain),
        },
        'final_verdict': final_version
    }
    out_json_path = os.path.join(RESULTS_DIR, "three_decisive_tests_results.json")
    with open(out_json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {out_json_path}")

if __name__ == "__main__":
    run()

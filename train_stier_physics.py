"""
6_train_81d_avg.py - S-Tier Model + 81-Day Solar Cycle Average
==============================================================
Space Weather Drag Model: Implementing Standard Atmospheric Physics

This script enhances the S-Tier model by adding the "81-Day Rule":
Thermospheric temperature (and density) depends on both:
1. Daily F10.7 (Immediate EUV heating)
2. 81-Day Centered Average F10.7 (Background temperature / Solar Cycle)

Refactors '4_train_optimized.py' & '5_train_physics.py' into one ultimate model.

Author: Principal Data Scientist
Phase: S-Tier + Solar Cycle Upgrade
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = 'final_training_set_v2.csv'
OUTPUT_PLOT = '81d_model_performance.png'
TRAIN_CUTOFF_DATE = '2024-01-01'

# XGBoost Optimized Params (Same as before)
XGB_PARAMS = {
    'n_estimators': 5000,
    'learning_rate': 0.02,
    'max_depth': 7,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'random_state': 42,
    'n_jobs': 1,
    'early_stopping_rounds': 100
}

print("=" * 78)
print("☀️ S-TIER PHYSICS MODEL - 81-DAY SOLAR CYCLE UPGRADE")
print("   Integrating Background Temperature Physics (Daily + 81d Avg F10.7)")
print("=" * 78)

# =============================================================================
# 1. LOAD & PREPARE DATA
# =============================================================================
print("\n📂 [1/5] Loading Dataset...", end=" ")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} rows")

# Convert date
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Handle missing F10.7
if df['f107_obs'].isna().sum() > 0:
    print(f"   ⚠️ Dropping {df['f107_obs'].isna().sum()} rows missing F10.7")
    df = df.dropna(subset=['f107_obs'])

# =============================================================================
# 2. FEATURE ENGINEERING: THE 81-DAY RULE
# =============================================================================
print("\n⚙️  [2/5] Engineering 81-Day Solar Cycle Feature...")

# CRITICAL: Calculate rolling average on UNIQUE dates only
# If we calculate on the full DF, we mix multiple satellites per day = WRONG.
unique_weather = df[['date', 'f107_obs']].drop_duplicates().sort_values('date').set_index('date')

# Calculate 81-Day Centered Average
# min_periods=1 allows computing even at edges (using available data)
unique_weather['f107_81d_avg'] = unique_weather['f107_obs'].rolling(
    window=81, center=True, min_periods=1
).mean()

# Forward fill edges if any NaNs remain (standard practice)
unique_weather['f107_81d_avg'] = unique_weather['f107_81d_avg'].ffill().bfill()

print(f"   ├── Calculated 81-day centered average for {len(unique_weather)} unique days")
print(f"   ├── Sample:\n{unique_weather.head(3)}")

# Merge back
df = pd.merge(df, unique_weather[['f107_81d_avg']], on='date', how='left')

# Re-engineer Physics Features (from Script 4)
# Drag Force Proxy = F10.7 / Altitude^2 (Simplified drag eq)
# Note: Using F10.7_81d_avg roughly proxies 'exospheric temperature' better
df['drag_force_proxy'] = df['f107_81d_avg'] / ((df['semi_major_axis_km'] / 1000) ** 2)

# Storm Exposure (Kp dependent)
df['storm_exposure'] = df['Kp_max'] * np.exp(-(df['altitude_km'] - 200) / 100)

# Inertia
df['Kp_Inertia_3d'] = df['Kp_Roll_72h_Mean'] - df['Kp_Roll_24h_Mean']

print("   └── Features merged & engineered ✓")

# =============================================================================
# 3. PHYSICS GATE FILTERS
# =============================================================================
print("\n🔬 [3/5] Applying Physics Gate Filters...")
df = df[
    (df['is_circular'] == True) & 
    (df['is_maneuver'] == False) & 
    (df['decay_rate_m'] > 0.01)
]
print(f"   └── Clean Physics Regime: {len(df):,} rows")

# =============================================================================
# 4. FEATURE SELECTION
# =============================================================================
print("\n️🎛️  [4/5] Preparing Feature Set...")

FEATURE_COLUMNS = [
    # Orbital State
    'semi_major_axis_km', 'eccentricity', 'perigee_alt_km', 'mean_motion',
    
    # Seasonality
    'sin_doy', 'cos_doy',
    
    # Space Weather
    'Kp_mean', 'Kp_max', 'Kp_Lag_24h', 'Kp_Inertia_3d',
    
    # S-TIER PHYSICS (Solar Cycle)
    'f107_obs',         # Daily EUV (Short-term var)
    'f107_81d_avg',     # 81-Day Avg (Background Temp) -- THE NEW STAR
    'static_bc_est',    # Ballistic Coeff
    
    # Physics Interaction Terms
    'drag_force_proxy',
    'storm_exposure'
]

TARGET_COLUMN = 'decay_rate_m'

print(f"   ├── Features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
print(f"   ├── REMOVED: 'norad_id'")

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

# Split
train_mask = df['date'] < TRAIN_CUTOFF_DATE
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

print(f"   ├── Train: {len(X_train):,} | Test: {len(X_test):,}")

# =============================================================================
# 5. TRAINING
# =============================================================================
print("\n🚀 [5/5] Training 81-Day Physics Model...")

model = XGBRegressor(**XGB_PARAMS)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=100
)

# =============================================================================
# 6. EVALUATION
# =============================================================================
print("\n📈 Evaluation:")

y_pred_test = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2 = r2_score(y_test, y_pred_test)

print(f"   ╔══════════════════════════════════════════════════════════╗")
print(f"   ║   81-DAY SOLAR CYCLE MODEL RESULTS                       ║")
print(f"   ╠══════════════════════════════════════════════════════════╣")
print(f"   ║   RMSE: {rmse:.4f} m/day                                 ║")
print(f"   ║   R²:   {r2:.4f}                                         ║")
print(f"   ╚══════════════════════════════════════════════════════════╝")

# Feature Importance Plot
feature_importance = pd.DataFrame({
    'feature': FEATURE_COLUMNS,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   📊 Feature Importance:")
for idx, row in feature_importance.iterrows():
    bar = '█' * int(row['importance'] * 50)
    print(f"   │ {row['feature']:20s} {bar} {row['importance']:.4f}")

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5, c=y_test, cmap='viridis')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title(f'81-Day Solar Cycle Model: Predictions vs Actual (R2={r2:.3f})')
plt.xlabel('Actual Decay (m/day)')
plt.ylabel('Predicted Decay (m/day)')
plt.grid(True, alpha=0.3)
plt.savefig(OUTPUT_PLOT)
print(f"\n   └── Plot saved to {OUTPUT_PLOT}")

# Save metrics for comparison
with open("model_comparison.txt", "a") as f:
    f.write(f"81-Day Solar Cycle Model: RMSE={rmse:.4f}, R2={r2:.4f}\n")

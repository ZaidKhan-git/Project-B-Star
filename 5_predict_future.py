"""
5_predict_future.py - The "Invisible Wall" Detector
====================================================
Operational Space Weather Early Warning System

Purpose:
Connects the S-Tier Physics Engine to LIVE NOAA forecasts to predict
atmospheric drag spikes before they happen.

Features:
1. LIVE FORECASTS: Fetches Kp and F10.7 from NOAA SWPC JSON APIs.
2. ORBIT PROPAGATION: Calculates future Solar Beta Angles for target satellites.
3. PRODUCTION MODEL: Retrains the Optimized Geometric Ensemble on ALL available data.
4. EARLY WARNING: Outputs "Drag Risk Index" alerts (Green/Yellow/Red).

Author: Principal Space Operations Engineer
"""

import pandas as pd
import numpy as np
import requests
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from xgboost import XGBRegressor
from skyfield.api import load, EarthSatellite, wgs84
import warnings

# Import Physics Engine components
# (Assumes 3_physics_engine.py is in the same directory)
from importlib.machinery import SourceFileLoader
physics = SourceFileLoader("physics_engine", "3_physics_engine.py").load_module()

warnings.filterwarnings('ignore')

import argparse

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = "training_set_geometric.csv"
DB_PATH = "satellite_data.db"

# NOAA API Endpoint
URL_KP_FORECAST = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json"
URL_F107_FORECAST = "https://services.swpc.noaa.gov/products/solar-10cm-flux.json"

# Production Model Config (Best from 4_train_geometric.py)
MODEL_PARAMS = {
    'objective': 'reg:squarederror',
    'n_estimators': 8000,
    'learning_rate': 0.01,
    'max_depth': 8,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'n_jobs': 1,
    'verbosity': 0
}


# =============================================================================
# 1. TRAIN PRODUCTION MODEL
# =============================================================================
def train_production_model():
    """
    Trains the 'Golden Model' on 100% of available historical data.
    No train/test split - we want maximum knowledge for the future.
    """
    print("🧠 [1/4] Training Production Model (100% Data)...")
    
    df = pd.read_csv(DATA_PATH)
    
    # Feature Columns (Must match training script exactly)
    features = [
        'sun_exposure_factor', 'beta_angle_deg',
        'f107_obs', 'f107_81d_avg',
        'Kp_mean', 'Kp_max', 'Kp_Lag_24h',
        'semi_major_axis_km', 'altitude_km', 'perigee_alt_km', 'eccentricity',
        'sin_doy', 'cos_doy',
        'static_bc_est'
    ]
    target = 'decay_rate_m'
    
    # Filter valid rows
    df = df.dropna(subset=features + [target])
    df = df[df['decay_rate_m'] > 0.01]  # Physics filter
    
    X = df[features]
    y = df[target]
    
    # Train robust model
    model = XGBRegressor(**MODEL_PARAMS)
    model.fit(X, y)
    
    print(f"   └── Trained on {len(X):,} samples")
    
    return model, df, features


# =============================================================================
# 2. FETCH NOAA FORECASTS
# =============================================================================
def fetch_forecasts() -> pd.DataFrame:
    """
    Fetches 3-day forecast for Kp and F10.7 from NOAA.
    """
    print("\n📡 [2/4] Fetching LIVE NOAA Forecasts...")
    
    # --- Fetch Kp ---
    try:
        resp = requests.get(URL_KP_FORECAST, timeout=10)
        data = resp.json()
        # Format: [time_tag, kp, observed, noaa_scale]
        # We want the forecasted values (observed=estimated means past typically, 
        # but check latest rows for forecast)
        # Actually NOAA JSON usually has: [time, kp, ...]
        # Let's clean it up.
        
        kp_rows = []
        for row in data[1:]: # Skip header
             # Time format: 2026-01-03 21:00:00
             time_tag = row[0]
             kp_val = float(row[1])
             kp_rows.append({'time': pd.to_datetime(time_tag), 'Kp': kp_val})
             
        kp_df = pd.DataFrame(kp_rows).sort_values('time')
        
        # Filter for future only (Starting from now)
        now = datetime.utcnow()
        kp_future = kp_df[kp_df['time'] > now].head(24) # Next ~72 hours (3-hr intervals)
        
        # Resample to Daily Mean/Max for the next 3 days
        kp_future['date'] = kp_future['time'].dt.date
        kp_daily = kp_future.groupby('date').agg({
            'Kp': ['mean', 'max']
        }).reset_index()
        kp_daily.columns = ['date', 'Kp_mean', 'Kp_max']
        
        print(f"   └── Retrieved Kp Forecast for: {kp_daily['date'].tolist()}")
        
    except Exception as e:
        print(f"   ❌ Failed to fetch Kp: {e}")
        # Fallback dummy values for demo
        kp_daily = pd.DataFrame({
            'date': [datetime.utcnow().date() + timedelta(days=i) for i in range(1, 4)],
            'Kp_mean': [3.0, 5.0, 7.0], # Simulation of a storm
            'Kp_max': [4.0, 6.0, 8.0]
        })
        print("   ⚠️ USING SIMULATED STORM PROFILE (Fallback)")

    # --- Fetch F10.7 ---
    # Simplified: Using last observed + persistence or NOAA 3-day if avail
    # For now, let's assume F10.7 is steady or slightly rising
    try:
        resp = requests.get(URL_F107_FORECAST, timeout=10)
        data = resp.json()
        # [time_tag, f107, ...]
        f107_rows = []
        for row in data[1:]:
            f107_rows.append({'time': pd.to_datetime(row[0]), 'f107': float(row[1])})
        
        f107_df = pd.DataFrame(f107_rows).sort_values('time')
        last_f107 = f107_df['f107'].iloc[-1]
        
    except:
        last_f107 = 150.0 # Default high solar max value
        
    # Add F10.7 to forecast
    kp_daily['f107_obs'] = last_f107
    kp_daily['f107_81d_avg'] = last_f107 # Approximation for short term
    
    return kp_daily


# =============================================================================
# 3. PROPAGATE ORBIT (PHYSICS ENGINE)
# =============================================================================
def get_future_geometry(target_id: int, forecast_days: pd.DataFrame) -> pd.DataFrame:
    """
    Propagates the satellite orbit to calculate future Beta Angles.
    """
    print("\n🛰️ [3/4] Propagating Orbit & Calculating Future Geometry...")
    
    # Load latest TLE
    ts = load.timescale()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT tle_line1, tle_line2, epoch_datetime FROM tle_history WHERE norad_id = ? ORDER BY epoch_datetime DESC LIMIT 1",
        (target_id,)
    )
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise ValueError(f"No TLE found for ID {target_id}")
        
    l1, l2, epoch_str = row
    sat = EarthSatellite(l1, l2, str(target_id), ts)
    
    print(f"   └── Propagating from Epoch: {epoch_str}")
    
    future_rows = []
    
    for _, day_row in forecast_days.iterrows():
        target_date = pd.to_datetime(day_row['date'])
        
        # Propagate to noon of target day
        t = ts.utc(target_date.year, target_date.month, target_date.day, 12, 0, 0)
        
        # Get State
        geocentric = sat.at(t)
        alt_km = wgs84.height_of(geocentric).km
        
        # Mean Elements (Approximate from SGP4)
        n0 = sat.model.no_kozai / 60.0 # rad/s
        e0 = sat.model.ecco
        i0_deg = np.degrees(sat.model.inclo)
        raan0_deg = np.degrees(sat.model.nodeo)
        
        # Simple RAAN Precession (J2 effect)
        # dΩ/dt ≈ -9.964 * (R/a)^3.5 * cos(i)  [deg/day]
        EARTH_R = 6378.137
        mu = 398600.4418
        a0 = (mu / n0**2)**(1/3) # Semi-major axis
        
        raan_drift = -9.964 * ((EARTH_R/a0)**3.5) * np.cos(np.radians(i0_deg))
        days_delta = (target_date.date() - pd.to_datetime(epoch_str).date()).days
        
        current_raan = (raan0_deg + raan_drift * days_delta) % 360
        
        # Build Row
        row_dict = day_row.to_dict()
        row_dict['inclination_deg'] = i0_deg
        row_dict['raan_deg'] = current_raan
        row_dict['semi_major_axis_km'] = a0
        row_dict['altitude_km'] = alt_km
        row_dict['perigee_alt_km'] = a0 * (1 - e0) - EARTH_R
        row_dict['eccentricity'] = e0
        row_dict['sin_doy'] = np.sin(2 * np.pi * target_date.dayofyear / 365.25)
        row_dict['cos_doy'] = np.cos(2 * np.pi * target_date.dayofyear / 365.25)
        
        future_rows.append(row_dict)
        
    future_df = pd.DataFrame(future_rows)
    
    # Calculate Beta Angle (Using Shared Physics Engine)
    future_df = physics.calculate_solar_beta(future_df)
    
    return future_df


# =============================================================================
# 4. PREDICT AND ALERT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Invisible Wall Detector - Satellite Drag Early Warning')
    parser.add_argument('--sat_id', type=int, default=25544, help='NORAD ID of the target satellite (default: 25544 ISS)')
    args = parser.parse_args()
    
    target_id = args.sat_id
    
    print("=" * 60)
    print("🔮 THE INVISIBLE WALL DETECTOR (OPERATIONAL)")
    print("   Early Warning System for Satellite Drag")
    print("=" * 60)
    
    # 1. Train
    # We return the dataframe (df) too so we can look up BCs for any sat
    model, df_train, feature_cols = train_production_model()
    
    # 2. Get Baseline and BC for requested satellite
    print(f"\n🔍 [1.5/4] Loading Profile for Satellite {target_id}...")
    
    sat_data = df_train[df_train['norad_id'] == target_id]
    
    if sat_data.empty:
        print(f"   ⚠️ WARNING: Satellite {target_id} not in training history.")
        print("   → Using default Ballistic Coefficient (Generic LEO)")
        print("   → Baseline will be estimated from generic constraints.")
        sat_bc = 1.0e7 # Generic Guess
        baseline = 10.0 # Generic Guess
        sat_name = f"Unknown-{target_id}"
    else:
        sat_bc = sat_data['static_bc_est'].iloc[-1]
        baseline = sat_data['decay_rate_m'].rolling(30, min_periods=5).median().iloc[-1]
        sat_name = sat_data['satellite_name'].iloc[0] if 'satellite_name' in sat_data.columns else f"ID {target_id}"
        print(f"   └── Identified: {sat_name}")
        print(f"   └── Ballistic Coefficient: {sat_bc:.4e}")
        print(f"   └── Baselined Drag: {baseline:.2f} m/day")

    # 3. Forecast
    forecast_df = fetch_forecasts()
    
    # 4. Propagate (Geometry)
    try:
        future_orbit = get_future_geometry(target_id, forecast_df)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("   (Ensure TLEs for this satellite are in the database)")
        return
    
    # 5. Add Static BC 
    future_orbit['static_bc_est'] = sat_bc
    
    # 6. Add Missing Lag Features (Assume persistence for demo)
    future_orbit['Kp_Lag_24h'] = future_orbit['Kp_mean'] # Worst case assumption
    
    # 7. Predict
    print("\n⚡ [4/4] Generating Drag Predictions...")
    X_future = future_orbit[feature_cols]
    preds = model.predict(X_future)
    future_orbit['predicted_decay'] = preds
    
    # 8. REPORT & ALERT
    print("\n" + "=" * 60)
    print(f"🚨 DRAG RISK INDEX: NEXT 72 HOURS | {sat_name} ({target_id})")
    print(f"   Baseline Risk: {baseline:.2f} m/day")
    print("=" * 60)
    
    for i, row in future_orbit.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        decay = row['predicted_decay']
        beta = row['beta_angle_deg']
        kp = row['Kp_max']
        
        if baseline > 0:
            risk_ratio = decay / baseline
            percent = risk_ratio * 100
        else:
            percent = 0
        
        # Risk Logic
        if percent > 250:
            status = "🔴 CRITICAL - EXECUTE SAFE MODE"
            color = "\033[91m" # Red
        elif percent > 150:
            status = "🟡 WARNING - DENSITY STORM"
            color = "\033[93m" # Yellow
        else:
            status = "🟢 NOMINAL - RESUME OPS"
            color = "\033[92m" # Green
        reset = "\033[0m"
        
        print(f"📅 {date_str} (T+{i+1} days)")
        print(f"   Conditions: Kp={kp:.1f} | Beta={beta:.1f}° (Factor={row['sun_exposure_factor']:.2f})")
        print(f"   Prediction: {decay:.1f} m/day ({percent:.0f}% of baseline)")
        print(f"   STATUS:     {status}")
        print("-" * 60)

if __name__ == "__main__":
    main()

import sqlite3
import pandas as pd
import numpy as np
from sgp4.api import Satrec, jday
from skyfield.api import EarthSatellite, load, wgs84
import logging

# --- CONFIGURATION ---
DB_NAME = "satellite_data.db"
OUTPUT_FILE = "final_training_set_v2.csv"  # UPGRADED TO V2 WITH F10.7 AND BC

# Updated "Golden Dataset"
TARGET_SATELLITES = [
    25544,  # ISS (High Sensitivity)
    48274,  # Tiangong (High Sensitivity)
    20580,  # Hubble (Pure Drag)
    33053,  # Fermi (Pure Drag)
    27386,  # Envisat (Heavyweight)
    25994,  # Terra (Heavyweight)
    5,      # Vanguard 1 (Control)
    11      # Vanguard 2 (Control)
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_midnight_epoch(date_obj):
    """Returns a TLE-compatible epoch for midnight of that day."""
    return jday(date_obj.year, date_obj.month, date_obj.day, 0, 0, 0)

def calculate_orbital_state(row, ts):
    """
    Propagates TLE to midnight and extracts Keplerian elements.
    Returns Series with: altitude_km, semi_major_axis_km, eccentricity, perigee_alt_km, mean_motion, is_circular
    """
    try:
        if not row.get('tle_line1') or not row.get('tle_line2'):
            return pd.Series([np.nan]*6)
            
        sat = EarthSatellite(row['tle_line1'], row['tle_line2'], str(row['norad_id']), ts)
        
        # 1. Physics Propagation (Midnight Baseline)
        epoch = row['epoch']
        t = ts.utc(epoch.year, epoch.month, epoch.day, 0, 0, 0)
        geocentric = sat.at(t)
        altitude_km = wgs84.height_of(geocentric).km
        
        # 2. Extract Keplerian Elements (SGP4 Model)
        eccentricity = sat.model.ecco
        mean_motion_rad_min = sat.model.no_kozai # radians/minute
        
        # 3. Derived Physics (Mean Elements)
        # Use Mean Motion to derive Semi-Major Axis (SMA) -> Stable Decay Metric
        MU = 398600.4418  # Earth Gravitational Parameter (km^3/s^2)
        EARTH_RADIUS_KM = 6378.137
        
        # Convert to radians/second
        mean_motion_rad_sec = mean_motion_rad_min / 60.0
        
        if mean_motion_rad_sec > 0:
            semi_major_axis_km = (MU / (mean_motion_rad_sec ** 2)) ** (1/3)
        else:
            semi_major_axis_km = np.nan
            
        perigee_alt_km = (semi_major_axis_km * (1 - eccentricity)) - EARTH_RADIUS_KM
        
        # 4. Smart Filtering Flag
        is_circular = eccentricity < 0.01
        
        return pd.Series([altitude_km, semi_major_axis_km, eccentricity, perigee_alt_km, mean_motion_rad_min, is_circular])
        
    except Exception as e:
        return pd.Series([np.nan]*6)

def run_merge_pipeline():
    print("🚀 Starting Final Merge Protocol (Phase 2)...")
    conn = sqlite3.connect(DB_NAME)
    
    # 1. LOAD FEATURES (Stream 2)
    print("   ↳ Loading Space Weather Features...")
    try:
        # Use the correct table name from feature_engineering.py
        weather_df = pd.read_sql("SELECT * FROM space_weather_features", conn)
        # Ensure we have date only for merging
        weather_df['date_key'] = pd.to_datetime(weather_df['time_tag']).dt.date
    except Exception as e:
        logger.error(f"Failed to load weather features: {e}")
        return
    
    # 1b. LOAD F10.7 SOLAR FLUX (NEW - S-TIER PHYSICS)
    print("   ↳ Loading F10.7 Solar Flux...")
    try:
        f107_df = pd.read_sql("SELECT * FROM solar_flux", conn)
        f107_df['date_key'] = pd.to_datetime(f107_df['date_tag']).dt.date
        print(f"     - Loaded {len(f107_df)} F10.7 records")
    except Exception as e:
        logger.warning(f"Failed to load F10.7 data: {e}")
        logger.warning("Continuing without F10.7 features - run ingestion first!")
        f107_df = pd.DataFrame()

    # 2. LOAD ORBITAL DATA (Stream 1)
    print("   ↳ Loading Raw TLEs...")
    placeholders = ','.join(['?'] * len(TARGET_SATELLITES))
    tle_query = f"SELECT * FROM tle_history WHERE norad_id IN ({placeholders})"
    tle_df = pd.read_sql(tle_query, conn, params=TARGET_SATELLITES)
    
    if tle_df.empty:
        logger.error("No TLE data found! Did you run ingestion?")
        return
        
    tle_df['epoch'] = pd.to_datetime(tle_df['epoch_datetime'])
    
    # 3. CALCULATE ORBITAL STATE (Physics Engine)
    print("   ↳ Propagating Orbits & Extracting Keplerian Elements...")
    ts = load.timescale()
    
    # Apply calculation and expand to columns
    orbital_cols = ['altitude_km', 'semi_major_axis_km', 'eccentricity', 'perigee_alt_km', 'mean_motion', 'is_circular']
    tle_df[orbital_cols] = tle_df.apply(lambda r: calculate_orbital_state(r, ts), axis=1)
    
    tle_df = tle_df.dropna(subset=['semi_major_axis_km'])
    
    # CRITICAL: Aggregate to one TLE per day per satellite
    # Multiple TLEs per day exist - take the first one for each date
    tle_df['date_only'] = tle_df['epoch'].dt.date
    tle_df = tle_df.sort_values(['norad_id', 'epoch'])
    tle_df = tle_df.drop_duplicates(subset=['norad_id', 'date_only'], keep='first')
    print(f"     - Aggregated to {len(tle_df)} unique (satellite, date) records")
    
    # 4. CALCULATE DAILY DROP (The Target Variable)
    # Sort by Satellite and Date
    tle_df = tle_df.sort_values(['norad_id', 'date_only'])
    
    # Calculate drops - now using date_only which is proper daily resolution
    # USE SEMI-MAJOR AXIS (MEAN ELEMENTS) FOR DECAY TO REMOVE GEOMETRIC NOISE
    tle_df['prev_sma'] = tle_df.groupby('norad_id')['semi_major_axis_km'].shift(1)
    tle_df['prev_date'] = tle_df.groupby('norad_id')['date_only'].shift(1)
    
    # Calculate day difference using date objects
    tle_df['days_diff'] = tle_df.apply(
        lambda r: (r['date_only'] - r['prev_date']).days if pd.notna(r['prev_date']) else None, 
        axis=1
    )
    
    # Strict Filter: Only accept consecutive days (diff = 1)
    clean_orbit = tle_df[tle_df['days_diff'] == 1].copy()
    
    # Target Y: Meters dropped in exactly 24 hours
    # Formula: (Prev_SMA - Current_SMA) * 1000
    clean_orbit['decay_rate_m'] = (clean_orbit['prev_sma'] - clean_orbit['semi_major_axis_km']) * 1000
    
    # 5. PHYSICS FILTER & MANEUVER DETECTION (3-STRIKE SYSTEM)
    print("   ↳ Applying Physics Filters (3-Strike Maneuver Detection)...")
    
    # ==============================================================================
    # MANEUVER DETECTION LOGIC
    # ==============================================================================
    # We flag rows that violate orbital mechanics physics.
    # These rows are KEPT for history (lags) but EXCLUDED from training (targets).
    
    # Strike 1: Anti-Gravity Check (Negative Decay)
    # Reality: Atmosphere creates friction. Friction only removes energy.
    # Meaning: Satellite gained altitude → thruster firing (reboost) or noisy data
    cond_anti_gravity = clean_orbit['decay_rate_m'] <= 0
    
    # Strike 2: Vacuum/Stale Data Check (Zero Decay)
    # Reality: In LEO, drag is continuous. It's never truly zero.
    # Meaning: Decay < 1cm usually means stale TLE (not updated) or sensor failure
    cond_stale = clean_orbit['decay_rate_m'] < 0.01
    
    # Strike 3: Crash Check (Impossible Speed)
    # Reality: Even during solar storms, stable LEO sats drop ~200-500m/day max
    # Meaning: Decay > 2km/day indicates de-orbit burn or severe tracking glitch
    cond_crash = clean_orbit['decay_rate_m'] > 2000.0
    
    # Combine all strikes (OR logic)
    clean_orbit['is_maneuver'] = cond_anti_gravity | cond_stale | cond_crash
    
    # Combine all strikes (OR logic)
    clean_orbit['is_maneuver'] = cond_anti_gravity | cond_stale | cond_crash
    
    # RETAIN ALL DATA (FLAG ONLY)
    # As per Senior Aerospace Scientist instruction: Do not remove data.
    # We keep all rows, including "impossible" maneuvers, trusting the flag to handle them downstream.
    valid_training_data = clean_orbit.copy()
    
    print(f"     - Flagged {clean_orbit['is_maneuver'].sum()} days as maneuvers (kept in dataset)")
    # print(f"     - Removed {clean_orbit['is_maneuver'].sum()} maneuver days")
    
    # 6. FEATURE ENGINEERING (Time)
    valid_training_data['doy'] = valid_training_data['epoch'].dt.dayofyear
    valid_training_data['sin_doy'] = np.sin(2 * np.pi * valid_training_data['doy'] / 365.25)
    valid_training_data['cos_doy'] = np.cos(2 * np.pi * valid_training_data['doy'] / 365.25)
    
    # 7. THE MERGE (FIX: Aggregate Weather to Daily FIRST)
    print("   ↳ Aggregating Weather to Daily Level...")
    valid_training_data['date_key'] = valid_training_data['epoch'].dt.date
    
    # CRITICAL FIX: One-to-Many Error
    # Weather data has 8 rows per day (3-hour intervals).
    # We must aggregate to 1 row per day BEFORE merging.
    daily_weather = weather_df.groupby('date_key').agg({
        'Kp_Current': ['mean', 'max', 'min'],
        'Kp_Lag_3h': 'mean',
        'Kp_Lag_6h': 'mean',
        'Kp_Lag_24h': 'mean',
        'Kp_Roll_24h_Mean': 'mean',
        'Kp_Roll_72h_Mean': 'mean'
    }).reset_index()
    
    # Flatten multi-level columns
    daily_weather.columns = [
        'date_key', 
        'Kp_mean', 'Kp_max', 'Kp_min',
        'Kp_Lag_3h', 'Kp_Lag_6h', 'Kp_Lag_24h',
        'Kp_Roll_24h_Mean', 'Kp_Roll_72h_Mean'
    ]
    
    print(f"     - Aggregated {len(weather_df)} hourly records → {len(daily_weather)} daily records")
    
    print("   ↳ Merging Streams (1:1 Inner Join)...")
    final_df = pd.merge(
        valid_training_data,
        daily_weather,
        on='date_key',
        how='inner'
    )
    
    # 7b. MERGE F10.7 SOLAR FLUX (NEW)
    if not f107_df.empty:
        print("   ↳ Merging F10.7 Solar Flux...")
        final_df = pd.merge(
            final_df,
            f107_df[['date_key', 'f107_obs', 'f107_adj']],
            on='date_key',
            how='left'
        )
        print(f"     - {final_df['f107_obs'].notna().sum()} rows have F10.7 data")
    else:
        final_df['f107_obs'] = np.nan
        final_df['f107_adj'] = np.nan
    
    # 8. PHYSICS-INFORMED FEATURE ENGINEERING (S-TIER BC CALCULATION)
    print("   ↳ Calculating Physics Features (Density Proxy & Ballistic Coefficient)...")
    
    # Step 1: Calculate Density Proxy
    # Formula: F10.7 drives EUV heating, Kp drives storm expansion
    # Altitude cubed approximates inverse density distribution
    if 'f107_obs' in final_df.columns and final_df['f107_obs'].notna().sum() > 0:
        final_df['density_proxy'] = (
            final_df['f107_obs'] * final_df['Kp_mean'] / 
            (final_df['altitude_km'] ** 3)
        )
        
        # Step 2: Calculate Observed BC (per row)
        # Ballistic Coefficient from drag equation: decay_rate ∝ density × BC
        # Therefore: BC ∝ decay_rate / density
        final_df['observed_bc'] = final_df['decay_rate_m'] / final_df['density_proxy']
        
        # Step 3: Calculate Static BC per satellite (TRAINING DATA ONLY)
        # Filter out maneuvers and outliers for clean median
        training_mask = (
            (final_df['is_maneuver'] == False) &
            (final_df['decay_rate_m'] > 0.01) &
            (final_df['decay_rate_m'] < 2000) &
            (final_df['observed_bc'].notna()) &
            (final_df['observed_bc'] > 0) &  # Positive BC only
            (final_df['observed_bc'] < 1e10)  # Remove infinite values
        )
        
        bc_stats = final_df[training_mask].groupby('norad_id').agg({
            'observed_bc': 'median'
        }).reset_index()
        bc_stats = bc_stats.rename(columns={'observed_bc': 'static_bc_est'})
        
        # Merge back to full dataset
        final_df = pd.merge(
            final_df,
            bc_stats,
            on='norad_id',
            how='left'
        )
        
        print(f"     - Calculated BC for {len(bc_stats)} satellites")
        print(f"     - BC statistics:")
        for _, row in bc_stats.iterrows():
            print(f"       • Satellite {row['norad_id']}: BC = {row['static_bc_est']:.6e}")
    else:
        print("     ⚠️ No F10.7 data available - skipping BC calculation")
        final_df['density_proxy'] = np.nan
        final_df['observed_bc'] = np.nan
        final_df['static_bc_est'] = np.nan
    
    # 9. FINAL CLEANUP
    # Drop rows with NaN in critical lag features (but keep F10.7 nans for now)
    lag_cols = [c for c in final_df.columns if 'Kp_' in c]
    final_df = final_df.dropna(subset=lag_cols)
    
    cols_to_keep = [
        'date_key', 'norad_id', 'satellite_name', 'altitude_km', 'semi_major_axis_km', 'decay_rate_m', 
        'is_maneuver', 'sin_doy', 'cos_doy',
        'eccentricity', 'perigee_alt_km', 'mean_motion', 'is_circular',
        'f107_obs', 'f107_adj', 'static_bc_est'  # NEW COLUMNS
    ] + lag_cols
    
    # Select available columns
    final_output = final_df[[c for c in cols_to_keep if c in final_df.columns]]
    
    # Rename date_key to date for cleanliness
    final_output = final_output.rename(columns={'date_key': 'date'})
    
    final_output.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ SUCCESS: Saved {len(final_output)} training samples to {OUTPUT_FILE}")
    
    # 9. HONESTY CHECK
    print("\n📊 DATA QUALITY REPORT:")
    for nid in final_output['norad_id'].unique():
        subset = final_output[final_output['norad_id'] == nid]
        sat_name = subset['satellite_name'].iloc[0] if 'satellite_name' in subset else "Unknown"
        avg_drop = subset['decay_rate_m'].mean()
        # Calculate circularity stats
        percent_circular = (subset['is_circular'].sum() / len(subset)) * 100 if 'is_circular' in subset else 0
        
        print(f"   - SAT {nid} ({sat_name}): {len(subset)} rows | Avg Drop: {avg_drop:.2f} m/day | {percent_circular:.1f}% Circular")
        
        if avg_drop < 0.5 and nid not in [5, 11]: # Not Vanguard
             print("     ⚠️ WARNING: This LEO satellite has suspiciously low drag.")
        elif avg_drop > 1000:
             print("     🔥 HIGH DRAG DETECTED (Valid for very low orbit)")

if __name__ == "__main__":
    run_merge_pipeline()
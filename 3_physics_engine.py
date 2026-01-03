"""
3_physics_engine.py - Vector-Based Astrodynamics Physics Engine
================================================================
Satellite Decay Prediction System: Geometric Physics Upgrade

This script transforms raw TLE data into physics-informed features by implementing:
1. Solar Beta Angle (Sun-Orbit Geometry) → Diurnal Bulge Exposure
2. F10.7 Solar Flux (Atmospheric Thermodynamics) → Background Temperature
3. Ballistic Coefficient Estimation (System Identification) → Drag Susceptibility

Physical Model:
- The thermosphere expands toward the Sun, creating a "Diurnal Bulge"
- Satellites crossing the bulge experience ~10x higher drag than on the night side
- Beta Angle (β) measures the angle between the orbital plane and Sun vector
- cos(β) ≈ 1.0 means the orbit passes through the bulge (maximum drag)
- cos(β) ≈ 0.0 means the orbit is edge-on to the Sun (terminator, minimum drag)

Author: Principal Astrodynamics Engineer
Phase: S-Tier Geometric Physics Upgrade
"""

import pandas as pd
import numpy as np
import sqlite3
import requests
from datetime import datetime
from typing import Tuple, Optional
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DB_PATH = "satellite_data.db"
INPUT_CSV = "final_training_set_v2.csv"
OUTPUT_CSV = "training_set_geometric.csv"
GFZ_URL = "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt"

# Physical Constants
EARTH_OBLIQUITY_DEG = 23.44  # Axial tilt (degrees)
J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)  # J2000.0 reference epoch


# =============================================================================
# PHASE 2A: SOLAR GEOMETRY (The "Diurnal Bulge")
# =============================================================================

def calculate_sun_vector(dates: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the Sun's unit vector in Earth-Centered Inertial (ECI) coordinates.
    
    Physics:
    --------
    The Sun's position varies throughout the year as Earth orbits.
    We approximate the Sun's direction using:
    1. Mean Longitude: λ_sun ≈ 280.46° + 0.9856474° × d (d = days since J2000)
    2. Ecliptic to Equatorial: Account for Earth's obliquity (ε = 23.44°)
    
    The Sun vector in ECI:
        S_x = cos(λ_sun)
        S_y = sin(λ_sun) × cos(ε)
        S_z = sin(λ_sun) × sin(ε)
    
    Args:
        dates: Pandas Series of datetime objects
        
    Returns:
        Tuple of (S_x, S_y, S_z) arrays - Sun unit vector components
    """
    # Convert dates to days since J2000
    dates_dt = pd.to_datetime(dates)
    days_since_j2000 = (dates_dt - pd.Timestamp(J2000_EPOCH)).dt.total_seconds() / 86400.0
    
    # Mean Solar Longitude (approximate formula, degrees)
    # λ_sun = 280.46° + 0.9856474° × d
    lambda_sun_deg = 280.46 + 0.9856474 * days_since_j2000
    lambda_sun_rad = np.radians(lambda_sun_deg % 360)  # Normalize to [0, 360)
    
    # Earth's Obliquity (radians)
    epsilon_rad = np.radians(EARTH_OBLIQUITY_DEG)
    
    # Sun vector in ECI (equatorial) coordinates
    # Transform from ecliptic to equatorial
    S_x = np.cos(lambda_sun_rad)
    S_y = np.sin(lambda_sun_rad) * np.cos(epsilon_rad)
    S_z = np.sin(lambda_sun_rad) * np.sin(epsilon_rad)
    
    return S_x.values, S_y.values, S_z.values


def calculate_orbit_normal(inclination_deg: np.ndarray, raan_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the orbit normal vector in ECI coordinates.
    
    Physics:
    --------
    The orbit normal (N) is perpendicular to the orbital plane.
    Given inclination (i) and RAAN (Ω):
        N_x = sin(i) × sin(Ω)
        N_y = -sin(i) × cos(Ω)
        N_z = cos(i)
    
    Args:
        inclination_deg: Orbital inclination array (degrees)
        raan_deg: Right Ascension of Ascending Node array (degrees)
        
    Returns:
        Tuple of (N_x, N_y, N_z) arrays - Orbit normal unit vector components
    """
    i_rad = np.radians(inclination_deg)
    omega_rad = np.radians(raan_deg)
    
    N_x = np.sin(i_rad) * np.sin(omega_rad)
    N_y = -np.sin(i_rad) * np.cos(omega_rad)
    N_z = np.cos(i_rad)
    
    return N_x, N_y, N_z


def calculate_solar_beta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Solar Beta Angle and Sun Exposure Factor for each orbit.
    
    Physics:
    --------
    Beta Angle (β): The angle between the orbital plane and the Sun vector.
        sin(β) = S · N (dot product of Sun vector and Orbit Normal)
        β = arcsin(S · N)
    
    Sun Exposure Factor:
        SEF = cos(β)
        - SEF = 1.0: Orbit plane contains Sun → Maximum drag (diurnal bulge crossing)
        - SEF = 0.0: Orbit plane perpendicular to Sun → Terminator orbit
    
    Args:
        df: DataFrame with columns ['date', 'inclination_deg', 'raan_deg']
        
    Returns:
        DataFrame with added columns ['beta_angle_deg', 'sun_exposure_factor']
    """
    print("   ↳ Calculating Solar Beta Angles (Vector Math)...")
    
    # Check required columns
    required_cols = ['date', 'inclination_deg', 'raan_deg']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Calculate Sun vector (S)
    S_x, S_y, S_z = calculate_sun_vector(df['date'])
    
    # Calculate Orbit Normal (N)
    N_x, N_y, N_z = calculate_orbit_normal(
        df['inclination_deg'].values, 
        df['raan_deg'].values
    )
    
    # Beta Angle: sin(β) = S · N
    sin_beta = S_x * N_x + S_y * N_y + S_z * N_z
    
    # Clamp to [-1, 1] to avoid numerical errors in arcsin
    sin_beta = np.clip(sin_beta, -1.0, 1.0)
    
    # Calculate Beta Angle (degrees)
    beta_rad = np.arcsin(sin_beta)
    df['beta_angle_deg'] = np.degrees(beta_rad)
    
    # Sun Exposure Factor = cos(β)
    # High exposure (1.0) = orbit passes through diurnal bulge
    # Low exposure (0.0) = orbit is edge-on to Sun
    df['sun_exposure_factor'] = np.cos(beta_rad)
    
    print(f"     ├── Beta Angle Range: {df['beta_angle_deg'].min():.1f}° to {df['beta_angle_deg'].max():.1f}°")
    print(f"     └── Sun Exposure Factor Range: {df['sun_exposure_factor'].min():.3f} to {df['sun_exposure_factor'].max():.3f}")
    
    return df


# =============================================================================
# PHASE 2B: ATMOSPHERIC THERMODYNAMICS (F10.7)
# =============================================================================

def fetch_and_merge_solar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch F10.7 solar flux data and merge into the main DataFrame.
    
    Physics:
    --------
    F10.7 (10.7 cm radio flux) is a proxy for solar EUV radiation.
    - Higher F10.7 → More EUV → Hotter thermosphere → Higher density → More drag
    - F10.7_81d_Avg represents the "background" thermospheric temperature
    - Daily F10.7 captures short-term solar activity
    
    Args:
        df: DataFrame with 'date' column
        
    Returns:
        DataFrame with added columns ['f107_obs', 'f107_81d_avg']
    """
    print("   ↳ Loading F10.7 Solar Flux Data...")
    
    # Check if already in dataframe
    if 'f107_obs' in df.columns and df['f107_obs'].notna().sum() > len(df) * 0.9:
        print("     └── F10.7 already present, calculating 81-day average only...")
    else:
        # Try loading from database first
        try:
            conn = sqlite3.connect(DB_PATH)
            f107_df = pd.read_sql("SELECT date_tag, f107_obs, f107_adj FROM solar_flux", conn)
            conn.close()
            
            if len(f107_df) > 0:
                f107_df['date'] = pd.to_datetime(f107_df['date_tag'])
                f107_df = f107_df[['date', 'f107_obs', 'f107_adj']]
                
                # Merge into main dataframe
                df['date'] = pd.to_datetime(df['date'])
                df = df.merge(f107_df, on='date', how='left', suffixes=('', '_new'))
                
                # Handle potential column conflicts
                if 'f107_obs_new' in df.columns:
                    df['f107_obs'] = df['f107_obs_new'].combine_first(df.get('f107_obs', pd.Series()))
                    df = df.drop(columns=['f107_obs_new'], errors='ignore')
                
                print(f"     ├── Loaded {len(f107_df)} F10.7 records from database")
        except Exception as e:
            print(f"     ⚠ Failed to load from database: {e}")
    
    # Calculate 81-day centered average
    print("     ├── Calculating 81-day centered average...")
    
    # Get unique dates to avoid satellite duplication issues
    df['date'] = pd.to_datetime(df['date'])
    unique_weather = df[['date', 'f107_obs']].drop_duplicates().sort_values('date').set_index('date')
    
    # Rolling mean with center=True
    unique_weather['f107_81d_avg'] = unique_weather['f107_obs'].rolling(
        window=81, center=True, min_periods=1
    ).mean()
    
    # Fill edges
    unique_weather['f107_81d_avg'] = unique_weather['f107_81d_avg'].ffill().bfill()
    
    # Merge back
    df = df.merge(unique_weather[['f107_81d_avg']], on='date', how='left', suffixes=('', '_new'))
    if 'f107_81d_avg_new' in df.columns:
        df['f107_81d_avg'] = df['f107_81d_avg_new'].combine_first(df.get('f107_81d_avg', pd.Series()))
        df = df.drop(columns=['f107_81d_avg_new'], errors='ignore')
    
    coverage = df['f107_81d_avg'].notna().sum() / len(df) * 100
    print(f"     └── F10.7 Coverage: {coverage:.1f}%")
    
    return df


# =============================================================================
# PHASE 2C: SYSTEM IDENTIFICATION (Reverse Engineering BC)
# =============================================================================

def estimate_ballistic_coefficient(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate the Ballistic Coefficient (BC) for each satellite using physics inversion.
    
    Physics:
    --------
    Drag acceleration: a_drag = (1/2) × ρ × v² × BC
    Where BC = C_d × A / m (drag coefficient × area / mass)
    
    We don't have v² or ρ directly, so we use a proxy:
        ρ_proxy ∝ (F10.7 × Kp) / h³
        BC_inst = DecayRate / ρ_proxy
    
    We then take the MEDIAN BC per satellite (robust to outliers/maneuvers).
    
    Why MEDIAN?: 
    - Individual measurements are noisy
    - Maneuvers create outliers
    - Median is robust to both
    
    Args:
        df: DataFrame with columns ['norad_id', 'decay_rate_m', 'f107_obs', 'Kp_mean', 'altitude_km']
        
    Returns:
        DataFrame with added column ['static_bc_est']
    """
    print("   ↳ Estimating Ballistic Coefficients (Physics Inversion)...")
    
    # Calculate density proxy
    # ρ ∝ F10.7 × Kp / h³ (simplified thermospheric model)
    df['density_proxy'] = (
        df['f107_obs'] * df['Kp_mean'] / 
        (df['altitude_km'] ** 3)
    )
    
    # Instantaneous BC = DecayRate / Density
    df['bc_inst'] = df['decay_rate_m'] / df['density_proxy']
    
    # Replace infinities and zeros with NaN
    df['bc_inst'] = df['bc_inst'].replace([np.inf, -np.inf], np.nan)
    
    # Filter: Only use "clean" physics regime for BC estimation
    # - No maneuvers
    # - Positive decay
    # - Reasonable BC range
    clean_mask = (
        (df.get('is_maneuver', False) == False) &
        (df['decay_rate_m'] > 0.01) &
        (df['decay_rate_m'] < 2000) &
        (df['bc_inst'].notna()) &
        (df['bc_inst'] > 0)
    )
    
    # Set dirty/maneuver BCs to NaN so they don't pollute the rolling window
    # We create a temporary column for this calculation
    df['bc_clean'] = df['bc_inst'].where(clean_mask, np.nan)
    
    # Ensure sorted by date
    df = df.sort_values(['norad_id', 'date'])
    
    print("     ├── Calculating 30-day Rolling Median BC...")
    
    # Calculate Rolling Median BC (30D window)
    # Must use iteration because transform() with time-window is tricky without index
    rolling_results = []
    
    for nid, group in df.groupby('norad_id'):
        # Set date as index for time-based rolling
        g_indexed = group.set_index('date').sort_index()
        
        # rolling 30D, center=True
        # min_periods=1 ensures we get values even with sparse data
        r_median = g_indexed['bc_clean'].rolling('30D', min_periods=1, center=True).median()
        
        # Prepare for merge
        r_df = r_median.reset_index()
        r_df.columns = ['date', 'static_bc_est']
        r_df['norad_id'] = nid
        rolling_results.append(r_df)
    
    # Combine all results
    bc_frames = pd.concat(rolling_results, ignore_index=True)
    
    # Merge back to main dataframe
    if 'static_bc_est' in df.columns:
        df = df.drop(columns=['static_bc_est'])
        
    # Ensure merge keys match
    df['date'] = pd.to_datetime(df['date'])
    bc_frames['date'] = pd.to_datetime(bc_frames['date'])
    
    df = df.merge(bc_frames, on=['norad_id', 'date'], how='left')
    
    # Fallback: Fill gaps (start/end of window) with global median per satellite
    global_medians = df.groupby('norad_id')['bc_clean'].median()
    
    def fill_bc_gaps(row):
        if pd.isna(row['static_bc_est']):
            return global_medians.get(row['norad_id'], np.nan)
        return row['static_bc_est']
        
    df['static_bc_est'] = df.apply(fill_bc_gaps, axis=1)
    
    # Stats
    bc_stats = df.groupby('norad_id')['static_bc_est'].median().reset_index()
    
    print(f"     ├── Estimated Rolling BC for {len(bc_stats)} satellites")
    for _, row in bc_stats.iterrows():
        print(f"     │   • Satellite {int(row['norad_id'])}: Median BC = {row['static_bc_est']:.4e}")
    
    # Cleanup intermediate columns
    df = df.drop(columns=['density_proxy', 'bc_inst', 'bc_clean'], errors='ignore')
    
    return df


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_physics_engine():
    """
    Main execution pipeline for the Physics Engine.
    """
    print("=" * 78)
    print("🔬 ASTRODYNAMICS PHYSICS ENGINE - GEOMETRIC UPGRADE")
    print("   Implementing Solar Beta Angle, F10.7, and Ballistic Coefficients")
    print("=" * 78)
    
    # Load raw data
    print("\n📂 [1/4] Loading Raw Training Data...")
    df = pd.read_csv(INPUT_CSV)
    print(f"   └── Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Check if geometric columns exist
    print("\n🔍 [2/4] Checking for Geometric Columns...")
    geometric_cols = ['inclination_deg', 'raan_deg']
    missing_geo = [c for c in geometric_cols if c not in df.columns]
    
    if missing_geo:
        print(f"   ⚠ Missing columns: {missing_geo}")
        print("   → Attempting to load from TLE database...")
        
        # Try to load from database
        try:
            conn = sqlite3.connect(DB_PATH)
            tle_df = pd.read_sql("""
                SELECT norad_id, epoch_datetime, inclination as inclination_deg, 
                       raan as raan_deg
                FROM tle_history
            """, conn)
            conn.close()
            
            tle_df['date'] = pd.to_datetime(tle_df['epoch_datetime']).dt.date
            tle_df = tle_df.drop(columns=['epoch_datetime'])
            
            # Convert date for merging
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Merge geometric data
            df = df.merge(tle_df, on=['norad_id', 'date'], how='left', suffixes=('', '_tle'))
            
            # Use TLE values if original missing
            for col in geometric_cols:
                if col not in df.columns and f'{col}_tle' in df.columns:
                    df[col] = df[f'{col}_tle']
                df = df.drop(columns=[f'{col}_tle'], errors='ignore')
            
            print(f"   ✓ Loaded geometric data from TLE database")
        except Exception as e:
            print(f"   ✗ Failed to load geometric data: {e}")
            print("   → Generating synthetic inclination/RAAN based on satellite characteristics...")
            
            # Fallback: Assign typical values based on known satellite characteristics
            # This is a workaround; real implementation should parse TLEs
            sat_geometry = {
                5: {'inclination_deg': 34.25, 'raan_deg': 0},      # Vanguard 1
                11: {'inclination_deg': 32.89, 'raan_deg': 0},     # Vanguard 2
                20580: {'inclination_deg': 28.47, 'raan_deg': 0},  # HST
                25544: {'inclination_deg': 51.64, 'raan_deg': 0},  # ISS
                25994: {'inclination_deg': 98.21, 'raan_deg': 0},  # Terra (Sun-sync)
                27386: {'inclination_deg': 98.52, 'raan_deg': 0},  # Envisat
                33053: {'inclination_deg': 25.58, 'raan_deg': 0},  # GLAST/Fermi
                48274: {'inclination_deg': 41.47, 'raan_deg': 0},  # CSS Tianhe
            }
            
            for norad_id, geo in sat_geometry.items():
                mask = df['norad_id'] == norad_id
                df.loc[mask, 'inclination_deg'] = geo['inclination_deg']
                # RAAN varies; approximate based on day of year
                df.loc[mask, 'raan_deg'] = (pd.to_datetime(df.loc[mask, 'date']).dt.dayofyear * 0.9856) % 360
            
            print(f"   ✓ Generated synthetic geometric data for {len(sat_geometry)} satellites")
    else:
        print(f"   ✓ Geometric columns present: {geometric_cols}")
    
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Phase 2A: Solar Geometry
    print("\n☀️ [3/4] Phase 2A: Solar Geometry (Beta Angle)...")
    df = calculate_solar_beta(df)
    
    # Phase 2B: Atmospheric Thermodynamics
    print("\n🌡️ [3/4] Phase 2B: Atmospheric Thermodynamics (F10.7)...")
    df = fetch_and_merge_solar(df)
    
    # Phase 2C: Ballistic Coefficient
    print("\n⚙️ [3/4] Phase 2C: System Identification (BC Estimation)...")
    df = estimate_ballistic_coefficient(df)
    
    # Final Cleanup
    print("\n💾 [4/4] Saving Enriched Dataset...")
    
    # Select key columns
    key_cols = [
        'date', 'norad_id', 'satellite_name',
        # Orbital
        'semi_major_axis_km', 'altitude_km', 'perigee_alt_km', 
        'eccentricity', 'mean_motion', 'is_circular',
        # Geometry (NEW)
        'inclination_deg', 'raan_deg', 'beta_angle_deg', 'sun_exposure_factor',
        # Space Weather
        'Kp_mean', 'Kp_max', 'Kp_Lag_24h', 'Kp_Roll_24h_Mean', 'Kp_Roll_72h_Mean',
        # Solar Flux
        'f107_obs', 'f107_81d_avg',
        # Physics
        'static_bc_est',
        # Seasonality
        'sin_doy', 'cos_doy',
        # Quality
        'is_maneuver',
        # Target
        'decay_rate_m'
    ]
    
    # Only keep columns that exist
    available_cols = [c for c in key_cols if c in df.columns]
    df_out = df[available_cols].copy()
    
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"   └── Saved {len(df_out):,} rows to {OUTPUT_CSV}")
    
    # Summary
    print("\n" + "=" * 78)
    print("✅ PHYSICS ENGINE COMPLETE")
    print("=" * 78)
    print(f"""
New Features Added:
  • beta_angle_deg       : Solar Beta Angle (orbit-sun geometry)
  • sun_exposure_factor  : cos(β), diurnal bulge exposure [0-1]
  • f107_81d_avg         : 81-day centered F10.7 average
  • static_bc_est        : Median Ballistic Coefficient per satellite

Output: {OUTPUT_CSV}
""")
    
    return df_out


if __name__ == "__main__":
    run_physics_engine()

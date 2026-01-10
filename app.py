"""
🚀 ORBITAL DECAY PREDICTOR - Space-Themed Streamlit Dashboard
==============================================================
A stunning, modern interface for satellite decay prediction using
the Geometric Physics Ensemble Model.

Author: Project B-Star Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import os
from xgboost import XGBRegressor
import warnings

# Import Space-Track and Physics Engine
try:
    from etl_pipeline import SpaceTrackIngestor
    SPACETRACK_AVAILABLE = True
except ImportError:
    SPACETRACK_AVAILABLE = False

try:
    from module_3_physics_engine import calculate_solar_beta, fetch_and_merge_solar, estimate_ballistic_coefficient
    PHYSICS_ENGINE_AVAILABLE = True
except ImportError:
    try:
        # Try direct import from 3_physics_engine.py (handle numeric prefix)
        import importlib.util
        spec = importlib.util.spec_from_file_location("physics_engine", "3_physics_engine.py")
        physics_engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(physics_engine)
        calculate_solar_beta = physics_engine.calculate_solar_beta
        fetch_and_merge_solar = physics_engine.fetch_and_merge_solar
        estimate_ballistic_coefficient = physics_engine.estimate_ballistic_coefficient
        PHYSICS_ENGINE_AVAILABLE = True
    except Exception:
        PHYSICS_ENGINE_AVAILABLE = False

warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Orbital Decay Predictor",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS - SPACE THEME
# =============================================================================
st.markdown("""
<style>
    /* Import Space Grotesk font for space-aligned typography */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Orbitron:wght@400;500;600;700;800;900&display=swap');
    
    /* Global text rendering reset - CRITICAL for preventing overlap */
    * {
        text-rendering: optimizeLegibility !important;
        -webkit-font-smoothing: antialiased !important;
        -moz-osx-font-smoothing: grayscale !important;
    }
    
    /* Force text not to overlap */
    [data-testid="stSidebar"] * {
        position: relative !important;
        z-index: auto !important;
    }
    
    /* Root variables */
    :root {
        --bg-deep-space: #0a0a1a;
        --bg-card: rgba(30, 30, 60, 0.6);
        --accent-cyan: #00d4ff;
        --accent-green: #00ff88;
        --accent-yellow: #ffd700;
        --accent-red: #ff4444;
        --text-primary: #ffffff;
        --text-secondary: #8b8ba7;
        --glow-cyan: 0 0 20px rgba(0, 212, 255, 0.5);
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3a 50%, #0a0a1a 100%);
        background-attachment: fixed;
    }
    
    /* Starfield animation */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, #fff, transparent),
            radial-gradient(2px 2px at 40px 70px, rgba(255,255,255,0.8), transparent),
            radial-gradient(1px 1px at 90px 40px, #fff, transparent),
            radial-gradient(2px 2px at 160px 120px, rgba(255,255,255,0.9), transparent),
            radial-gradient(1px 1px at 230px 80px, #fff, transparent),
            radial-gradient(2px 2px at 300px 150px, rgba(255,255,255,0.7), transparent),
            radial-gradient(1px 1px at 350px 200px, #fff, transparent),
            radial-gradient(2px 2px at 400px 50px, rgba(255,255,255,0.8), transparent),
            radial-gradient(1px 1px at 500px 180px, #fff, transparent),
            radial-gradient(2px 2px at 600px 100px, rgba(255,255,255,0.9), transparent);
        background-size: 650px 250px;
        animation: twinkle 8s ease-in-out infinite alternate;
        pointer-events: none;
        z-index: 0;
        opacity: 0.6;
    }
    
    @keyframes twinkle {
        0% { opacity: 0.4; }
        100% { opacity: 0.8; }
    }
    
    /* Center main content - ROBUST FIX */
    .block-container {
        max-width: 1000px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
        width: 100% !important;
    }
    
    /* Ensure the main view adapts correctly */
    .main .block-container {
        transition: margin 0.3s ease;
    }
    
    /* Typography */
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Orbitron', sans-serif !important;
        color: var(--text-primary) !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    p, span, label, .stMarkdown p {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text-secondary) !important;
    }
    
    /* Sidebar styling - Fixed width and hamburger icon */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10,10,26,0.95) 0%, rgba(20,20,50,0.95) 100%);
        border-right: 1px solid rgba(0, 212, 255, 0.2);
        min-width: 280px !important;
        max-width: 280px !important;
    }
    
    /* Hide sidebar collapse controls completely */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    button[kind="headerNoPadding"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Ensure arrow replacement is also hidden if it exists */
    [data-testid="collapsedControl"]::after,
    button[kind="headerNoPadding"]::after {
        display: none !important;
        content: "" !important;
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--accent-cyan) !important;
        text-shadow: var(--glow-cyan);
    }
    
    /* Selected button highlight */
    .date-btn-selected {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.3) 0%, rgba(0, 255, 136, 0.5) 100%) !important;
        border: 2px solid #00ff88 !important;
        color: #00ff88 !important;
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.4) !important;
    }
    
    .date-btn-normal {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 212, 255, 0.2) 100%) !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
    }
    
    /* Glass cards */
    .glass-card {
        background: rgba(30, 30, 60, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(0, 212, 255, 0.5);
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.2);
    }
    
    /* Metric cards - Uniform heights with flexbox */
    .metrics-row {
        display: flex;
        gap: 16px;
        margin: 20px 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(30, 30, 60, 0.6) 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 12px;
        padding: 24px 16px;
        text-align: center;
        flex: 1;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .metric-value {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent-cyan) !important;
        text-shadow: var(--glow-cyan);
        line-height: 1.2;
        word-break: break-word;
    }
    
    .metric-label {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.75rem;
        color: var(--text-secondary) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 12px;
        line-height: 1.3;
    }
    
    /* Risk indicators */
    .risk-green { color: var(--accent-green) !important; text-shadow: 0 0 15px rgba(0, 255, 136, 0.5); }
    .risk-yellow { color: var(--accent-yellow) !important; text-shadow: 0 0 15px rgba(255, 215, 0, 0.5); }
    .risk-red { color: var(--accent-red) !important; text-shadow: 0 0 15px rgba(255, 68, 68, 0.5); }
    
    /* Main content buttons */
    .stButton > button {
        font-family: 'Space Grotesk', sans-serif !important;
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.2) 0%, rgba(0, 212, 255, 0.4) 100%);
        border: 1px solid var(--accent-cyan);
        color: var(--accent-cyan) !important;
        border-radius: 8px;
        padding: 10px 20px;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        width: 100%;
        font-size: 0.8rem !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.4) 0%, rgba(0, 212, 255, 0.6) 100%);
        box-shadow: var(--glow-cyan);
    }
    
    /* PRIMARY button style (selected state) - PERMANENT HIGHLIGHT */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.3) 0%, rgba(0, 255, 136, 0.5) 100%) !important;
        border: 2px solid #00ff88 !important;
        color: #00ff88 !important;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.4) !important;
        font-weight: 600 !important;
    }
    
    /* Secondary button (unselected) */
    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="baseButton-secondary"] {
        background: rgba(30, 30, 60, 0.6) !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
        color: var(--text-secondary) !important;
        box-shadow: none !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: rgba(0, 212, 255, 0.2) !important;
        border-color: var(--accent-cyan) !important;
        color: var(--accent-cyan) !important;
    }
    
    /* Sidebar specific button sizing */
    [data-testid="stSidebar"] .stButton > button {
        font-size: 0.75rem !important;
        padding: 10px 8px !important;
        letter-spacing: 0.5px !important;
        min-height: 42px !important;
    }
    
    /* Sidebar text and labels - LARGER */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p {
        font-size: 0.85rem !important;
    }
    
    [data-testid="stSidebar"] strong {
        font-size: 0.9rem !important;
        color: #ffffff !important;
    }
    
    /* Slider labels in sidebar */
    [data-testid="stSidebar"] .stSlider label {
        font-size: 0.8rem !important;
        color: var(--text-secondary) !important;
    }
    
    /* Select boxes and inputs */
    .stSelectbox > div > div, .stDateInput > div > div > input {
        background: rgba(30, 30, 60, 0.8) !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: var(--accent-cyan) !important;
    }
    
    /* Expander - Complete fix for overlapping text */
    [data-testid="stExpander"] {
        background: rgba(30, 30, 60, 0.6) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 8px !important;
        overflow: visible !important;
    }
    
    [data-testid="stExpander"] summary {
        padding: 12px 16px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.85rem !important;
    }
    
    [data-testid="stExpander"] summary span {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text-secondary) !important;
    }
    
    /* Legacy expander selector */
    .streamlit-expanderHeader {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    /* Sidebar spacing fixes */
    [data-testid="stSidebar"] .stMarkdown h4,
    [data-testid="stSidebar"] .stMarkdown h5 {
        margin-top: 12px !important;
        margin-bottom: 8px !important;
        font-size: 0.85rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        font-size: 0.8rem !important;
    }
    
    [data-testid="stSidebar"] hr {
        margin: 12px 0 !important;
        border-color: rgba(0, 212, 255, 0.2) !important;
    }
    
    /* Fix info boxes in sidebar */
    [data-testid="stSidebar"] .stAlert {
        padding: 10px 12px !important;
        font-size: 0.8rem !important;
    }
    
    [data-testid="stSidebar"] .stAlert p {
        font-size: 0.8rem !important;
        line-height: 1.4 !important;
    }
    
    /* Hide Streamlit branding and toolbar */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stDecoration"] {visibility: hidden;}
    [data-testid="stHeader"] {background: transparent;}
    
    /* Processing animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.05); }
    }
    
    .processing-text {
        font-family: 'Orbitron', sans-serif;
        color: var(--accent-cyan);
        animation: pulse 1.5s ease-in-out infinite;
        text-align: center;
        font-size: 1.2rem;
    }
    
    /* Recommendation box */
    .recommendation-box {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(30, 30, 60, 0.6) 100%);
        border: 1px solid rgba(0, 255, 136, 0.3);
        border-radius: 12px;
        padding: 24px;
        margin-top: 24px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    .recommendation-box.warning {
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.1) 0%, rgba(30, 30, 60, 0.6) 100%);
        border-color: rgba(255, 215, 0, 0.3);
    }
    
    .recommendation-box.critical {
        background: linear-gradient(135deg, rgba(255, 68, 68, 0.1) 0%, rgba(30, 30, 60, 0.6) 100%);
        border-color: rgba(255, 68, 68, 0.3);
    }
    
    /* Data freshness notice */
    .data-freshness {
        background: rgba(0, 212, 255, 0.1);
        border-left: 3px solid var(--accent-cyan);
        padding: 12px 16px;
        margin: 16px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================
DATA_PATH = "training_set_geometric.csv"

FEATURE_COLUMNS = [
    'sun_exposure_factor', 'beta_angle_deg',
    'f107_obs', 'f107_81d_avg',
    'Kp_mean', 'Kp_max', 'Kp_Lag_24h',
    'semi_major_axis_km', 'altitude_km', 'perigee_alt_km', 'eccentricity',
    'sin_doy', 'cos_doy',
    'static_bc_est'
]

MODEL_PARAMS = {
    'objective': 'reg:squarederror',
    'n_estimators': 5000,
    'learning_rate': 0.01,
    'max_depth': 8,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'n_jobs': -1,
    'verbosity': 0,
    'random_state': 42
}

# Operator Recommendations based on risk level
RECOMMENDATIONS = {
    'green': {
        'title': '🟢 NOMINAL OPERATIONS',
        'message': 'All systems nominal. Continue standard operations.',
        'actions': [
            '✓ Maintain current orbit monitoring schedule',
            '✓ No fuel conservation measures required',
            '✓ Proceed with planned maneuvers if any'
        ]
    },
    'yellow': {
        'title': '🟡 ELEVATED DRAG ADVISORY',
        'message': 'Increased atmospheric drag detected. Enhanced monitoring recommended.',
        'actions': [
            '⚠️ Increase telemetry monitoring frequency',
            '⚠️ Prepare for potential orbit maintenance burn',
            '⚠️ Review fuel reserves and contingency plans',
            '⚠️ Alert ground control for standby support'
        ]
    },
    'red': {
        'title': '🔴 CRITICAL DRAG EVENT',
        'message': 'Severe atmospheric drag spike predicted. Immediate action required.',
        'actions': [
            '🚨 Execute orbit-raising maneuver if fuel permits',
            '🚨 Reduce cross-sectional area (safe mode orientation)',
            '🚨 Suspend non-essential operations',
            '🚨 Continuous orbit tracking required',
            '🚨 Prepare collision avoidance procedures'
        ]
    }
}


# =============================================================================
# SPACE-TRACK API CONFIGURATION
# =============================================================================
# Load credentials from secrets.toml or environment variables
def get_spacetrack_credentials():
    """Get Space-Track credentials from Streamlit secrets or environment."""
    try:
        user = st.secrets.get("SPACETRACK_USER", os.getenv("SPACETRACK_USER", ""))
        pwd = st.secrets.get("SPACETRACK_PASS", os.getenv("SPACETRACK_PASS", ""))
        return user, pwd
    except Exception:
        return os.getenv("SPACETRACK_USER", ""), os.getenv("SPACETRACK_PASS", "")

DB_PATH = "satellite_data.db"

@st.cache_resource
def get_space_track_ingestor():
    """Get cached SpaceTrack ingestor instance (if credentials available)."""
    if not SPACETRACK_AVAILABLE:
        return None
    user, pwd = get_spacetrack_credentials()
    if user and pwd:
        return SpaceTrackIngestor(user, pwd, DB_PATH)
    return None


def fetch_custom_satellite_data(norad_id, target_date=None):
    """
    Fetch satellite data for a custom NORAD ID from Space-Track API.
    Uses physics engine to calculate required features.
    
    Args:
        norad_id: NORAD catalog ID
        target_date: If backtesting, the historical date to use
        
    Returns:
        DataFrame with satellite features, or None if fetch failed
    """
    ingestor = get_space_track_ingestor()
    if ingestor is None:
        return None, "Space-Track credentials not configured"
    
    try:
        # Fetch TLE data from Space-Track (with caching)
        with st.spinner(f"📡 Fetching data for NORAD ID {norad_id} from Space-Track..."):
            tle_df = ingestor.fetch_history(norad_id, limit=1000)
        
        if tle_df is None or len(tle_df) == 0:
            return None, f"No TLE data found for NORAD ID {norad_id}"
        
        # =================================================================
        # MAP SPACE-TRACK COLUMNS TO PHYSICS ENGINE FORMAT
        # =================================================================
        column_mapping = {
            'EPOCH': 'date',
            'epoch_datetime': 'date',
            'INCLINATION': 'inclination_deg',
            'inclination': 'inclination_deg',
            'RA_OF_ASC_NODE': 'raan_deg',
            'MEAN_MOTION': 'mean_motion_rev_day',
            'ECCENTRICITY': 'eccentricity',
            'eccentricity': 'eccentricity',
            'NORAD_CAT_ID': 'norad_id',
            'norad_id': 'norad_id',
            'OBJECT_NAME': 'satellite_name',
            'satellite_name': 'satellite_name',
            'SEMIMAJOR_AXIS': 'semi_major_axis_km',
            'APOAPSIS': 'apogee_alt_km',
            'PERIAPSIS': 'perigee_alt_km',
            'mean_motion': 'mean_motion_rev_day'
        }
        
        # Rename columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in tle_df.columns and new_name not in tle_df.columns:
                tle_df[new_name] = tle_df[old_name]
        
        # explicit type conversion for numeric columns
        numeric_cols = [
            'inclination_deg', 'raan_deg', 'mean_motion_rev_day', 'eccentricity', 
            'semi_major_axis_km', 'apogee_alt_km', 'perigee_alt_km', 'decay_rate_m',
            'bstar', 'BSTAR', 'altitude_km', 'mean_motion', 'inclination'
        ]
        
        for col in numeric_cols:
            if col in tle_df.columns:
                 tle_df[col] = pd.to_numeric(tle_df[col], errors='coerce')

        # Ensure date column is datetime
        if 'date' in tle_df.columns:
            tle_df['date'] = pd.to_datetime(tle_df['date'], utc=True, errors='coerce')
        
        # Calculate orbital parameters from mean motion if not present
        # Try both column name variants
        mean_motion_col = None
        if 'mean_motion_rev_day' in tle_df.columns:
            mean_motion_col = 'mean_motion_rev_day'
        elif 'mean_motion' in tle_df.columns:
            mean_motion_col = 'mean_motion'
        
        if 'semi_major_axis_km' not in tle_df.columns and mean_motion_col is not None:
            # a = (mu / (2*pi*n)^2)^(1/3)
            # For Earth: mu = 398600.4 km³/s²
            mu = 398600.4  # km³/s²
            mean_motion_rad_s = tle_df[mean_motion_col] * 2 * np.pi / 86400
            tle_df['semi_major_axis_km'] = (mu / (mean_motion_rad_s ** 2)) ** (1/3)
            # Debug output
            print(f"[DEBUG] Calculated semi-major axis for NORAD {norad_id}: {tle_df['semi_major_axis_km'].iloc[-1]:.2f} km")
        
        if 'altitude_km' not in tle_df.columns and 'semi_major_axis_km' in tle_df.columns:
            tle_df['altitude_km'] = tle_df['semi_major_axis_km'] - 6371.0  # Earth radius
            print(f"[DEBUG] Calculated altitude for NORAD {norad_id}: {tle_df['altitude_km'].iloc[-1]:.2f} km")
        
        if 'perigee_alt_km' not in tle_df.columns:
            if 'semi_major_axis_km' in tle_df.columns and 'eccentricity' in tle_df.columns:
                tle_df['perigee_alt_km'] = tle_df['semi_major_axis_km'] * (1 - tle_df['eccentricity'].astype(float)) - 6371.0
            else:
                tle_df['perigee_alt_km'] = tle_df.get('altitude_km', 400.0)
        
        # Add norad_id if missing
        if 'norad_id' not in tle_df.columns:
            tle_df['norad_id'] = norad_id
        
        # Process with physics engine if available AND required columns exist
        required_cols = ['date', 'inclination_deg', 'raan_deg']
        has_required = all(col in tle_df.columns for col in required_cols)
        
        if PHYSICS_ENGINE_AVAILABLE and has_required:
            with st.spinner("⚙️ Computing physics features..."):
                # Calculate solar geometry (beta angle)
                tle_df = calculate_solar_beta(tle_df)
                
                # Merge solar flux data
                tle_df = fetch_and_merge_solar(tle_df)
                
                # Estimate ballistic coefficient (needs decay_rate_m, add placeholder if missing)
                if 'decay_rate_m' not in tle_df.columns:
                    tle_df['decay_rate_m'] = 10.0  # Placeholder
                if 'Kp_mean' not in tle_df.columns:
                    tle_df['Kp_mean'] = 3.0  # Default
                
                tle_df = estimate_ballistic_coefficient(tle_df)
        else:
            # Fallback: Add placeholder physics values
            if 'beta_angle_deg' not in tle_df.columns:
                tle_df['beta_angle_deg'] = 30.0
            if 'sun_exposure_factor' not in tle_df.columns:
                tle_df['sun_exposure_factor'] = 0.8
            if 'static_bc_est' not in tle_df.columns:
                tle_df['static_bc_est'] = 1.0e7
            if 'f107_obs' not in tle_df.columns:
                tle_df['f107_obs'] = 150.0
            if 'f107_81d_avg' not in tle_df.columns:
                tle_df['f107_81d_avg'] = 150.0
            if 'decay_rate_m' not in tle_df.columns:
                tle_df['decay_rate_m'] = 10.0
        
        # If target date specified (backtest), filter to that date
        if target_date is not None and 'date' in tle_df.columns:
            # Convert target_date to timezone-aware UTC datetime for comparison
            if hasattr(target_date, 'tzinfo') and target_date.tzinfo is not None:
                target_dt = pd.to_datetime(target_date)
            else:
                # target_date is likely a date object or naive datetime
                target_dt = pd.to_datetime(target_date).tz_localize('UTC')
            
            date_str = target_date.strftime('%Y-%m-%d')
            tle_df['date_str'] = tle_df['date'].dt.strftime('%Y-%m-%d')
            match = tle_df[tle_df['date_str'] == date_str]
            if len(match) > 0:
                return match, None
            else:
                # Return closest date before target
                earlier = tle_df[tle_df['date'] <= target_dt]
                if len(earlier) > 0:
                    return earlier.tail(1), f"Using closest available date: {earlier.iloc[-1]['date'].strftime('%Y-%m-%d')}"
        
        return tle_df, None
        
    except Exception as e:
        return None, f"Space-Track API error: {str(e)}"


# =============================================================================
# DATA & MODEL CACHING
# =============================================================================
# =============================================================================
# DATA & MODEL CACHING
# =============================================================================
@st.cache_data
def load_training_data():
    """Load and cache base training data from CSV."""
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    return df

def load_custom_satellites_from_db(base_df):
    """Load cached custom satellites from database and merge with base dataframe."""
    if not SPACETRACK_AVAILABLE:
        return base_df
        
    try:
        ingestor = get_space_track_ingestor()
        if not ingestor:
            return base_df
            
        cache_status = ingestor.get_cache_status()
        if cache_status.empty:
            return base_df
            
        # Get list of NORAD IDs in cache but not in CSV
        csv_ids = set(base_df['norad_id'].unique())
        cached_ids = set(cache_status['norad_id'].unique())
        new_ids = cached_ids - csv_ids
        
        if new_ids:
            new_records = []
            for nid in new_ids:
                # Get satellite name from cache_status (more reliable)
                sat_info = cache_status[cache_status['norad_id'] == nid].iloc[0]
                sat_name = sat_info.get('satellite_name', f"Custom Sat {nid}")
                
                # If satellite_name is empty or null, use default
                if pd.isna(sat_name) or sat_name == '':
                    sat_name = f"Custom Sat {nid}"
                
                # Create a minimal record for the dropdown
                new_records.append({
                    'norad_id': nid,
                    'satellite_name': sat_name,
                    'date': pd.Timestamp.now(tz='UTC')  # Placeholder date
                })
            
            if new_records:
                combined_new = pd.DataFrame(new_records)
                # Merge with base dataframe
                merged_df = pd.concat([base_df, combined_new], ignore_index=True)
                # Notify user (unobtrusive)
                sat_names = [r['satellite_name'] for r in new_records]
                st.toast(f"✅ Loaded {len(new_ids)} custom satellites: {', '.join(sat_names[:3])}", icon="💾")
                return merged_df
                
    except Exception as e:
        print(f"Warning: Failed to load cached satellites: {e}")
        
    return base_df


@st.cache_resource
def train_model(df):
    """Train and cache the XGBoost model."""
    df_clean = df.dropna(subset=FEATURE_COLUMNS + ['decay_rate_m'])
    df_clean = df_clean[df_clean['decay_rate_m'] > 0.01]
    
    X = df_clean[FEATURE_COLUMNS]
    y = df_clean['decay_rate_m']
    
    model = XGBRegressor(**MODEL_PARAMS)
    model.fit(X, y)
    
    return model


def get_satellite_options(df):
    """Get unique satellites from training data."""
    satellites = df.groupby('norad_id').agg({
        'satellite_name': 'first'
    }).reset_index()
    satellites['display_name'] = satellites.apply(
        lambda x: f"{x['satellite_name']} (ID: {x['norad_id']})", axis=1
    )
    return satellites


# =============================================================================
# PREDICTION ENGINE
# =============================================================================
def calculate_features(date, sat_data, kp_mean, kp_max, f107, is_backtest=False, df=None, norad_id=None):
    """
    Calculate all required features for prediction.
    
    In BACKTEST MODE: Uses historical feature values from the training data for that specific date.
    In FORECAST MODE: Uses user-provided space weather values and latest satellite parameters.
    """
    # Day of year encoding (always from the target date)
    doy = date.timetuple().tm_yday
    sin_doy = np.sin(2 * np.pi * doy / 365.25)
    cos_doy = np.cos(2 * np.pi * doy / 365.25)
    
    # Try to get historical data for exact date (BACKTEST MODE)
    historical_data = None
    if is_backtest and df is not None and norad_id is not None:
        date_str = date.strftime('%Y-%m-%d')
        match = df[(df['norad_id'] == norad_id) & (df['date'].dt.strftime('%Y-%m-%d') == date_str)]
        if len(match) > 0:
            historical_data = match.iloc[0]
    
    if historical_data is not None:
        # USE HISTORICAL VALUES - This is proper backtesting
        features = {
            'sun_exposure_factor': historical_data['sun_exposure_factor'],
            'beta_angle_deg': historical_data['beta_angle_deg'],
            'f107_obs': historical_data['f107_obs'],
            'f107_81d_avg': historical_data['f107_81d_avg'],
            'Kp_mean': historical_data['Kp_mean'],
            'Kp_max': historical_data['Kp_max'],
            'Kp_Lag_24h': historical_data['Kp_Lag_24h'],
            'semi_major_axis_km': historical_data['semi_major_axis_km'],
            'altitude_km': historical_data['altitude_km'],
            'perigee_alt_km': historical_data['perigee_alt_km'],
            'eccentricity': historical_data['eccentricity'],
            'sin_doy': sin_doy,
            'cos_doy': cos_doy,
            'static_bc_est': historical_data['static_bc_est']
        }
    else:
        # FORECAST MODE or historical data not found - Use user inputs + latest orbital params
        latest = sat_data.iloc[-1]
        beta_angle = sat_data['beta_angle_deg'].mean() if 'beta_angle_deg' in sat_data.columns else 30.0
        sun_exposure = sat_data['sun_exposure_factor'].mean() if 'sun_exposure_factor' in sat_data.columns else 0.8
        
        # Get orbital parameters with fallbacks
        semi_major = latest.get('semi_major_axis_km', 6778.0)
        altitude = latest.get('altitude_km', 400.0)
        perigee = latest.get('perigee_alt_km', 395.0)
        ecc = latest.get('eccentricity', 0.0005)
        bc_est = latest.get('static_bc_est', 1.0e7)
        
        # DEBUG: Print orbital parameters being used
        print(f"[DEBUG] calculate_features for custom satellite:")
        print(f"  semi_major_axis_km: {semi_major}")
        print(f"  altitude_km: {altitude}")
        print(f"  perigee_alt_km: {perigee}")
        print(f"  eccentricity: {ecc}")
        
        features = {
            'sun_exposure_factor': sun_exposure,
            'beta_angle_deg': beta_angle,
            'f107_obs': f107,
            'f107_81d_avg': f107,  # Approximation
            'Kp_mean': kp_mean,
            'Kp_max': kp_max,
            'Kp_Lag_24h': kp_mean,  # Approximation
            'semi_major_axis_km': semi_major,
            'altitude_km': altitude,
            'perigee_alt_km': perigee,
            'eccentricity': ecc,
            'sin_doy': sin_doy,
            'cos_doy': cos_doy,
            'static_bc_est': bc_est
        }
    
    return pd.DataFrame([features])


def get_baseline_decay(sat_data):
    """Calculate baseline decay rate from historical data."""
    if len(sat_data) < 5:
        return 10.0  # Default
    return sat_data['decay_rate_m'].rolling(30, min_periods=5).median().iloc[-1]


def get_actual_decay(df, norad_id, date):
    """Get actual decay rate for a specific date (for backtesting)."""
    date_str = date.strftime('%Y-%m-%d')
    match = df[(df['norad_id'] == norad_id) & (df['date'].dt.strftime('%Y-%m-%d') == date_str)]
    if len(match) > 0:
        return match['decay_rate_m'].values[0]
    return None


def determine_risk_level(predicted, baseline):
    """Determine risk level based on prediction vs baseline."""
    if baseline <= 0:
        return 'green', 100
    
    ratio = (predicted / baseline) * 100
    
    if ratio > 250:
        return 'red', ratio
    elif ratio > 150:
        return 'yellow', ratio
    else:
        return 'green', ratio


# =============================================================================
# VISUALIZATION COMPONENTS
# =============================================================================
def create_risk_gauge(ratio, risk_level):
    """Create a semicircular risk gauge."""
    colors = {'green': '#00ff88', 'yellow': '#ffd700', 'red': '#ff4444'}
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=min(ratio, 300),  # Cap at 300% for display
        number={'suffix': '%', 'font': {'size': 48, 'color': colors[risk_level], 'family': 'Orbitron'}},
        gauge={
            'axis': {'range': [0, 300], 'tickcolor': '#8b8ba7', 'tickfont': {'color': '#8b8ba7', 'family': 'Space Grotesk'}},
            'bar': {'color': colors[risk_level]},
            'bgcolor': 'rgba(30,30,60,0.6)',
            'bordercolor': 'rgba(0,212,255,0.3)',
            'steps': [
                {'range': [0, 100], 'color': 'rgba(0,255,136,0.2)'},
                {'range': [100, 150], 'color': 'rgba(0,255,136,0.1)'},
                {'range': [150, 250], 'color': 'rgba(255,215,0,0.2)'},
                {'range': [250, 300], 'color': 'rgba(255,68,68,0.2)'}
            ],
            'threshold': {
                'line': {'color': '#00d4ff', 'width': 3},
                'thickness': 0.8,
                'value': 100
            }
        },
        title={'text': 'DRAG RISK INDEX', 'font': {'size': 18, 'color': '#00d4ff', 'family': 'Orbitron'}}
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=280,
        margin=dict(l=30, r=30, t=60, b=30)
    )
    
    return fig


def create_comparison_chart(predicted, baseline, actual=None):
    """Create a bar chart comparing predicted vs baseline (and actual if available)."""
    categories = ['Baseline', 'Predicted']
    values = [baseline, predicted]
    colors = ['#00d4ff', '#00ff88']
    
    if actual is not None:
        categories.append('Actual')
        values.append(actual)
        colors.append('#ffd700')
    
    # Calculate max value for proper scaling
    max_val = max(values) * 1.3  # 30% headroom for text
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f'{v:.2f}' for v in values],
            textposition='inside',
            textfont={'family': 'Orbitron', 'size': 14, 'color': '#ffffff'},
            insidetextanchor='middle'
        )
    ])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=320,
        yaxis_title='Decay Rate (m/day)',
        yaxis={'color': '#8b8ba7', 'gridcolor': 'rgba(0,212,255,0.1)', 'range': [0, max_val]},
        xaxis={'color': '#8b8ba7'},
        font={'family': 'Space Grotesk', 'color': '#8b8ba7'},
        margin=dict(l=60, r=30, t=40, b=60),
        bargap=0.3
    )
    
    return fig


def show_processing_animation():
    """Display animated processing sequence."""
    messages = [
        "🛰️ Initializing orbital parameters...",
        "📡 Calculating solar beta angle...",
        "☀️ Processing space weather data...",
        "🧠 Running ensemble prediction model...",
        "📊 Generating risk assessment..."
    ]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, msg in enumerate(messages):
        status_text.markdown(f'<p class="processing-text">{msg}</p>', unsafe_allow_html=True)
        progress_bar.progress((i + 1) / len(messages))
        time.sleep(0.6)
    
    status_text.empty()
    progress_bar.empty()


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0 30px 0;">
        <h1 style="font-size: 2.8rem; margin: 0; background: linear-gradient(135deg, #00d4ff, #00ff88); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   text-shadow: none;">🛰️ ORBITAL DECAY PREDICTOR</h1>
        <p style="font-size: 1.1rem; margin-top: 10px; letter-spacing: 3px;">
            SATELLITE DRAG EARLY WARNING SYSTEM
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    # Load data
    try:
        base_df = load_training_data()
        model = train_model(base_df)
        
        # Load and merge custom satellites (dynamically)
        df = load_custom_satellites_from_db(base_df)
        
        satellites = get_satellite_options(df)
    except FileNotFoundError:
        st.error("❌ Training data not found. Please ensure `training_set_geometric.csv` exists.")
        return
    
    # Get date range from data
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    # =============================================================================
    # SIDEBAR
    # =============================================================================
    
    # Initialize session state for date selection and features
    if 'date_selection' not in st.session_state:
        st.session_state['date_selection'] = 'tomorrow'
        st.session_state['selected_date'] = datetime.now().date() + timedelta(days=1)
    
    if 'last_date' not in st.session_state:
        st.session_state['last_date'] = st.session_state['selected_date']
    
    # Default feature values (will be auto-set based on date)
    DEFAULT_KP_MEAN = 3.0
    DEFAULT_KP_MAX = 4.0
    DEFAULT_F107 = 150.0
    
    if 'features_modified' not in st.session_state:
        st.session_state['features_modified'] = False
        st.session_state['kp_mean'] = DEFAULT_KP_MEAN
        st.session_state['kp_max'] = DEFAULT_KP_MAX
        st.session_state['f107'] = DEFAULT_F107
    
    with st.sidebar:
        st.markdown("""
        <h2 style="font-size: 1.2rem; margin-bottom: 16px; font-family: Orbitron, sans-serif; 
                   color: #00d4ff;">⚙️ MISSION CONTROL</h2>
        """, unsafe_allow_html=True)
        
        # Satellite Selection
        st.markdown("**🛰️ Satellite**")
        input_mode = st.radio(
            "Input method:",
            ["Database", "Custom"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if input_mode == "Database":
            selected_display = st.selectbox(
                "Satellite:",
                satellites['display_name'].tolist(),
                index=satellites[satellites['norad_id'] == 25544].index.tolist()[0] if 25544 in satellites['norad_id'].values else 0,
                label_visibility="collapsed"
            )
            selected_norad = satellites[satellites['display_name'] == selected_display]['norad_id'].values[0]
            selected_name = satellites[satellites['display_name'] == selected_display]['satellite_name'].values[0]
        else:
            selected_norad = st.number_input("NORAD ID:", min_value=1, value=25544, step=1)
            selected_name = st.text_input("Name:", value="Custom Satellite")
        
        st.divider()
        
        # Date Selection - Simplified with radio for reliability
        st.markdown("**📅 Date**")
        
        # Quick date options as radio buttons (more reliable than individual buttons)
        date_options = ["Tomorrow", "Next Week", "Custom Date"]
        
        # Initialize date choice if not present
        if 'date_choice' not in st.session_state:
            st.session_state['date_choice'] = "Tomorrow"
        
        date_choice = st.radio(
            "Quick select:",
            date_options,
            index=date_options.index(st.session_state.get('date_choice', 'Tomorrow')),
            horizontal=True,
            label_visibility="collapsed"
        )
        
        # Update session state when choice changes
        if date_choice != st.session_state.get('date_choice'):
            st.session_state['date_choice'] = date_choice
            st.session_state['features_modified'] = False  # Reset features on date change
        
        # Calculate the actual date based on selection
        if date_choice == "Tomorrow":
            selected_date = datetime.now().date() + timedelta(days=1)
        elif date_choice == "Next Week":
            selected_date = datetime.now().date() + timedelta(days=7)
        else:
            # Custom date picker
            col1, col2 = st.columns(2)
            with col1:
                quick_year = st.selectbox("Year", list(range(2015, 2028)), index=9, label_visibility="collapsed")
            with col2:
                quick_month = st.selectbox("Month", list(range(1, 13)), index=0, label_visibility="collapsed")
            
            # Show date picker
            custom_default = datetime(quick_year, quick_month, 15).date()
            selected_date = st.date_input(
                "Pick date:",
                value=custom_default,
                label_visibility="collapsed"
            )
        
        # Show selected date clearly
        st.markdown(f"""
        <div style="background: rgba(0,212,255,0.1); border-radius: 6px; padding: 8px 12px; 
                    margin: 8px 0; text-align: center;">
            <span style="color: #00d4ff; font-size: 0.9rem;">📆 {selected_date.strftime('%B %d, %Y')}</span>
        </div>
        """, unsafe_allow_html=True)
        
        is_backtest = selected_date <= max_date
        
        if is_backtest:
            st.success("🔄 Backtest Mode")
        else:
            st.info("🔮 Forecast Mode")
        
        st.divider()
        
        # Space Weather - Auto-reset on date change, but respect user modifications
        st.markdown("**🌡️ Space Weather**")
        
        # Get default values (reset if date changed and user hasn't modified)
        if not st.session_state['features_modified']:
            default_kp_mean = DEFAULT_KP_MEAN
            default_kp_max = DEFAULT_KP_MAX
            default_f107 = DEFAULT_F107
        else:
            default_kp_mean = st.session_state['kp_mean']
            default_kp_max = st.session_state['kp_max']
            default_f107 = st.session_state['f107']
        
        kp_mean = st.slider("Kp Mean", 0.0, 9.0, default_kp_mean, 0.1, key="slider_kp_mean")
        kp_max = st.slider("Kp Max", 0.0, 9.0, default_kp_max, 0.1, key="slider_kp_max")
        f107 = st.slider("F10.7 Flux", 70.0, 300.0, default_f107, 1.0, key="slider_f107")
        
        # Track if user modified the sliders
        if (kp_mean != DEFAULT_KP_MEAN or kp_max != DEFAULT_KP_MAX or f107 != DEFAULT_F107):
            st.session_state['features_modified'] = True
            st.session_state['kp_mean'] = kp_mean
            st.session_state['kp_max'] = kp_max
            st.session_state['f107'] = f107
        
        st.divider()
        
        # Predict Button
        predict_clicked = st.button("🚀 PREDICT DECAY", use_container_width=True, type="primary")
    
    # =============================================================================
    # MAIN CONTENT
    # =============================================================================
    if predict_clicked:
        # Check if satellite exists in database
        sat_data = df[df['norad_id'] == selected_norad]
        
        if sat_data.empty:
            # Custom satellite - try Space-Track API first
            st.info(f"🛰️ Satellite {selected_name} (ID: {selected_norad}) not in training database. Attempting Space-Track lookup...")
            
            # Try to fetch from Space-Track
            spacetrack_data, error_msg = fetch_custom_satellite_data(
                selected_norad, 
                target_date=selected_date if is_backtest else None
            )
            
            if spacetrack_data is not None and len(spacetrack_data) > 0:
                st.success(f"✅ Successfully retrieved data from Space-Track! ({len(spacetrack_data)} records)")
                if error_msg:
                    st.info(error_msg)  # Show any info messages (e.g., "using closest date")
                sat_data = spacetrack_data
                baseline = sat_data['decay_rate_m'].mean() if 'decay_rate_m' in sat_data.columns else 10.0
            else:
                # Fallback to generic parameters
                if error_msg:
                    st.warning(f"⚠️ {error_msg}")
                st.warning("Using generic LEO parameters for prediction.")
                
                # Create synthetic satellite data based on generic LEO values
                generic_data = {
                    'semi_major_axis_km': 6778.0,  # ~400km altitude
                    'altitude_km': 400.0,
                    'perigee_alt_km': 395.0,
                    'eccentricity': 0.0005,
                    'beta_angle_deg': 30.0,
                    'sun_exposure_factor': 0.8,
                    'static_bc_est': 1.0e7,
                    'decay_rate_m': 10.0
                }
                sat_data = pd.DataFrame([generic_data])
                baseline = 10.0
        else:
            baseline = get_baseline_decay(sat_data)
        
        # Show processing animation
        show_processing_animation()
        
        # Calculate features and predict
        start_time = time.time()
        X_pred = calculate_features(selected_date, sat_data, kp_mean, kp_max, f107, 
                                    is_backtest=is_backtest, df=df, norad_id=selected_norad)
        predicted_decay = model.predict(X_pred)[0]
        processing_time = time.time() - start_time
        
        # Determine risk
        risk_level, ratio = determine_risk_level(predicted_decay, baseline)
        
        # Get actual value if backtesting
        actual_decay = None
        if is_backtest and selected_norad in df['norad_id'].values:
            actual_decay = get_actual_decay(df, selected_norad, selected_date)
        
        # =============================================================================
        # RESULTS DISPLAY
        # =============================================================================
        st.markdown(f"""
        <div class="glass-card">
            <h2 style="text-align: center; margin-bottom: 20px;">
                📊 PREDICTION RESULTS FOR {selected_name.upper()}
            </h2>
            <p style="text-align: center; font-size: 1.1rem;">
                Target Date: <span style="color: #00d4ff;">{selected_date.strftime('%B %d, %Y')}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Data Freshness Notice
        st.markdown(f"""
        <div class="data-freshness">
            📌 <strong>Data Notice:</strong> This prediction is generated using the ensemble model trained on historical data 
            up to <strong>{max_date.strftime('%B %Y')}</strong>. Results are most accurate for satellites and conditions 
            similar to the training dataset.
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics Row - 3 Cards Only (Full Width)
        risk_class = f"risk-{risk_level}"
        
        st.markdown(f"""
        <div class="metrics-row">
            <div class="metric-card">
                <div class="metric-value">{predicted_decay:.2f}</div>
                <div class="metric-label">Predicted Decay (m/day)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{baseline:.2f}</div>
                <div class="metric-label">Baseline Decay</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {risk_class}">{ratio:.0f}%</div>
                <div class="metric-label">% of Baseline</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Terminal-style accuracy readout (if backtesting)
        if actual_decay:
            accuracy = (1 - abs(predicted_decay - actual_decay) / actual_decay) * 100
            st.markdown(f"""
            <div style="background: #0d1117; border: 1px solid #30363d; border-radius: 8px; 
                        padding: 16px 20px; margin: 16px 0; font-family: 'Consolas', 'Monaco', monospace;">
                <div style="color: #58a6ff; font-size: 0.8rem; margin-bottom: 8px;">📊 VALIDATION RESULTS</div>
                <div style="color: #8b949e; font-size: 0.85rem; line-height: 1.8;">
                    <span style="color: #7ee787;">▶</span> Actual Decay Rate: <span style="color: #ffd700;">{actual_decay:.2f}</span> m/day<br>
                    <span style="color: #7ee787;">▶</span> Predicted Decay Rate: <span style="color: #00d4ff;">{predicted_decay:.2f}</span> m/day<br>
                    <span style="color: #7ee787;">▶</span> Model Accuracy: <span style="color: #7ee787; font-weight: bold;">{accuracy:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts Row
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_risk_gauge(ratio, risk_level), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_comparison_chart(predicted_decay, baseline, actual_decay), use_container_width=True)
        
        # Recommendation Box
        rec = RECOMMENDATIONS[risk_level]
        box_class = 'recommendation-box' if risk_level == 'green' else f'recommendation-box {risk_level}'
        if risk_level == 'yellow':
            box_class = 'recommendation-box warning'
        elif risk_level == 'red':
            box_class = 'recommendation-box critical'
        
        actions_html = ''.join([f'<li>{action}</li>' for action in rec['actions']])
        
        st.markdown(f"""
        <div class="{box_class}">
            <h3 style="margin-top: 0;">{rec['title']}</h3>
            <p style="font-size: 1.1rem; color: #ffffff !important;">{rec['message']}</p>
            <h4 style="margin-top: 16px;">Recommended Actions:</h4>
            <ul style="color: #ffffff !important; line-height: 1.8;">
                {actions_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Processing Time Footer
        st.markdown(f"""
        <div style="text-align: center; margin-top: 30px; padding: 16px; 
                    border-top: 1px solid rgba(0,212,255,0.2);">
            <p style="font-size: 0.9rem;">
                ⚡ Prediction completed in <span style="color: #00d4ff; font-family: Orbitron;">{processing_time:.3f}s</span>
                &nbsp;|&nbsp; Model: <span style="color: #00ff88;">XGBoost Ensemble (5-Fold)</span>
                &nbsp;|&nbsp; Features: <span style="color: #ffd700;">14 Physics-Based</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Default state - show instructions (RESPONSIVE)
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 40px 20px; max-width: 800px; margin: 0 auto;">
            <h2 style="margin-bottom: 24px; font-size: 1.5rem;">🚀 MISSION BRIEFING</h2>
            <p style="font-size: 1rem; color: #ffffff !important; line-height: 1.8; margin-bottom: 30px;">
                Welcome to the Orbital Decay Prediction System.<br>
                Select a satellite and date from the sidebar to begin analysis.
            </p>
            <div style="display: flex; justify-content: center; gap: 30px; flex-wrap: wrap;">
                <div style="min-width: 100px; padding: 16px;">
                    <p style="font-size: 2.5rem; margin: 0; color: #ffffff; font-family: Orbitron, sans-serif;">9</p>
                    <p style="font-size: 0.8rem; color: #8b8ba7; margin-top: 8px;">Trained Satellites</p>
                </div>
                <div style="min-width: 100px; padding: 16px;">
                    <p style="font-size: 2.5rem; margin: 0; color: #00d4ff; font-family: Orbitron, sans-serif;">33K+</p>
                    <p style="font-size: 0.8rem; color: #8b8ba7; margin-top: 8px;">Training Samples</p>
                </div>
                <div style="min-width: 100px; padding: 16px;">
                    <p style="font-size: 2.5rem; margin: 0; color: #00ff88; font-family: Orbitron, sans-serif;">14</p>
                    <p style="font-size: 0.8rem; color: #8b8ba7; margin-top: 8px;">Physics Features</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

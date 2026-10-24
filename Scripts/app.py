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
import sys

# Ensure centralized path and dependency resolution
try:
    from path_config import get_data_file, get_db_path, get_model_path, PROJECT_ROOT, DATAFILES_DIR, MODELS_DIR
except ImportError:
    _cur_dir = os.path.dirname(os.path.abspath(__file__))
    if _cur_dir not in sys.path:
        sys.path.insert(0, _cur_dir)
    from path_config import get_data_file, get_db_path, get_model_path, PROJECT_ROOT, DATAFILES_DIR, MODELS_DIR
from xgboost import XGBRegressor
import warnings
import sqlite3

# Import Stratified LEO Prediction System
try:
    from stratified_model import (
        StratifiedLEOPredictor,
        is_active_payload,
        get_satellite_tag,
        propagate_future_geometry
    )
    STRATIFIED_AVAILABLE = True
except Exception as e:
    STRATIFIED_AVAILABLE = False
    print(f"Notice: Stratified module fallback: {e}")

# Import Space-Track and Physics Engine
try:
    from etl_pipeline import SpaceTrackIngestor
    SPACETRACK_AVAILABLE = True
except ImportError:
    SPACETRACK_AVAILABLE = False

try:
    from physics_engine import calculate_solar_beta, fetch_and_merge_solar, estimate_ballistic_coefficient
    PHYSICS_ENGINE_AVAILABLE = True
except Exception as e:
    PHYSICS_ENGINE_AVAILABLE = False

# Import 7-Day Maneuver & Attitude Optimizer
try:
    from step7_maneuver_optimizer import BStarManeuverOptimizer
    from plotly.subplots import make_subplots
    OPTIMIZER_AVAILABLE = True
except Exception as e:
    OPTIMIZER_AVAILABLE = False
    print(f"Notice: Maneuver optimizer fallback: {e}")

# Initialize cached predictor so it only loads seed weights once on startup
@st.cache_resource
def load_predictor():
    if STRATIFIED_AVAILABLE:
        try:
            p = StratifiedLEOPredictor()
            p.load_models()
            p.load_seed_ensembles()  # Loads the 5-seed ensembles for deep & upper regimes
            return p
        except Exception as e:
            print(f"Notice: Failed to load StratifiedLEOPredictor: {e}")
            return None
    return None

predictor = load_predictor()

warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="B-Star | Orbital decay early warning",
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
        max-width: 1200px !important;
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
    
    /* Typography & Header Collision Prevention */
    h1, h2, h3, h4, h5, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        font-family: 'Orbitron', sans-serif !important;
        color: var(--text-primary) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        line-height: 1.4 !important;
        word-break: break-word !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }
    
    h1, .stMarkdown h1 { font-size: clamp(1.8rem, 3.5vw, 2.2rem) !important; }
    h2, .stMarkdown h2 { font-size: clamp(1.3rem, 2.5vw, 1.55rem) !important; }
    h3, .stMarkdown h3 { font-size: clamp(1.05rem, 2vw, 1.2rem) !important; }
    h4, .stMarkdown h4 { font-size: clamp(0.9rem, 1.5vw, 1.0rem) !important; }

    p, .stMarkdown p {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text-secondary) !important;
        line-height: 1.55 !important;
        margin-bottom: 0.5rem;
    }

    /* Widget Labels - Prevent overlapping input boxes without breaking flex containers */
    [data-testid="stWidgetLabel"] > label,
    [data-testid="stWidgetLabel"] p {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text-secondary) !important;
        line-height: 1.35 !important;
        margin-bottom: 6px !important;
        display: block !important;
        word-break: break-word !important;
    }
    
    /* Toggle switch layout - Proper flexbox alignment */
    [data-testid="stToggle"] {
        margin-top: 6px !important;
        margin-bottom: 12px !important;
    }
    [data-testid="stToggle"] label {
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
        cursor: pointer !important;
    }
    [data-testid="stToggle"] label p,
    [data-testid="stToggle"] label span {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.88rem !important;
        line-height: 1.35 !important;
        color: var(--text-primary) !important;
        margin: 0 !important;
        word-break: normal !important;
    }
    
    /* Form controls vertical margins to avoid overlap */
    [data-testid="stNumberInput"], [data-testid="stSelectbox"], [data-testid="stSlider"] {
        margin-bottom: 14px !important;
    }
    
    /* Sidebar styling - Fixed width and hamburger icon */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10,10,26,0.95) 0%, rgba(20,20,50,0.95) 100%);
        border-right: 1px solid rgba(0, 212, 255, 0.2);
        min-width: 310px !important;
        max-width: 310px !important;
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

    /* Metric cards - Uniform heights with flexible wrapping */
    .metrics-row {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        margin: 20px 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(30, 30, 60, 0.6) 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        flex: 1 1 200px;
        min-height: 135px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-sizing: border-box;
    }
    
    .metric-value {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--accent-cyan) !important;
        text-shadow: var(--glow-cyan);
        line-height: 1.25;
        word-break: break-word;
    }
    
    .metric-label {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.78rem;
        color: var(--text-secondary) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 8px;
        margin-bottom: 2px;
        line-height: 1.35;
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
    
    /* Expander - Complete fix for overlapping text and high-tech styling */
    [data-testid="stExpander"] {
        background: rgba(30, 30, 60, 0.6) !important;
        border: 1px solid rgba(0, 212, 255, 0.25) !important;
        border-radius: 10px !important;
        margin: 16px 0 !important;
        overflow: visible !important;
    }
    
    [data-testid="stExpander"] summary {
        padding: 14px 18px !important;
        display: flex !important;
        align-items: center !important;
        cursor: pointer !important;
    }
    
    [data-testid="stExpander"] summary p {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: var(--accent-cyan) !important;
        letter-spacing: 0.5px !important;
        margin: 0 !important;
        line-height: 1.4 !important;
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
    
    /* Top Persistent Telemetry HUD Bar */
    .hud-bar {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
        background: linear-gradient(90deg, rgba(20, 24, 52, 0.85), rgba(30, 30, 60, 0.65));
        border: 1px solid rgba(0, 212, 255, 0.35);
        border-radius: 12px;
        padding: 14px 22px;
        margin-bottom: 20px;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.25);
    }
    
    .hud-item {
        display: flex;
        flex-direction: column;
    }
    
    .hud-val {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .hud-lbl {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: var(--text-secondary);
        margin-bottom: 3px;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-family: 'Orbitron', sans-serif !important;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    .badge-deep { background: rgba(255, 68, 68, 0.18); color: #ff6b6b; border: 1px solid #ff4444; }
    .badge-upper { background: rgba(0, 212, 255, 0.15); color: #00d4ff; border: 1px solid #00d4ff; }
    .badge-backtest { background: rgba(255, 215, 0, 0.15); color: #ffd700; border: 1px solid #ffd700; }
    .badge-forecast { background: rgba(0, 255, 136, 0.15); color: #00ff88; border: 1px solid #00ff88; }
    
    .metric-sub {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.76rem;
        margin-top: 6px;
        font-weight: 600;
    }
    
    .glow-cyan { color: var(--accent-cyan) !important; text-shadow: 0 0 15px rgba(0, 212, 255, 0.45); }
    .glow-green { color: var(--accent-green) !important; text-shadow: 0 0 15px rgba(0, 255, 136, 0.35); }
    .glow-yellow { color: var(--accent-yellow) !important; text-shadow: 0 0 15px rgba(255, 215, 0, 0.35); }
    .glow-red { color: var(--accent-red) !important; text-shadow: 0 0 15px rgba(255, 68, 68, 0.35); }
    
    /* Tabs Navigation Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(20, 20, 45, 0.6);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid rgba(0, 212, 255, 0.2);
        margin-bottom: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        color: var(--text-secondary) !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        transition: all 0.2s ease !important;
        border: none !important;
        background: transparent !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--accent-cyan) !important;
        background: rgba(0, 212, 255, 0.1) !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #00ff88 !important;
        background: rgba(0, 255, 136, 0.15) !important;
        border-bottom: 2px solid #00ff88 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- v2 additions: new classes only; every existing class above is untouched ---
st.markdown("""
<style>
    .brand-bar { display:flex; align-items:baseline; gap:14px; flex-wrap:wrap;
                 padding:2px 0 12px 0; margin-bottom:8px; border-bottom:1px solid rgba(0,212,255,0.15); }
    .brand-name { font-family:'Orbitron',sans-serif; font-weight:800; font-size:clamp(1.3rem,2.6vw,1.8rem);
                  color:#00d4ff; letter-spacing:0.04em; }
    .brand-sub { color:#8b8ba7; font-size:0.95rem; }
    .step-h { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:0.98rem; color:#e8e8f5; margin:2px 0 6px 0; }
    .verdict { background:rgba(30,30,60,0.6); border:1px solid rgba(255,255,255,0.09);
               border-left:6px solid var(--vt,#00d4ff); border-radius:10px; padding:16px 22px; margin:4px 0 16px 0; }
    .verdict-title { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.4rem; line-height:1.3; color:#ffffff; margin:6px 0 6px 0; }
    .verdict p { margin:4px 0; color:#d6d6e6; line-height:1.55; font-size:0.96rem; }
    .verdict-badge { display:inline-block; padding:2px 12px; border-radius:999px; font-size:0.82rem;
                     border:1px solid var(--vt,#00d4ff); color:var(--vt,#00d4ff); }
    @media (prefers-reduced-motion: reduce) { * { animation:none !important; transition:none !important; } }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================
DATA_PATH = get_data_file("2014-2025.csv")
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

        # Physical LEO check: Atmospheric drag models are only valid for LEO (<= 1000 km)
        latest_alt = tle_df['altitude_km'].iloc[-1] if 'altitude_km' in tle_df.columns else 400.0
        if latest_alt > 1000.0:
            return None, f"Satellite altitude (~{latest_alt:.0f} km) exceeds Low Earth Orbit limit (1,000 km). Atmospheric drag models (Kp, F10.7) are physically invalid outside LEO."
        
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
    """Load serialized Stratified LEO Predictor (Deep LEO + Upper LEO + Calibration)."""
    p = load_predictor()
    if p is not None:
        return p

    candidate_model_files = [
        get_model_path("Model_1_Single_Global_XGBoost", "xgb_single_global.json"),
        os.path.join("models", "xgb_leo_model.json")
    ]
    for model_file in candidate_model_files:
        if os.path.exists(model_file):
            try:
                model = XGBRegressor()
                model.load_model(model_file)
                return model
            except Exception as e:
                print(f"Warning: Failed to load pre-trained model ({e}), falling back to training.")

    if 'altitude_km' in df.columns:
        df = df[df['altitude_km'] <= 1000.0]
    df_clean = df.dropna(subset=FEATURE_COLUMNS + ['decay_rate_m'])
    df_clean = df_clean[df_clean['decay_rate_m'] > 0.01]
    
    X = df_clean[FEATURE_COLUMNS]
    y = df_clean['decay_rate_m']
    
    model = XGBRegressor(**MODEL_PARAMS)
    model.fit(X, y)
    
    return model


def get_satellite_options(df):
    """Get unique satellites from training data (LEO only, <= 1000 km) tagged by regime."""
    if 'altitude_km' in df.columns:
        df = df[df['altitude_km'] <= 1000.0]
    satellites = df.groupby('norad_id').agg({
        'satellite_name': 'first',
        'altitude_km': 'mean'
    }).reset_index()
    
    if STRATIFIED_AVAILABLE:
        satellites['tag'] = satellites.apply(
            lambda x: get_satellite_tag(x['norad_id'], x['altitude_km']), axis=1
        )
        satellites['display_name'] = satellites.apply(
            lambda x: f"{x['satellite_name']} (ID: {x['norad_id']}) — {x['tag']}", axis=1
        )
    else:
        satellites['display_name'] = satellites.apply(
            lambda x: f"{x['satellite_name']} (ID: {x['norad_id']})", axis=1
        )
    return satellites


# =============================================================================
# SPACE WEATHER DATABASE BASELINE RETRIEVAL
# =============================================================================
def get_space_weather_for_date(target_date, db_path=DB_PATH):
    """
    Retrieve space weather from database for target_date.
    If target_date is in the future or not found, return latest available 24h baseline.
    """
    try:
        conn = sqlite3.connect(db_path)
        date_str = target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else str(target_date)
        
        # Check if exact date exists in solar flux
        sf_row = conn.cursor().execute(
            "SELECT f107_obs FROM solar_flux WHERE date(date_tag) = date(?)", (date_str,)
        ).fetchone()
        
        # Check if exact date exists in space_weather
        kp_rows = conn.cursor().execute(
            "SELECT kp FROM space_weather WHERE date(time_tag) = date(?)", (date_str,)
        ).fetchall()
        
        if sf_row and kp_rows and len(kp_rows) > 0:
            kps = [r[0] for r in kp_rows]
            kp_mean = float(sum(kps) / len(kps))
            kp_max = float(max(kps))
            f107 = float(sf_row[0])
            conn.close()
            return {
                'kp_mean': round(kp_mean, 2),
                'kp_max': round(kp_max, 2),
                'f107': round(f107, 1),
                'source': f"Archived DB ({date_str})",
                'is_historical': True
            }
        
        # Fallback / Future date: Query latest 24h baseline
        latest_sf = conn.cursor().execute(
            "SELECT date_tag, f107_obs FROM solar_flux ORDER BY date(date_tag) DESC LIMIT 1"
        ).fetchone()
        
        latest_kp = conn.cursor().execute(
            "SELECT time_tag, kp FROM space_weather ORDER BY time_tag DESC LIMIT 8"
        ).fetchall()
        conn.close()
        
        if latest_sf and latest_kp:
            kps = [r[1] for r in latest_kp]
            kp_mean = float(sum(kps) / len(kps))
            kp_max = float(max(kps))
            f107 = float(latest_sf[1])
            latest_date = latest_sf[0]
            return {
                'kp_mean': round(kp_mean, 2),
                'kp_max': round(kp_max, 2),
                'f107': round(f107, 1),
                'source': f"Latest DB Baseline ({latest_date})",
                'is_historical': False
            }
    except Exception as e:
        print(f"Notice: Failed to fetch space weather baseline: {e}")
        
    return {
        'kp_mean': 3.0,
        'kp_max': 4.0,
        'f107': 150.0,
        'source': "Default Static Fallback",
        'is_historical': False
    }


# =============================================================================
# PREDICTION ENGINE
# =============================================================================
def calculate_features(date, sat_data, kp_mean, kp_max, f107, is_backtest=False, df=None, norad_id=None, is_overridden=False):
    """
    Calculate all required features for prediction.
    
    In BACKTEST MODE: Strictly causal. Quarantines target date T and retrieves observation strictly
                      prior to T (at least T-1). Uses prior orbital elements and space weather lags.
                      If user manually overrides space weather sliders, uses the overridden values.
    In FORECAST MODE: Uses user-provided space weather values and latest satellite parameters with SGP4.
    """
    # Day of year encoding (always from the target date)
    doy = date.timetuple().tm_yday
    sin_doy = np.sin(2 * np.pi * doy / 365.25)
    cos_doy = np.cos(2 * np.pi * doy / 365.25)
    # Try to get strictly causal prior data (BACKTEST MODE: strictly < target date)
    prior_record = None
    prior_date_used = None
    days_lag = 1
    if is_backtest and df is not None and norad_id is not None:
        target_d = date.date() if hasattr(date, 'date') else date
        # Quarantine day T: retrieve observations strictly BEFORE target date (at least 1 day prior)
        prior_matches = df[(df['norad_id'] == norad_id) & (df['date'].dt.date < target_d)].sort_values('date')
        if len(prior_matches) > 0:
            prior_record = prior_matches.iloc[-1]
            prior_date_used = prior_record['date'].date()
            days_lag = max(1, (target_d - prior_date_used).days)
    
    if prior_record is not None:
        # STRICT CAUSAL BACKTESTING: Features from observation prior to target date (<= T-1)
        # If user explicitly overrode sliders, honor their manual simulation parameters
        use_manual = is_overridden
        kp_min_val = max(0.0, float(kp_mean) - 1.0) if use_manual else prior_record['Kp_min']
        kp_std_val = max(0.1, (float(kp_max) - float(kp_mean)) / 2.0) if use_manual else prior_record['Kp_std']
        
        features = {
            'sun_exposure_factor': prior_record['sun_exposure_factor'],
            'beta_angle_deg': prior_record['beta_angle_deg'],
            'f107_obs': f107 if use_manual else prior_record['f107_obs'],
            'f107_81d_avg': f107 if use_manual else prior_record['f107_81d_avg'],
            'Kp_mean': kp_mean if use_manual else prior_record['Kp_mean'],
            'Kp_max': kp_max if use_manual else prior_record['Kp_max'],
            'Kp_min': kp_min_val,
            'Kp_std': kp_std_val,
            'Kp_Lag_3h': kp_mean if use_manual else prior_record['Kp_Lag_3h'],
            'Kp_Lag_6h': kp_mean if use_manual else prior_record['Kp_Lag_6h'],
            'Kp_Lag_12h': kp_mean if use_manual else prior_record['Kp_Lag_12h'],
            'Kp_Lag_24h': kp_mean if use_manual else prior_record['Kp_Lag_24h'],
            'Kp_Roll_24h_Mean': kp_mean if use_manual else prior_record['Kp_Roll_24h_Mean'],
            'Kp_Roll_24h_Std': kp_std_val if use_manual else prior_record['Kp_Roll_24h_Std'],
            'Kp_Roll_72h_Mean': kp_mean if use_manual else prior_record['Kp_Roll_72h_Mean'],
            'semi_major_axis_km': prior_record['semi_major_axis_km'],
            'altitude_km': prior_record['altitude_km'],
            'perigee_alt_km': prior_record['perigee_alt_km'],
            'eccentricity': prior_record['eccentricity'],
            'inclination_deg': prior_record.get('inclination_deg', 51.6),
            'sin_doy': sin_doy,
            'cos_doy': cos_doy,
            'static_bc_est': prior_record['static_bc_est']
        }
        res_df = pd.DataFrame([features])
        res_df.attrs['prior_date'] = prior_date_used
        res_df.attrs['days_lag'] = days_lag
        res_df.attrs['is_causal'] = True
        return res_df
    else:
        # FORECAST MODE: Dynamically propagate orbit via SGP4 if TLE is available
        latest = sat_data.iloc[-1]
        bc_est = latest.get('static_bc_est', 1.0e7)
        
        # Default baseline values from latest data point
        semi_major = latest.get('semi_major_axis_km', 6778.0)
        altitude = latest.get('altitude_km', 400.0)
        perigee = latest.get('perigee_alt_km', 395.0)
        ecc = latest.get('eccentricity', 0.0005)
        inclination = latest.get('inclination_deg', 51.6)
        beta_angle = sat_data['beta_angle_deg'].mean() if 'beta_angle_deg' in sat_data.columns else 30.0
        sun_exposure = sat_data['sun_exposure_factor'].mean() if 'sun_exposure_factor' in sat_data.columns else 0.8
        
        # Dynamic SGP4 propagation to target forecast date
        if STRATIFIED_AVAILABLE and norad_id is not None:
            try:
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("""
                    SELECT tle_line1, tle_line2 FROM tle_history 
                    WHERE norad_id = ? ORDER BY epoch_datetime DESC LIMIT 1
                """, (int(norad_id),))
                tle_row = c.fetchone()
                conn.close()
                
                if tle_row and tle_row[0] and tle_row[1]:
                    target_dt = datetime.combine(date, datetime.min.time())
                    dyn_geom = propagate_future_geometry(tle_row[0], tle_row[1], target_dt)
                    altitude = dyn_geom['altitude_km']
                    semi_major = dyn_geom['semi_major_axis_km']
                    perigee = dyn_geom['perigee_alt_km']
                    ecc = dyn_geom['eccentricity']
                    inclination = dyn_geom['inclination_deg']
                    beta_angle = dyn_geom['beta_angle_deg']
                    sun_exposure = dyn_geom['sun_exposure_factor']
            except Exception as e:
                print(f"Notice: SGP4 dynamic forecast fallback used: {e}")
        
        # Space weather derivations
        kp_min = max(0.0, float(kp_mean) - 1.0)
        kp_std = max(0.1, (float(kp_max) - float(kp_mean)) / 2.0)
        
        features = {
            'sun_exposure_factor': sun_exposure,
            'beta_angle_deg': beta_angle,
            'f107_obs': f107,
            'f107_81d_avg': f107,  # Approximation
            'Kp_mean': kp_mean,
            'Kp_max': kp_max,
            'Kp_min': kp_min,
            'Kp_std': kp_std,
            'Kp_Lag_3h': kp_mean,
            'Kp_Lag_6h': kp_mean,
            'Kp_Lag_12h': kp_mean,
            'Kp_Lag_24h': kp_mean,
            'Kp_Roll_24h_Mean': kp_mean,
            'Kp_Roll_24h_Std': kp_std,
            'Kp_Roll_72h_Mean': kp_mean,
            'semi_major_axis_km': semi_major,
            'altitude_km': altitude,
            'perigee_alt_km': perigee,
            'eccentricity': ecc,
            'inclination_deg': inclination,
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

def get_backtest_ground_truth(df, norad_id, target_date):
    """
    Retrieve ground truth actuals for target_date and previous observation.
    Returns dictionary with actual decay, maneuver flag, altitude changes, and space weather.
    """
    if df is None or norad_id is None:
        return {'has_ground_truth': False}
        
    target_d = target_date.date() if hasattr(target_date, 'date') else target_date
    gt_match = df[(df['norad_id'] == norad_id) & (df['date'].dt.date == target_d)]
    prior_matches = df[(df['norad_id'] == norad_id) & (df['date'].dt.date < target_d)].sort_values('date')
    
    gt_row = gt_match.iloc[0] if len(gt_match) > 0 else None
    prior_row = prior_matches.iloc[-1] if len(prior_matches) > 0 else None
    
    if gt_row is None:
        return {'has_ground_truth': False}
        
    actual_decay = float(gt_row['decay_rate_m'])
    is_maneuver = bool(gt_row.get('is_maneuver', False)) or (actual_decay <= 0) or (actual_decay > 2000.0)
    decay_happened = actual_decay > 0.01
    
    prior_date = prior_row['date'].date() if prior_row is not None else None
    prior_alt = float(prior_row['altitude_km']) if prior_row is not None else None
    actual_alt = float(gt_row['altitude_km'])
    alt_delta_m = (actual_alt - prior_alt) * 1000.0 if prior_alt is not None else None
    
    return {
        'has_ground_truth': True,
        'target_date': target_d,
        'actual_decay': actual_decay,
        'decay_happened': decay_happened,
        'is_maneuver': is_maneuver,
        'prior_date': prior_date,
        'prior_alt_km': prior_alt,
        'actual_alt_km': actual_alt,
        'alt_delta_m': alt_delta_m,
        'actual_kp_mean': float(gt_row['Kp_mean']),
        'actual_kp_max': float(gt_row['Kp_max']),
        'actual_f107': float(gt_row['f107_obs'])
    }


def get_actual_decay(df, norad_id, date):
    """Get actual decay rate for a specific date (for backtesting)."""
    gt = get_backtest_ground_truth(df, norad_id, date)
    return gt['actual_decay'] if gt.get('has_ground_truth') else None


def determine_risk_level(predicted, baseline):
    """Determine risk level based on prediction vs baseline."""
    if baseline <= 0:
        return 'green', 100
    
    ratio = (predicted / baseline) * 100
    
    if ratio > 200:
        return 'red', ratio
    elif ratio > 120:
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
                {'range': [0, 120], 'color': 'rgba(0,255,136,0.2)'},
                {'range': [120, 200], 'color': 'rgba(255,215,0,0.2)'},
                {'range': [200, 300], 'color': 'rgba(255,68,68,0.2)'}
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

def create_comparison_chart(predicted, baseline, actual=None, is_maneuver=False):
    """Create a bar chart comparing predicted vs baseline (and actual if available)."""
    categories = ['Baseline', 'Predicted']
    values = [baseline, predicted]
    colors = ['#00d4ff', '#00ff88']
    
    if actual is not None:
        label = 'Actual (Maneuver)' if is_maneuver else 'Actual Observed'
        categories.append(label)
        values.append(actual)
        colors.append('#ff4d6d' if is_maneuver or actual <= 0 else '#ffd700')
    
    # Calculate min and max value for proper scaling (handles negative decay during maneuvers)
    min_val = min(0.0, min(values) * 1.3)
    max_val = max(10.0, max(values) * 1.3)    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f'{v:.2f}' for v in values],
            textposition='outside',
            textfont={'family': 'Orbitron', 'size': 13, 'color': '#ffffff'}
        )
    ])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=320,
        yaxis_title='Decay Rate (m/day)',
        yaxis={'color': '#8b8ba7', 'gridcolor': 'rgba(0,212,255,0.1)', 'range': [min_val, max_val]},
        xaxis={'color': '#8b8ba7'},
        font={'family': 'Space Grotesk', 'color': '#8b8ba7'},
        margin=dict(l=60, r=30, t=40, b=60),
        bargap=0.3
    )
    
    return fig

def generate_7day_forecast_features(base_date, sat_data, norad_id, scenario_type, 
                                    custom_kp_mean=3.0, custom_kp_max=4.5, custom_f107=160.0):
    """
    Constructs a 7-day feature DataFrame and space weather arrays for trajectory optimization.
    Scenarios:
      - 'severe_storm': G4/G5 Extreme Space Weather Event (Kp 8-9 on days 3-4)
      - 'moderate_storm': G2 Moderate Event (Kp 5-6 on days 3-4)
      - 'baseline': Extended nominal baseline conditions
    """
    days = 7
    dates = [base_date + timedelta(days=i) for i in range(1, days + 1)]
    days_labels = [f"Day {i+1} ({(base_date + timedelta(days=i+1)).strftime('%b %d')})" for i in range(days)]
    
    # 1. Define Space Weather profiles
    if scenario_type == 'severe_storm':
        kp_mean_arr = np.array([3.0, 5.0, 7.7, 8.3, 5.3, 3.7, 2.3])
        kp_max_arr = np.array([4.0, 6.3, 8.7, 9.0, 6.7, 4.7, 3.0])
        f107_arr = np.array([160.0, 195.0, 245.0, 260.0, 205.0, 170.0, 155.0])
    elif scenario_type == 'moderate_storm':
        kp_mean_arr = np.array([2.3, 3.7, 5.3, 5.7, 4.0, 3.0, 2.0])
        kp_max_arr = np.array([3.3, 4.7, 6.0, 6.3, 4.7, 3.7, 2.7])
        f107_arr = np.array([145.0, 165.0, 185.0, 180.0, 160.0, 150.0, 145.0])
    else:
        kp_mean_arr = np.full(days, float(custom_kp_mean))
        kp_max_arr = np.full(days, float(custom_kp_max))
        f107_arr = np.full(days, float(custom_f107))
        
    # 2. Extract base satellite parameters
    latest = sat_data.iloc[-1] if not sat_data.empty else {}
    base_alt = float(latest.get('altitude_km', 400.0))
    base_sma = float(latest.get('semi_major_axis_km', 6778.0))
    base_perigee = float(latest.get('perigee_alt_km', 395.0))
    base_ecc = float(latest.get('eccentricity', 0.0005))
    base_inc = float(latest.get('inclination_deg', 51.6))
    base_beta = float(latest.get('beta_angle_deg', 30.0))
    base_sef = float(latest.get('sun_exposure_factor', 0.8))
    base_bc = float(latest.get('static_bc_est', 1.0e7))
    
    # Check if TLE exists for dynamic propagation
    tle_tuple = None
    if STRATIFIED_AVAILABLE and norad_id is not None:
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("""
                SELECT tle_line1, tle_line2 FROM tle_history 
                WHERE norad_id = ? ORDER BY epoch_datetime DESC LIMIT 1
            """, (int(norad_id),))
            tle_tuple = c.fetchone()
            conn.close()
        except Exception:
            pass

    rows = []
    sef_list = []
    
    for i in range(days):
        cur_d = dates[i]
        doy = cur_d.timetuple().tm_yday
        sin_doy = np.sin(2 * np.pi * doy / 365.25)
        cos_doy = np.cos(2 * np.pi * doy / 365.25)
        
        alt_i = base_alt
        sma_i = base_sma
        perigee_i = base_perigee
        ecc_i = base_ecc
        inc_i = base_inc
        beta_i = base_beta + i * 0.8
        sef_i = base_sef
        
        if STRATIFIED_AVAILABLE and tle_tuple and tle_tuple[0] and tle_tuple[1]:
            try:
                target_dt = datetime.combine(cur_d, datetime.min.time()) if hasattr(cur_d, 'year') else datetime.combine(cur_d.date(), datetime.min.time())
                dyn = propagate_future_geometry(tle_tuple[0], tle_tuple[1], target_dt)
                alt_i = dyn['altitude_km']
                sma_i = dyn['semi_major_axis_km']
                perigee_i = dyn['perigee_alt_km']
                ecc_i = dyn['eccentricity']
                inc_i = dyn['inclination_deg']
                beta_i = dyn['beta_angle_deg']
                sef_i = dyn['sun_exposure_factor']
            except Exception:
                pass
                
        km = kp_mean_arr[i]
        kx = kp_max_arr[i]
        f107_val = f107_arr[i]
        k_min = max(0.0, km - 1.0)
        k_std = max(0.1, (kx - km) / 2.0)
        
        row = {
            'sun_exposure_factor': sef_i,
            'beta_angle_deg': beta_i,
            'f107_obs': f107_val,
            'f107_81d_avg': f107_val,
            'Kp_mean': km,
            'Kp_max': kx,
            'Kp_min': k_min,
            'Kp_std': k_std,
            'Kp_Lag_3h': km,
            'Kp_Lag_6h': km,
            'Kp_Lag_12h': km,
            'Kp_Lag_24h': km,
            'Kp_Roll_24h_Mean': km,
            'Kp_Roll_24h_Std': k_std,
            'Kp_Roll_72h_Mean': km,
            'semi_major_axis_km': sma_i,
            'altitude_km': alt_i,
            'perigee_alt_km': perigee_i,
            'eccentricity': ecc_i,
            'inclination_deg': inc_i,
            'sin_doy': sin_doy,
            'cos_doy': cos_doy,
            'static_bc_est': base_bc
        }
        rows.append(row)
        sef_list.append(sef_i)
        
    df_7day = pd.DataFrame(rows)
    return df_7day, days_labels, kp_max_arr, np.array(sef_list)


def create_maneuver_comparison_plot(opt_res, days_labels, kp_profile, p90_drag, initial_alt_km):
    """
    Creates a dual-row interactive Plotly chart comparing:
    Row 1: 7-Day Orbital Altitude Trajectory (Uncontrolled vs Naive vs B-Star MPC Optimal)
           with Safety Floor line and hover tooltips.
    Row 2: Control Actuation Schedule (Optimal Burns on left y-axis, Attitude Drag Factor alpha on right y-axis).
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        row_heights=[0.62, 0.38],
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
        subplot_titles=(
            "<b>7-DAY ORBITAL ALTITUDE TRAJECTORY (STRATEGY COMPARISON)</b>",
            "<b>DAILY CONTROL ACTUATION: REBOOST BURNS vs. ATTITUDE FEATHERING</b>"
        )
    )
    
    x_trajectory = ['Day 0 (Initial)'] + [f"Day {i+1}" for i in range(len(days_labels))]
    uncontrolled_y = [initial_alt_km] + list(opt_res['uncontrolled_alt_km'])
    naive_y = [initial_alt_km] + list(opt_res['naive_alt_km'])
    optimal_y = [initial_alt_km] + list(opt_res['optimal_alt_km'])
    floor_y = [opt_res['alt_floor_km']] * len(x_trajectory)
    
    # 1. Uncontrolled Drift (Red Dashed Line)
    fig.add_trace(
        go.Scatter(
            x=x_trajectory,
            y=uncontrolled_y,
            mode='lines+markers',
            name='🔴 Uncontrolled Drift (No Thrust)',
            line=dict(color='#ff4444', width=2.5, dash='dash'),
            marker=dict(size=6, color='#ff4444'),
            hovertemplate='Uncontrolled: <b>%{y:.3f} km</b><extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Naive Station-Keeping (Gray Line)
    fig.add_trace(
        go.Scatter(
            x=x_trajectory,
            y=naive_y,
            mode='lines+markers',
            name='⚪ Naive Daily Reboost (Blind Burning)',
            line=dict(color='#a0aec0', width=2, dash='solid'),
            marker=dict(size=6, color='#a0aec0'),
            hovertemplate='Naive Reboost: <b>%{y:.3f} km</b><extra></extra>'
        ),
        row=1, col=1
    )
    
    # 3. B-Star MPC Optimal Schedule (Vibrant Green Line)
    fig.add_trace(
        go.Scatter(
            x=x_trajectory,
            y=optimal_y,
            mode='lines+markers',
            name='🟢 B-Star MPC Optimal (Smart Feathering + Reboost)',
            line=dict(color='#00ff88', width=3.5, dash='solid'),
            marker=dict(size=8, color='#00ff88', symbol='circle'),
            hovertemplate='B-Star Optimal: <b>%{y:.3f} km</b><extra></extra>'
        ),
        row=1, col=1
    )
    
    # 4. Safety Floor (Yellow Dot)
    fig.add_trace(
        go.Scatter(
            x=x_trajectory,
            y=floor_y,
            mode='lines',
            name='⚠️ Operational Safety Floor',
            line=dict(color='#ffd700', width=2, dash='dot'),
            hovertemplate='Safety Floor: <b>%{y:.3f} km</b><extra></extra>'
        ),
        row=1, col=1
    )
    
    # ROW 2: Control Actuation
    actuation_x = [f"Day {i+1}" for i in range(len(days_labels))]
    
    # Bar chart for burns (m/day)
    fig.add_trace(
        go.Bar(
            x=actuation_x,
            y=opt_res['optimal_burns_m_day'],
            name='🚀 Optimal Reboost Thrust (m/day)',
            marker_color='#00d4ff',
            opacity=0.85,
            hovertemplate='Reboost: <b>%{y:.1f} m/day</b><extra></extra>'
        ),
        row=2, col=1, secondary_y=False
    )
    
    # Line chart for alpha attitude feathering factor (right axis)
    fig.add_trace(
        go.Scatter(
            x=actuation_x,
            y=opt_res['optimal_feather_alpha'],
            name='🪶 Drag Factor α (Attitude Feathering)',
            mode='lines+markers',
            line=dict(color='#ff007f', width=2.5, dash='dash'),
            marker=dict(size=9, symbol='diamond', color='#ff007f'),
            hovertemplate='Attitude Drag α: <b>%{y:.3f}</b> (1.0=Ops, 0.4=Feathered)<extra></extra>'
        ),
        row=2, col=1, secondary_y=True
    )
    
    # Layout styling
    fig.update_layout(
        paper_bgcolor='rgba(10, 10, 26, 0.7)',
        plot_bgcolor='rgba(20, 20, 45, 0.5)',
        height=660,
        margin=dict(l=60, r=60, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(family='Space Grotesk', size=11, color='#ffffff')
        ),
        hovermode='x unified',
        font=dict(family='Space Grotesk', color='#8b8ba7')
    )
    
    fig.update_yaxes(
        title_text="Altitude (km)",
        row=1, col=1,
        gridcolor='rgba(0, 212, 255, 0.1)',
        color='#ffffff'
    )
    fig.update_yaxes(
        title_text="Reboost Burn (m/day)",
        row=2, col=1,
        secondary_y=False,
        gridcolor='rgba(0, 212, 255, 0.1)',
        color='#00d4ff'
    )
    fig.update_yaxes(
        title_text="Drag Factor α",
        row=2, col=1,
        secondary_y=True,
        range=[0.3, 1.08],
        gridcolor='rgba(255, 0, 127, 0.1)',
        color='#ff007f'
    )
    fig.update_xaxes(
        gridcolor='rgba(0, 212, 255, 0.1)',
        color='#8b8ba7'
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

@st.fragment
def render_mission_control_view(selected_sat_row, forecast_features_df):
    """
    Renders the operational forecast and the integrated MPC maneuver optimizer (Tab 2).
    """
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <h3 style="font-family: 'Orbitron', sans-serif; color: #00d4ff; font-size: 1.3rem; line-height: 1.35; margin: 0 0 6px 0; letter-spacing: 0.8px;">
            🚀 7-DAY AUTONOMOUS MPC STATION-KEEPING PLANNER
        </h3>
        <p style="color: #8b8ba7; font-size: 0.92rem; margin: 0; line-height: 1.5;">
            Receding-Horizon Model Predictive Control (MPC) Trajectory Optimization under Stratified 5-Seed Ensemble Uncertainty.
        </p>
    </div>
    """, unsafe_allow_html=True)

    sat_name = selected_sat_row.get("satellite_name", "Unknown Satellite")
    initial_alt_km = float(selected_sat_row.get("altitude_km", 500.0))

    # 1. Run Stratified 5-Seed Uncertainty Prediction across the forecast horizon
    pred = predictor if predictor is not None else StratifiedLEOPredictor()
    uncertainty_res = pred.predict_with_uncertainty(
        forecast_features_df, altitude_km=initial_alt_km
    )

    st.info(
        f"**Active Regime Isolation:** `{uncertainty_res.get('regime', 'Deep LEO')}` | "
        f"Mean Predicted Decay: `{np.mean(uncertainty_res['decay_mean_m_day']):.2f} m/day` | "
        f"Conservative P90 Bound: `{np.mean(uncertainty_res['decay_p90_m_day']):.2f} m/day`"
    )

    # Reset the mass input to the new satellite's default when the target changes
    # (the widget key persists across satellites otherwise and keeps the old mass).
    _mass_for = selected_sat_row.get("norad_id")
    if st.session_state.get("_opt_mass_for") != _mass_for:
        st.session_state.pop("opt_sat_mass", None)
        st.session_state["_opt_mass_for"] = _mass_for

    # 2. Interactive Optimizer Controls (4-column layout)
    with st.expander("⚙️ Maneuver Optimizer Configuration & Spacecraft Parameters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sat_mass_kg = st.number_input(
                "Wet Mass (kg)",
                min_value=50.0,
                max_value=15000.0,
                value=float(selected_sat_row.get("mass_kg", 500.0)),
                step=50.0,
                key="opt_sat_mass",
                help="Total wet mass of the spacecraft including onboard propellant."
            )
        with c2:
            isp_sec = st.selectbox(
                "Propulsion System (Isp)",
                options=[220.0, 300.0, 1500.0],
                index=1,
                format_func=lambda x: f"{int(x)}s ({'Hydrazine' if x==220 else 'Biprop' if x==300 else 'Hall Ion'})",
                key="opt_isp",
                help="Specific impulse (Isp) determining propellant consumption via Tsiolkovsky rocket equation."
            )
        with c3:
            deadband_m = st.slider(
                "Safety Corridor Floor (m)",
                min_value=20.0,
                max_value=500.0,
                value=100.0,
                step=10.0,
                key="opt_deadband",
                help="Maximum allowable drop below starting baseline before hard safety floor penalties trigger."
            )
        with c4:
            allow_feather = st.toggle(
                "Solar Panel Feathering (α)",
                value=True,
                key="opt_feather",
                help="Pitches solar arrays edge-on during storm peak to cut aerodynamic drag up to 60%."
            )
            use_p90 = st.toggle(
                "Robust P90 Drag Bound",
                value=True,
                key="opt_use_p90",
                help="Uses 90th-percentile conservative drag bound (mean + 1.28σ) for risk-averse planning."
            )

    # 3. Execute Receding-Horizon MPC Optimization
    optimizer = BStarManeuverOptimizer(sat_mass_kg=sat_mass_kg, isp_sec=isp_sec)
    
    # Ensure forecast_features_df contains 'Kp_max' and 'beta_angle_deg' columns across the 7-day window
    kp_forecast = forecast_features_df["Kp_max"].values[:7] if "Kp_max" in forecast_features_df.columns else np.full(7, 3.5)
    beta_forecast = forecast_features_df["beta_angle_deg"].values[:7] if "beta_angle_deg" in forecast_features_df.columns else np.full(7, 30.0)

    plan = optimizer.optimize_7day_horizon(
        initial_alt_km=initial_alt_km,
        corridor_floor_offset_m=deadband_m,
        decay_mean_m_day=uncertainty_res["decay_mean_m_day"][:7],
        decay_p90_m_day=uncertainty_res["decay_p90_m_day"][:7],
        kp_max_forecast=kp_forecast,
        beta_angle_deg=beta_forecast,
        allow_feathering=allow_feather,
        use_p90_robust=use_p90,
    )

    # 4. KPI Scorecards (Value First, Label Second)
    total_drop_m = float(np.sum(plan["design_drag_m_day"]))
    floor_breached = total_drop_m > deadband_m
    saved_fuel_g = max(0.0, plan['total_naive_fuel_g'] - plan['total_opt_fuel_g'])

    st.markdown(f"""
    <div class="metrics-row" style="margin-top: 14px; margin-bottom: 20px;">
        <div class="metric-card">
            <div class="metric-value glow-green">{plan['total_opt_dv_m_s']:.4f} m/s</div>
            <div class="metric-label">Optimal 7-Day Propulsive Δv</div>
            <div class="metric-sub glow-green">▼ {plan['dv_savings_pct']:.1f}% Saved vs Naive Daily Burn</div>
        </div>
        <div class="metric-card">
            <div class="metric-value glow-cyan">{plan['total_opt_fuel_g']:.2f} g</div>
            <div class="metric-label">Propellant Mass Consumed</div>
            <div class="metric-sub glow-green">Saved {saved_fuel_g:.2f} g onboard fuel</div>
        </div>
        <div class="metric-card">
            <div class="metric-value {'glow-red' if floor_breached else 'glow-yellow'}">-{total_drop_m:.1f} m</div>
            <div class="metric-label">7-Day Uncontrolled Drop</div>
            <div class="metric-sub {'glow-red' if floor_breached else 'glow-green'}">
                {'⚠️ Breaches -' + str(int(deadband_m)) + 'm Safety Floor' if floor_breached else '✅ Remains Within Buffer'}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{int(np.sum(plan['optimal_burns_m_day'] > 0))} Burns</div>
            <div class="metric-label">Scheduled Thruster Firings</div>
            <div class="metric-sub glow-cyan">Timed Post-Storm Recovery</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 5. Interactive Plotly Trajectory Visualization (3-Strategy Comparison)
    fig = go.Figure()
    days_x = [f"Day +{i+1}" for i in range(len(plan["days"]))]
    
    fig.add_trace(go.Scatter(
        x=days_x,
        y=(plan["uncontrolled_alt_km"] - initial_alt_km) * 1000.0,
        mode="lines+markers",
        name="1. Uncontrolled Drift (No Control)",
        line=dict(color="#ff4444", dash="dash", width=2.5),
        marker=dict(size=6, color="#ff4444"),
        hovertemplate="Uncontrolled: <b>%{y:+.1f} m</b><extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=days_x,
        y=(plan["naive_alt_km"] - initial_alt_km) * 1000.0,
        mode="lines",
        name="2. Naive Daily Station-Keeping",
        line=dict(color="#8b8ba7", dash="dot", width=2),
        hovertemplate="Naive: <b>%{y:+.1f} m</b><extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=days_x,
        y=(plan["optimal_alt_km"] - initial_alt_km) * 1000.0,
        mode="lines+markers",
        name="3. B-Star MPC Optimal Steering",
        line=dict(color="#00ff88", width=3.5),
        marker=dict(size=7, color="#00ff88"),
        hovertemplate="B-Star Optimal: <b>%{y:+.1f} m</b><extra></extra>"
    ))
    fig.add_hline(
        y=-deadband_m,
        line_dash="dash",
        line_color="#ffd700",
        annotation_text=f"Hard Safety Corridor Floor (-{deadband_m:.0f} m)",
        annotation_position="bottom right",
        annotation_font=dict(color="#ffd700", family="Space Grotesk")
    )
    fig.update_layout(
        title=dict(
            text=f"<b>7-DAY RELATIVE ALTITUDE CORRIDOR TRAJECTORY — {sat_name.upper()} (Δh in meters)</b>",
            font=dict(family="Orbitron", size=13, color="#00d4ff")
        ),
        xaxis_title="Forecast Horizon",
        yaxis_title="Altitude Deviation Δh (meters)",
        paper_bgcolor="rgba(10, 10, 26, 0.7)",
        plot_bgcolor="rgba(20, 20, 45, 0.5)",
        height=400,
        margin=dict(l=60, r=40, t=60, b=40),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(family="Space Grotesk", size=11, color="#ffffff")
        ),
        font=dict(family="Space Grotesk", color="#8b8ba7")
    )
    fig.update_yaxes(gridcolor="rgba(0, 212, 255, 0.1)")
    fig.update_xaxes(gridcolor="rgba(0, 212, 255, 0.1)")
    st.plotly_chart(fig, use_container_width=True)

    # 6. Daily Command Schedule Table
    st.markdown("""
    <h4 style="color: #00d4ff; font-family: 'Orbitron', sans-serif; font-size: 1.1rem; margin-top: 1.2rem; margin-bottom: 0.6rem;">
        📋 AUTONOMOUS 7-DAY FLIGHT COMMAND SCHEDULE
    </h4>
    """, unsafe_allow_html=True)
    schedule_df = pd.DataFrame({
        "Horizon": [f"Day +{i+1}" for i in range(len(plan["design_drag_m_day"]))],
        "Forecast Kp": kp_forecast,
        "Design Drag (m/d)": plan["design_drag_m_day"],
        "Attitude α": plan["optimal_feather_alpha"],
        "Reboost Burn (m)": plan["optimal_burns_m_day"],
        "Propulsive Δv (m/s)": plan["optimal_dv_m_s"],
        "Flight Command": [
            "🦅 FEATHER PANELS (Hold Thrusters)" if a < 0.95 and b == 0
            else ("🚀 EXECUTE PROGRADE REBOOST" if b > 0 else "🛰️ NOMINAL FREE FLIGHT")
            for a, b in zip(plan["optimal_feather_alpha"], plan["optimal_burns_m_day"])
        ],
    })
    st.dataframe(
        schedule_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Forecast Kp": st.column_config.ProgressColumn("Kp Index", min_value=0, max_value=9, format="%.1f"),
            "Attitude α": st.column_config.NumberColumn("Feathering α", format="%.2f"),
            "Reboost Burn (m)": st.column_config.ProgressColumn("Reboost (m)", min_value=0, max_value=350, format="%.1f m"),
            "Design Drag (m/d)": st.column_config.NumberColumn("Design Drag (m/d)", format="%.1f"),
            "Propulsive Δv (m/s)": st.column_config.NumberColumn("Propulsive Δv (m/s)", format="%.4f"),
        },
    )


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def start_scenario(display_name, date_choice, custom_date=None, weather=None):
    """Quick-start card callback: pre-fills the sidebar widgets, then runs the prediction."""
    ss = st.session_state
    ss['input_mode'] = "Database"
    ss['sat_select'] = display_name
    ss['date_choice'] = date_choice
    if custom_date is not None:
        ss['custom_date_pick'] = custom_date
    ss['_pending_weather'] = weather if weather is not None else 'db'
    ss['_qs_run'] = True


def main():
    # Header
    st.markdown("""
    <div class="brand-bar">
        <span class="brand-name">🛰️ B-STAR</span>
        <span class="brand-sub">Orbital decay early warning &amp; 7-day maneuver planning for Low Earth Orbit</span>
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

    # Initialize session state for date selection
    if 'date_selection' not in st.session_state:
        st.session_state['date_selection'] = 'tomorrow'
        st.session_state['selected_date'] = datetime.now().date() + timedelta(days=1)
    
    if 'last_date' not in st.session_state:
        st.session_state['last_date'] = st.session_state['selected_date']
    
    with st.sidebar:
        st.markdown("""
        <h2 style="font-size: 1.2rem; margin-bottom: 16px; font-family: Orbitron, sans-serif; 
                   color: #00d4ff;">⚙️ MISSION CONTROL</h2>
        """, unsafe_allow_html=True)
        
        # Satellite Selection
        st.markdown('<div class="step-h">1. Satellite</div>', unsafe_allow_html=True)
        input_mode = st.radio(
            "Input method:",
            ["Database", "Custom"],
            horizontal=True,
            key="input_mode",
            label_visibility="collapsed"
        )
        
        if input_mode == "Database":
            default_idx = satellites[satellites['norad_id'] == 20580].index.tolist()[0] if 20580 in satellites['norad_id'].values else 0
            if st.session_state.get('sat_select') not in satellites['display_name'].tolist():
                st.session_state.pop('sat_select', None)
            selected_display = st.selectbox(
                "Satellite:",
                satellites['display_name'].tolist(),
                index=default_idx,
                key="sat_select",
                label_visibility="collapsed"
            )
            selected_norad = satellites[satellites['display_name'] == selected_display]['norad_id'].values[0]
            selected_name = satellites[satellites['display_name'] == selected_display]['satellite_name'].values[0]
        else:
            selected_norad = st.number_input("NORAD ID:", min_value=1, value=20580, step=1)
            selected_name = st.text_input("Name:", value="Custom Satellite")
        
        if STRATIFIED_AVAILABLE and is_active_payload(selected_norad):
            st.caption("⚠️ **Active Payload**: Regularly reboosted station.")
        
        st.divider()
        
        # Date Selection - Simplified with radio for reliability
        st.markdown('<div class="step-h">2. Date</div>', unsafe_allow_html=True)
        
        # Quick date options as radio buttons (more reliable than individual buttons)
        date_options = ["Tomorrow", "Next Week", "Custom Date"]
        
        # Initialize date choice if not present
        if 'date_choice' not in st.session_state:
            st.session_state['date_choice'] = "Tomorrow"
        
        date_choice = st.radio(
            "Quick select:",
            date_options,
            key="date_choice",
            horizontal=True,
            label_visibility="collapsed"
        )
        
        # Calculate the actual date based on selection
        if date_choice == "Tomorrow":
            selected_date = datetime.now().date() + timedelta(days=1)
        elif date_choice == "Next Week":
            selected_date = datetime.now().date() + timedelta(days=7)
        else:
            # Custom date picker (the calendar has its own year / month dropdowns)
            st.session_state.setdefault('custom_date_pick', datetime(2024, 1, 15).date())
            selected_date = st.date_input(
                "Pick date:",
                key="custom_date_pick",
                min_value=datetime(2015, 1, 1).date(),
                max_value=datetime.now().date() + timedelta(days=365),
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
            st.caption("Replays the past and compares the forecast with what really happened.")
        else:
            st.info("🔮 Forecast Mode")
            st.caption("Looks ahead using propagated orbit geometry. No ground truth exists yet.")
        
        st.divider()
        
        # Space Weather Baseline Retrieval from Database (T-1 causal lookup in backtest mode)
        sw_query_date = selected_date - timedelta(days=1) if is_backtest else selected_date
        sw_baseline = get_space_weather_for_date(sw_query_date)
        
        # Auto-synchronize sliders to database baseline when date changes or on first run
        if 'sw_target_date' not in st.session_state or st.session_state['sw_target_date'] != selected_date:
            st.session_state['sw_target_date'] = selected_date
            st.session_state['slider_kp_mean'] = min(9.0, max(0.0, float(sw_baseline['kp_mean'])))
            st.session_state['slider_kp_max'] = min(9.0, max(0.0, float(sw_baseline['kp_max'])))
            st.session_state['slider_f107'] = min(350.0, max(60.0, float(sw_baseline['f107'])))
            
        # Quick-start cards leave a pending weather request: apply it after the baseline sync
        if '_pending_weather' in st.session_state:
            _pw = st.session_state.pop('_pending_weather')
            if _pw == 'db':
                _pw = (sw_baseline['kp_mean'], sw_baseline['kp_max'], sw_baseline['f107'])
            st.session_state['slider_kp_mean'] = min(9.0, max(0.0, float(_pw[0])))
            st.session_state['slider_kp_max'] = min(9.0, max(0.0, float(_pw[1])))
            st.session_state['slider_f107'] = min(350.0, max(60.0, float(_pw[2])))

        def reset_weather_to_db():
            st.session_state['slider_kp_mean'] = min(9.0, max(0.0, float(sw_baseline['kp_mean'])))
            st.session_state['slider_kp_max'] = min(9.0, max(0.0, float(sw_baseline['kp_max'])))
            st.session_state['slider_f107'] = min(350.0, max(60.0, float(sw_baseline['f107'])))

        st.markdown('<div class="step-h">3. Space weather</div>', unsafe_allow_html=True)
        st.caption("⚡ **Quick Storm Scenarios:**")
        p_col1, p_col2, p_col3 = st.columns(3)
        if p_col1.button("☀️ Quiet", use_container_width=True, help="Quiet Solar Minimum (Kp=1.3, F10.7=75)"):
            st.session_state['slider_kp_mean'] = 1.3
            st.session_state['slider_kp_max'] = 2.0
            st.session_state['slider_f107'] = 75.0
            st.rerun()
        if p_col2.button("⚡ G3", use_container_width=True, help="Strong Geomagnetic Storm (Kp=5.5, F10.7=185)"):
            st.session_state['slider_kp_mean'] = 5.5
            st.session_state['slider_kp_max'] = 7.0
            st.session_state['slider_f107'] = 185.0
            st.rerun()
        if p_col3.button("🔥 G5 Max", use_container_width=True, help="May 2024 G5 Superstorm (Kp=7.8, F10.7=240)"):
            st.session_state['slider_kp_mean'] = 7.8
            st.session_state['slider_kp_max'] = 9.0
            st.session_state['slider_f107'] = 240.0
            st.rerun()
        
        kp_mean = st.slider("Kp Mean (24h)", 0.0, 9.0, key="slider_kp_mean", step=0.1)
        kp_max = st.slider("Kp Max (Peak Shock)", 0.0, 9.0, key="slider_kp_max", step=0.1)
        f107 = st.slider("F10.7 Solar EUV Flux (sfu)", 60.0, 350.0, key="slider_f107", step=0.5)
        
        # Track if user manually modified the sliders away from the database baseline
        is_overridden = (
            abs(kp_mean - sw_baseline['kp_mean']) > 0.05 or
            abs(kp_max - sw_baseline['kp_max']) > 0.05 or
            abs(f107 - sw_baseline['f107']) > 0.5
        )
        
        if is_overridden:
            st.caption(f"⚠️ **Manual Override Active**\n(DB Baseline: Kp={sw_baseline['kp_mean']}, F10.7={sw_baseline['f107']})")
            if st.button("🔄 Reset to DB Baseline", use_container_width=True):
                reset_weather_to_db()
                st.rerun()
        else:
            tag = " [T-1 Prior Baseline]" if is_backtest else ""
            st.caption(f"📡 **DB Synced**: {sw_baseline['source']}{tag}")
        
        st.divider()
        
        # Model Engine Status
        if STRATIFIED_AVAILABLE and hasattr(model, 'get_status'):
            m_stat = model.get_status()
            if m_stat.get('stratified', False):
                st.caption(f"⚡ **Stratified LEO Engine**\n• Deep LEO (<{m_stat['threshold_km']:.0f} km)\n• Upper LEO (≥{m_stat['threshold_km']:.0f} km)\n• Calibration Active")
            else:
                st.caption("⚡ **Unified LEO Engine**")
        
        # Predict Button
        predict_clicked = st.button("🚀 PREDICT DECAY & OPTIMIZE", use_container_width=True, type="primary")
        
        # If prediction active, allow returning to mission briefing
        if st.session_state.get('prediction_active', False):
            if st.button("🏠 Return to Mission Briefing", use_container_width=True):
                st.session_state['prediction_active'] = False
                st.session_state['predicted_norad'] = None
                st.session_state['predicted_date'] = None
                st.session_state['should_animate'] = False
                st.rerun()
    
    # =============================================================================
    # STATE MANAGEMENT & MAIN CONTENT
    # =============================================================================
    if 'prediction_active' not in st.session_state:
        st.session_state['prediction_active'] = False
        st.session_state['predicted_norad'] = None
        st.session_state['predicted_date'] = None
        st.session_state['should_animate'] = False
        
    if st.session_state.pop('_qs_run', False):
        predict_clicked = True
    if predict_clicked:
        st.session_state['prediction_active'] = True
        st.session_state['predicted_norad'] = selected_norad
        st.session_state['predicted_date'] = selected_date
        st.session_state['should_animate'] = True
        
    # If the user changed the target satellite or date while prediction was active, reset to prompt prediction
    target_changed = False
    if st.session_state.get('prediction_active', False) and not predict_clicked:
        if (st.session_state.get('predicted_norad') != selected_norad or 
            st.session_state.get('predicted_date') != selected_date):
            st.session_state['prediction_active'] = False
            target_changed = True

    if st.session_state.get('prediction_active', False):
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
        # Show processing animation ONLY when user explicitly pressed the predict button
        if st.session_state.get('should_animate', False):
            show_processing_animation()
            st.session_state['should_animate'] = False        
        # Calculate features and predict
        start_time = time.time()
        X_pred = calculate_features(selected_date, sat_data, kp_mean, kp_max, f107, 
                                    is_backtest=is_backtest, df=df, norad_id=selected_norad,
                                    is_overridden=is_overridden)
        
        alt_arr = X_pred['altitude_km'].values if 'altitude_km' in X_pred.columns else None
        alt_val = float(alt_arr[0]) if alt_arr is not None and len(alt_arr) > 0 else 500.0
        
        # 1-day prediction with 5-seed uncertainty
        if predictor is not None and hasattr(predictor, 'predict_with_uncertainty'):
            try:
                unc_res_1d = predictor.predict_with_uncertainty(X_pred, altitudes=alt_arr, apply_calibration=True)
                predicted_decay = float(unc_res_1d['decay_mean_m_day'][0])
                pred_decay_std = float(unc_res_1d['decay_std_m_day'][0])
                pred_decay_p90 = float(unc_res_1d['decay_p90_m_day'][0])
            except Exception:
                try:
                    predicted_decay = float(model.predict(X_pred, altitudes=alt_arr, apply_calibration=True)[0])
                except TypeError:
                    predicted_decay = float(model.predict(X_pred)[0])
                pred_decay_std = predicted_decay * (0.12 if kp_max < 5.0 else 0.22)
                pred_decay_p90 = predicted_decay + 1.28 * pred_decay_std
        else:
            try:
                predicted_decay = float(model.predict(X_pred, altitudes=alt_arr, apply_calibration=True)[0])
            except TypeError:
                predicted_decay = float(model.predict(X_pred)[0])
            pred_decay_std = predicted_decay * (0.12 if kp_max < 5.0 else 0.22)
            pred_decay_p90 = predicted_decay + 1.28 * pred_decay_std
            
        processing_time = time.time() - start_time
        
        # Determine risk
        risk_level, ratio = determine_risk_level(predicted_decay, baseline)
        # Get actual value and ground truth audit if backtesting
        gt_info = get_backtest_ground_truth(df, selected_norad, selected_date) if is_backtest else {'has_ground_truth': False}
        actual_decay = gt_info.get('actual_decay')
        is_maneuver = gt_info.get('is_maneuver', False)
        
        # =============================================================================
        # PERSISTENT TOP TELEMETRY HUD BAR
        # =============================================================================
        is_deep_leo = alt_val < 550.0
        regime_badge = (
            "<span class='badge-pill badge-deep'>🔥 Deep LEO (&lt; 550 km · Atomic Oxygen)</span>"
            if is_deep_leo
            else "<span class='badge-pill badge-upper'>🌌 Upper LEO (&ge; 550 km · Helium Exosphere)</span>"
        )
        mode_badge = (
            "<span class='badge-pill badge-backtest'>🛡️ Backtest Mode (Causal T-1)</span>"
            if is_backtest
            else "<span class='badge-pill badge-forecast'>🔮 Forecast Mode (SGP4)</span>"
        )
        
        st.markdown(f"""
        <div class="hud-bar">
            <div class="hud-item">
                <span class="hud-lbl">Target Spacecraft</span>
                <span class="hud-val">🛰️ {selected_name} (#{selected_norad})</span>
            </div>
            <div class="hud-item">
                <span class="hud-lbl">Mean Altitude</span>
                <span class="hud-val">{alt_val:.1f} km</span>
            </div>
            <div class="hud-item">
                <span class="hud-lbl">Thermospheric Regime</span>
                <div>{regime_badge}</div>
            </div>
            <div class="hud-item">
                <span class="hud-lbl">Target Epoch & Protocol</span>
                <div><span class="hud-val" style="margin-right: 8px;">{selected_date.strftime('%Y-%m-%d')}</span>{mode_badge}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        

        # Active Payload Alert
        if STRATIFIED_AVAILABLE and is_active_payload(selected_norad):
            st.warning(
                "⚠️ **Active / Reboosted Payload Notice**: Space stations undergo periodic thruster reboosts "
                "and attitude maneuvers. Predictions represent natural unassisted aerodynamic drag potential."
            )
            
        # Space Weather Baseline / Override Notice
        if is_overridden:
            st.info(
                f"🌡️ **Custom Space Weather Override Active**: Kp Mean = **{kp_mean:.2f}** | Kp Max = **{kp_max:.2f}** | "
                f"F10.7 = **{f107:.1f} sfu** (Database baseline was: Kp={sw_baseline['kp_mean']}, F10.7={sw_baseline['f107']})"
            )
        
        # =============================================================================
        # 3 TASK-ORIENTED WORKSPACE TABS
        # =============================================================================
        tab_advisory, tab_mpc, tab_audit, tab_trust = st.tabs([
            "🎯 Forecast & risk",
            "🚀 7-day maneuver plan",
            "🔬 Audit & features",
            "🧭 Can I trust this?"
        ])
        
        # ---------------------------------------------------------------------
        # TAB 1: 24H DRAG PREDICTIONS & ACTIONABLE RISK ADVISORY
        # ---------------------------------------------------------------------
        with tab_advisory:
            risk_color = "glow-green" if ratio < 120 else ("glow-yellow" if ratio <= 200 else "glow-red")
            risk_status = "NOMINAL" if ratio < 120 else ("ELEVATED DRAG" if ratio <= 200 else "CRITICAL STORM SURGE")
            
            # Plain-language verdict (same thresholds as the scorecards: <120 / 120-200 / >200)
            _lvl = 'nominal' if ratio < 120 else ('elevated' if ratio <= 200 else 'critical')
            _lvl_color = {'nominal': '#00ff88', 'elevated': '#ffd700', 'critical': '#ff4444'}[_lvl]
            _lvl_label = {'nominal': 'Nominal', 'elevated': 'Elevated drag', 'critical': 'Critical storm surge'}[_lvl]
            _small = ("<p>In absolute terms this is small. Percentages swing a lot when decay is below a few metres per day.</p>"
                      if predicted_decay < 3.0 else "")
            _reality = ""
            if actual_decay is not None:
                _reality = (f"<p>Reality check: a reboost burn was detected that day, so the measured value ({actual_decay:.1f} m/day) is not natural decay.</p>"
                            if is_maneuver else
                            f"<p>Reality check: the satellite actually lost {actual_decay:.1f} m/day.</p>")
            st.markdown(
                f'<div class="verdict" style="--vt:{_lvl_color}">'
                f'<span class="verdict-badge">{_lvl_label}</span>'
                f'<div class="verdict-title">{selected_name} should lose about {predicted_decay:.1f} m on {selected_date.strftime("%d %b %Y")}</div>'
                f'<p>That is {ratio:.0f}% of its quiet-sun rate. Plan for up to {pred_decay_p90:.1f} m (conservative P90 estimate).</p>'
                f'{_small}{_reality}</div>',
                unsafe_allow_html=True
            )

            # Value-First, Label-Second 4 Scorecards
            st.markdown(f"""
            <div class="metrics-row">
                <div class="metric-card">
                    <div class="metric-value glow-cyan">{predicted_decay:.2f} m/day</div>
                    <div class="metric-label">Predicted Orbital Decay (μ)</div>
                    <div class="metric-sub" style="color: #8b8ba7;">5-Seed Spread: ±{pred_decay_std:.2f} m/day</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{pred_decay_p90:.2f} m/day</div>
                    <div class="metric-label">Conservative P90 Bound (μ + 1.28σ)</div>
                    <div class="metric-sub glow-cyan">Used for Risk-Averse MPC Planning</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{baseline:.2f} m/day</div>
                    <div class="metric-label">Quiet-Sun Historical Baseline</div>
                    <div class="metric-sub" style="color: #8b8ba7;">Regime Reference Floor</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {risk_color}">{ratio:.1f}%</div>
                    <div class="metric-label">% of Quiet Baseline ({risk_status})</div>
                    <div class="metric-sub {risk_color}">Kp_max = {kp_max:.1f} | F10.7 = {f107:.0f} sfu</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col_gauge, col_checklist = st.columns([1.15, 1], gap="medium")
            
            with col_gauge:
                st.plotly_chart(create_risk_gauge(ratio, risk_level), use_container_width=True)
                st.plotly_chart(create_comparison_chart(predicted_decay, baseline, actual_decay, is_maneuver=is_maneuver), use_container_width=True)
                
            with col_checklist:
                rec = RECOMMENDATIONS[risk_level]
                actions_html = ''.join([f'<li style="margin-bottom: 6px;">{action}</li>' for action in rec['actions']])
                
                feather_directive = "Command solar arrays to feather mode (α = 0.50) during storm peak to mitigate drag surge." if ratio > 150 else "Maintain nominal nadir science pointing (α = 1.00)."
                thruster_directive = "Quarantine prograde station-keeping burns until geomagnetic shock subsides." if kp_max >= 5.0 else "Corridor clear for routine station-keeping maneuvers."
                cadence_directive = "Request high-rate Space-Track radar tasking every 6 hours." if ratio > 200 else "Standard 24-hour TLE ingestion cycle sufficient."
                
                st.markdown(f"""
                <div class="metric-card" style="padding: 22px; text-align: left; height: 100%; box-sizing: border-box; justify-content: flex-start; align-items: flex-start;">
                    <div class="metric-label" style="color: #00d4ff; font-weight: bold; margin-top: 0; font-size: 0.82rem;">
                        FLIGHT DIRECTOR ACTION CHECKLIST
                    </div>
                    <h3 style="font-family: Orbitron, sans-serif; margin: 8px 0 14px 0; color: #ffffff; font-size: 1.15rem;">
                        Status: <span class="{risk_color}">{risk_status}</span>
                    </h3>
                    <ol style="color: #d1d5db; font-size: 0.88rem; line-height: 1.65; padding-left: 20px; margin: 0 0 18px 0;">
                        <li><b>Attitude Steering:</b> {feather_directive}</li>
                        <li><b>Thruster Hold:</b> {thruster_directive}</li>
                        <li><b>Tracking Cadence:</b> {cadence_directive}</li>
                    </ol>
                    <div style="border-top: 1px solid rgba(0, 212, 255, 0.2); padding-top: 14px; width: 100%;">
                        <div style="font-size: 0.8rem; color: #8b8ba7; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; font-weight: 600;">
                            Recommended Operational Directives:
                        </div>
                        <ul style="color: #ffffff; font-size: 0.88rem; line-height: 1.6; padding-left: 18px; margin: 0;">
                            {actions_html}
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        # ---------------------------------------------------------------------
        # TAB 2: 7-DAY AUTONOMOUS MPC STATION-KEEPING PLANNER
        # ---------------------------------------------------------------------
        with tab_mpc:
            KNOWN_MASSES = {
                20580: 11110.0,  # Hubble Space Telescope (HST)
                33053: 4303.0,   # Fermi (GLAST)
                25994: 4864.0,   # TERRA (EOS AM-1)
                27386: 8211.0,   # ENVISAT
                22803: 9000.0,   # SL-16 Rocket Body
                25544: 450000.0, # ISS (Zarya)
                48274: 100000.0, # CSS (Tianhe-1)
            }
            sat_mass_default = KNOWN_MASSES.get(selected_norad, 500.0)
            selected_sat_row = {
                "satellite_name": selected_name,
                "altitude_km": alt_val,
                "norad_id": selected_norad,
                "mass_kg": sat_mass_default
            }
            df_7day, _, _, _ = generate_7day_forecast_features(
                base_date=selected_date,
                sat_data=sat_data,
                norad_id=selected_norad,
                scenario_type='severe_storm',
                custom_kp_mean=kp_mean,
                custom_kp_max=kp_max,
                custom_f107=f107
            )
            if OPTIMIZER_AVAILABLE and (predictor is not None or STRATIFIED_AVAILABLE):
                render_mission_control_view(selected_sat_row, df_7day)
            else:
                st.warning("The 7-day planner is unavailable: `step7_maneuver_optimizer` or `stratified_model` failed to import. Check the terminal for the import notice.")
            
        # ---------------------------------------------------------------------
        # TAB 3: CAUSAL BACKTEST AUDIT & 23-FEATURE TELEMETRY
        # ---------------------------------------------------------------------
        with tab_audit:
            if is_backtest and gt_info.get('has_ground_truth'):
                actual_decay_val = gt_info['actual_decay']
                decay_happened = gt_info['decay_happened']
                maneuver_happened = gt_info['is_maneuver']
                prior_date = gt_info['prior_date']
                prior_alt = gt_info['prior_alt_km']
                actual_alt = gt_info['actual_alt_km']
                alt_delta_m = gt_info['alt_delta_m']
                actual_kp = gt_info['actual_kp_mean']
                actual_f107 = gt_info['actual_f107']
                
                error = predicted_decay - actual_decay_val
                abs_error = abs(error)
                denom = max(abs(actual_decay_val), 1.0)
                accuracy = max(0.0, (1.0 - abs_error / denom) * 100.0)
                days_lag = (selected_date - prior_date).days if prior_date else 1
                
                # Anti-leakage protocol banner
                st.markdown(f"""
                <div style="background: rgba(0, 255, 136, 0.08); border: 1px solid rgba(0, 255, 136, 0.3); 
                            border-radius: 8px; padding: 14px 18px; margin-bottom: 18px;">
                    <div style="color: #00ff88; font-family: Orbitron, sans-serif; font-size: 0.95rem; font-weight: bold; margin-bottom: 4px;">
                        🛡️ STRICT CAUSAL BACKTEST PROTOCOL (ANTI-LEAKAGE ACTIVE)
                    </div>
                    <div style="color: #c9d1d9; font-size: 0.88rem; line-height: 1.7;">
                        • <strong>Prediction Target:</strong> {selected_date.strftime('%B %d, %Y')} (Day T)<br>
                        • <strong>Telemetry Sourced Strictly From:</strong> {prior_date.strftime('%B %d, %Y') if prior_date else 'Prior Epoch'} (T - {days_lag} Day{'s' if days_lag > 1 else ''})<br>
                        • <strong>Feature Isolation:</strong> Target date physical telemetry was quarantined to verify genuine forward predictive performance.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Diagnostic Status Cards Row
                decay_badge = "✅ Natural Decay" if decay_happened else "⚠️ Orbital Boost (No Decay)"
                decay_color = "glow-green" if decay_happened else "glow-red"
                
                maneuver_badge = "🚀 Reboost Maneuver" if maneuver_happened else "🛰️ Nominal Free Flight"
                maneuver_color = "glow-red" if maneuver_happened else "glow-cyan"
                
                err_color = "glow-green" if abs_error < 15.0 else ("glow-yellow" if abs_error < 30.0 else "glow-red")
                
                st.markdown(f"""
                <div class="metrics-row">
                    <div class="metric-card">
                        <div class="metric-value {decay_color}" style="font-size: 1.25rem;">{decay_badge}</div>
                        <div class="metric-label">1. Did Decay Occur? (Δa > 0.01m)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value {maneuver_color}" style="font-size: 1.25rem;">{maneuver_badge}</div>
                        <div class="metric-label">2. Maneuver Detection Audit</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value glow-yellow" style="font-size: 1.4rem;">{actual_decay_val:.2f} m/d</div>
                        <div class="metric-label">3. Realized Ground Truth Decay</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value {err_color}" style="font-size: 1.4rem;">{accuracy:.1f}%</div>
                        <div class="metric-label">4. Forward Accuracy (Err: {abs_error:.2f} m)</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Ground Truth Audit Breakdown Table
                decay_desc = f"Orbital decay confirmed: Satellite lost {actual_decay_val:.2f} meters of altitude in 24h." if decay_happened else f"Decay was negative ({actual_decay_val:+.2f} m/day); satellite gained altitude due to propulsive burn."
                maneuver_desc = "Active thruster boost or stationkeeping maneuver detected on this date." if maneuver_happened else "No thruster burns detected; orbital trajectory was governed strictly by natural atmospheric drag."
                alt_shift_desc = f"{prior_alt:.2f} km → {actual_alt:.2f} km ({alt_delta_m:+.1f} m net altitude shift)" if prior_alt else f"Measured at {actual_alt:.2f} km"
                weather_desc = f"Input used (T-{days_lag}): Kp={kp_mean:.2f}, F10.7={f107:.1f} sfu | Actual realized on Day T: Kp={actual_kp:.2f}, F10.7={actual_f107:.1f} sfu"
                
                st.markdown(f"""
                <div class="glass-card" style="padding: 20px; margin-bottom: 22px;">
                    <h3 style="margin-top: 0; color: #00d4ff; font-family: Orbitron, sans-serif; font-size: 1.05rem;">
                        🎯 DAY T GROUND TRUTH DIAGNOSTIC AUDIT
                    </h3>
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem; color: #c9d1d9;">
                        <tr style="border-bottom: 1px solid rgba(0,212,255,0.15);">
                            <td style="padding: 10px 8px; font-weight: bold; color: #8b8ba7; width: 28%;">Decay Occurrence</td>
                            <td style="padding: 10px 8px; color: {'#00ff88' if decay_happened else '#ff4d6d'}; font-weight: 600;">{decay_desc}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid rgba(0,212,255,0.15);">
                            <td style="padding: 10px 8px; font-weight: bold; color: #8b8ba7;">Maneuver Detection</td>
                            <td style="padding: 10px 8px; color: {'#ff4d6d' if maneuver_happened else '#00d4ff'}; font-weight: 600;">{maneuver_desc}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid rgba(0,212,255,0.15);">
                            <td style="padding: 10px 8px; font-weight: bold; color: #8b8ba7;">Altitude Transition</td>
                            <td style="padding: 10px 8px; color: #ffffff;">{alt_shift_desc}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid rgba(0,212,255,0.15);">
                            <td style="padding: 10px 8px; font-weight: bold; color: #8b8ba7;">Space Weather Evolution</td>
                            <td style="padding: 10px 8px; color: #ffffff;">{weather_desc}</td>
                        </tr>
                        <tr>
                            <td style="padding: 10px 8px; font-weight: bold; color: #8b8ba7;">Model vs Reality</td>
                            <td style="padding: 10px 8px; color: #ffd700;">
                                Predicted: <strong>{predicted_decay:.2f} m/day</strong> | Actual: <strong>{actual_decay_val:.2f} m/day</strong> | 
                                Residual: <strong>{error:+.2f} m/day</strong> (Accuracy: <strong>{accuracy:.1f}%</strong>)
                            </td>
                        </tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
            elif not is_backtest:
                cur_beta = float(X_pred['beta_angle_deg'].iloc[0]) if 'beta_angle_deg' in X_pred.columns else 0.0
                cur_sef = float(X_pred['sun_exposure_factor'].iloc[0]) if 'sun_exposure_factor' in X_pred.columns else 0.8
                st.markdown(f"""
                <div style="background: rgba(0, 212, 255, 0.08); border: 1px solid rgba(0, 212, 255, 0.3); 
                            border-radius: 8px; padding: 14px 18px; margin-bottom: 18px;">
                    <div style="color: #00d4ff; font-family: Orbitron, sans-serif; font-size: 0.95rem; font-weight: bold; margin-bottom: 4px;">
                        🔮 DYNAMIC SGP4 PROPAGATED GEOMETRY (FORECAST MODE)
                    </div>
                    <div style="color: #c9d1d9; font-size: 0.88rem; line-height: 1.7;">
                        • <strong>Forward Horizon:</strong> +{(selected_date - datetime.now().date()).days} Days<br>
                        • <strong>Orbital Altitude:</strong> {alt_val:.1f} km<br>
                        • <strong>Solar Beta Angle (β):</strong> {cur_beta:+.1f}°<br>
                        • <strong>Sun Exposure Factor (SEF):</strong> {cur_sef:.3f} (Diurnal bulge solar irradiance)
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.info("ℹ️ No ground-truth observation exists for this date in the training database, so there is nothing to audit against. Pick another past date, or a future date for a forecast.")

            # 23-Feature Vector Inspector
            st.markdown("""
            <h4 style="color: #00d4ff; font-family: 'Orbitron', sans-serif; font-size: 1.1rem; margin-top: 1.2rem; margin-bottom: 0.8rem;">
                🧬 23-FEATURE ASTRODYNAMIC & THERMOSPHERIC INPUT VECTOR
            </h4>
            """, unsafe_allow_html=True)
            
            c_f1, c_f2, c_f3 = st.columns(3)
            with c_f1:
                st.markdown("**☀️ Solar & Geometry (Features 1–4, 21–23)**")
                f_geom = {
                    "Feature": ["sun_exposure_factor", "beta_angle_deg", "f107_obs", "f107_81d_avg", "sin_doy", "cos_doy", "static_bc_est"],
                    "Value": [
                        f"{float(X_pred.get('sun_exposure_factor', [0.8]).iloc[0]):.4f}",
                        f"{float(X_pred.get('beta_angle_deg', [30.0]).iloc[0]):+.2f}°",
                        f"{float(X_pred.get('f107_obs', [f107]).iloc[0]):.1f} sfu",
                        f"{float(X_pred.get('f107_81d_avg', [f107]).iloc[0]):.1f} sfu",
                        f"{float(X_pred.get('sin_doy', [0.0]).iloc[0]):.4f}",
                        f"{float(X_pred.get('cos_doy', [1.0]).iloc[0]):.4f}",
                        f"{float(X_pred.get('static_bc_est', [1.0e7]).iloc[0]):.2e}",
                    ]
                }
                st.dataframe(pd.DataFrame(f_geom), hide_index=True, use_container_width=True)
                
            with c_f2:
                st.markdown("**⚡ Geomagnetic Thermal Lags (Features 5–15)**")
                f_geo = {
                    "Feature": ["Kp_mean", "Kp_max", "Kp_min", "Kp_std", "Kp_Lag_3h", "Kp_Lag_6h", "Kp_Lag_12h", "Kp_Lag_24h", "Kp_Roll_24h_Mean", "Kp_Roll_72h_Mean"],
                    "Value": [
                        f"{float(X_pred.get('Kp_mean', [kp_mean]).iloc[0]):.2f}",
                        f"{float(X_pred.get('Kp_max', [kp_max]).iloc[0]):.2f}",
                        f"{float(X_pred.get('Kp_min', [0.0]).iloc[0]):.2f}",
                        f"{float(X_pred.get('Kp_std', [0.5]).iloc[0]):.2f}",
                        f"{float(X_pred.get('Kp_Lag_3h', [kp_mean]).iloc[0]):.2f}",
                        f"{float(X_pred.get('Kp_Lag_6h', [kp_mean]).iloc[0]):.2f}",
                        f"{float(X_pred.get('Kp_Lag_12h', [kp_mean]).iloc[0]):.2f}",
                        f"{float(X_pred.get('Kp_Lag_24h', [kp_mean]).iloc[0]):.2f}",
                        f"{float(X_pred.get('Kp_Roll_24h_Mean', [kp_mean]).iloc[0]):.2f}",
                        f"{float(X_pred.get('Kp_Roll_72h_Mean', [kp_mean]).iloc[0]):.2f}",
                    ]
                }
                st.dataframe(pd.DataFrame(f_geo), hide_index=True, use_container_width=True)
                
            with c_f3:
                st.markdown("**🪐 SGP4 Keplerian State (Features 16–20)**")
                f_sgp4 = {
                    "Feature": ["semi_major_axis_km", "altitude_km", "perigee_alt_km", "eccentricity", "inclination_deg"],
                    "Value": [
                        f"{float(X_pred.get('semi_major_axis_km', [6778.0]).iloc[0]):.2f} km",
                        f"{float(X_pred.get('altitude_km', [alt_val]).iloc[0]):.2f} km",
                        f"{float(X_pred.get('perigee_alt_km', [alt_val-5]).iloc[0]):.2f} km",
                        f"{float(X_pred.get('eccentricity', [0.0005]).iloc[0]):.6f}",
                        f"{float(X_pred.get('inclination_deg', [51.6]).iloc[0]):.2f}°",
                    ]
                }
                st.dataframe(pd.DataFrame(f_sgp4), hide_index=True, use_container_width=True)

        with tab_trust:
            if is_deep_leo:
                st.success("**Reasonably reliable here.** On unseen 2026 data, 85% of Deep LEO forecasts landed within 50% of the real value.")
            else:
                st.warning("**Treat this as an order of magnitude.** Absolute error is small (about 1.5 m/day), but only 28% of Upper LEO "
                           "forecasts landed within 50% of the real value, and R² is negative (-0.70), so a constant guess would score better on that measure.")
            st.markdown("**Unseen 2026 holdout (Jan 1 to Oct 19, 2026)**")
            st.dataframe(pd.DataFrame({
                "Metric": ["Test samples", "Mean actual decay (m/day)", "RMSE (m/day)", "MAE (m/day)",
                           "R²", "Within 50% of actual", "Median % error"],
                "Deep LEO (<550 km)": ["544", "28.24", "12.37", "8.69", "0.46", "85.1%", "26.7%"],
                "Upper LEO (>=550 km)": ["745", "1.71", "1.80", "1.52", "-0.70", "28.2%", "109.5%"],
            }), hide_index=True, use_container_width=True)
            st.markdown(
                "**Known limits**\n"
                "- Large surges are underpredicted. Above about 60 m/day the model has missed by up to 31 m/day, which is why planning uses the P90 bound.\n"
                "- Deep LEO R² is 0.46: good at spotting that something is happening, weaker on the exact size.\n"
                "- Percent errors look huge for Upper LEO because true values are tiny (a 0.9 m miss on 0.4 m is 225%).\n"
                "- ISS and CSS are excluded from training because their reboosts look like negative drag.\n"
                "- Manual space-weather overrides are your own inputs; the model has not been validated on every combination.\n"
                "- Feathering savings in the planner assume panels cut drag by up to 50%. That is a modelling assumption, not a measurement."
            )

        # Processing Time Footer
        st.markdown(f"""
        <div style="text-align: center; margin-top: 28px; padding: 16px; 
                    border-top: 1px solid rgba(0,212,255,0.2);">
            <p style="font-size: 0.88rem; color: #8b8ba7; margin: 0;">
                ⚡ Prediction completed in <span style="color: #00d4ff; font-family: Orbitron;">{processing_time:.3f}s</span>
                &nbsp;|&nbsp; Engine: <span style="color: #00ff88;">Stratified Dual-Regime LEO (Deep / Upper)</span>
                &nbsp;|&nbsp; Telemetry: <span style="color: #ffd700;">23 Physical Astrodynamic Features</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        if target_changed:
            st.info(f"🎯 Target changed to **{selected_name}** ({selected_date.strftime('%B %d, %Y')}). Click **🚀 PREDICT DECAY & OPTIMIZE** in the sidebar to run analysis.")
        
        # Mission Briefing Home Screen
        st.markdown("""
        <div style="margin-bottom: 14px;">
            <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: clamp(1.5rem, 3vw, 2rem); color: #ffffff; line-height: 1.25;">
                Will my satellite sink faster than usual?
            </div>
            <p style="color: #8b8ba7; font-size: 1.02rem; max-width: 820px; margin: 8px 0 0 0; line-height: 1.6;">
                B-Star forecasts daily orbital decay for Low Earth Orbit (up to 1,000 km) from space weather,
                shows how confident it is, and plans a 7-day, fuel-efficient reboost schedule. Start with one of these, or use the sidebar.
            </p>
        </div>
        """, unsafe_allow_html=True)

        def _disp(n):
            m_ = satellites[satellites['norad_id'] == n]
            return m_['display_name'].iloc[0] if len(m_) else None

        _G5 = (7.8, 9.0, 240.0)
        _cards = []
        if _disp(20580):
            _cards.append(("Hubble, tomorrow", "A normal forecast on a Deep LEO satellite.",
                           dict(display_name=_disp(20580), date_choice="Tomorrow")))
            _cards.append(("Replay the May 2024 G5 storm", "See how drag surged and what the planner would have done.",
                           dict(display_name=_disp(20580), date_choice="Custom Date", custom_date=datetime(2024, 5, 11).date())))
        if _disp(27386):
            _cards.append(("G5 storm next week on ENVISAT", "Stress-test an Upper LEO object with a what-if storm.",
                           dict(display_name=_disp(27386), date_choice="Next Week", weather=_G5)))
        if _cards:
            for _col, (_title, _desc, _kw) in zip(st.columns(len(_cards)), _cards):
                with _col.container(border=True):
                    st.markdown(f"**{_title}**")
                    st.caption(_desc)
                    st.button("Open", key=f"qs_{_title}", on_click=start_scenario, kwargs=_kw, use_container_width=True)

        st.markdown(f"""
        <div class="metrics-row" style="margin-top: 18px;">
            <div class="metric-card">
                <div class="metric-value glow-cyan">{len(satellites)} LEO Objects</div>
                <div class="metric-label">Spacecraft in the catalog</div>
            </div>
            <div class="metric-card">
                <div class="metric-value glow-green">172K+ TLEs</div>
                <div class="metric-label">SQLite astrodynamic cache</div>
            </div>
            <div class="metric-card">
                <div class="metric-value glow-cyan">23 Features</div>
                <div class="metric-label">Causal thermal and ephemeris engine</div>
            </div>
            <div class="metric-card">
                <div class="metric-value glow-green">5-Seed Ensemble</div>
                <div class="metric-label">Uncertainty and conservative P90 bound</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        h1, h2, h3 = st.columns(3)
        h1.markdown("**1. Read space weather.** Solar flux (F10.7) and geomagnetic activity (Kp) heat and expand the upper atmosphere.")
        h2.markdown("**2. Predict drag.** Five models vote; their disagreement becomes a conservative P90 estimate.")
        h3.markdown("**3. Plan the burn.** A 7-day optimizer picks when to feather panels and when to reboost.")

        st.info("👈 **To begin:** pick a card above, or set your spacecraft, date and storm parameters in the sidebar and click **🚀 PREDICT DECAY & OPTIMIZE**.")

if __name__ == "__main__":
    main()

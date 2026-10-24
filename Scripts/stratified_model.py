"""
stratified_model.py - Stratified LEO Prediction & Astrodynamics Dispatcher
===========================================================================
Production module providing:
1. StratifiedLEOPredictor: Routes LEO orbits dynamically between Deep LEO (< 550 km)
   and Upper LEO (>= 550 km) specialized gradient boosted models, applying regime-aware
   calibration.
2. Dynamic SGP4 Orbit Propagation: Propagates TLE forward to forecast dates and computes
   authentic future Solar Beta Angles (β) and Sun Exposure Factors (SEF).
3. Payload Classification: Distinguishes passive orbital bodies from active / reboosted
   space stations (ISS, CSS).
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
import numpy as np
import pandas as pd
from datetime import datetime
from xgboost import XGBRegressor
from sgp4.api import Satrec, jday
from physics_engine import calculate_sun_vector

# Active reboosted space stations to flag/handle
ACTIVE_PAYLOAD_IDS = {
    25544: "ISS (Zarya)",
    48274: "CSS (Tianhe-1)"
}

DEFAULT_ALTITUDE_THRESHOLD = 550.0  # km
DEFAULT_MODEL_DIR = get_model_path("Model_2_Stratified_Dual_Regime")


class StratifiedLEOPredictor:
    """
    Unified Predictor supporting Deep LEO (< 550 km), Upper LEO (>= 550 km),
    regime-aware calibration, and 5-fold ensemble uncertainty estimation.
    """
    def __init__(self, model_dir=None):
        self.model_dir = model_dir if model_dir is not None else DEFAULT_MODEL_DIR
        self.deep_model = None
        self.upper_model = None
        self.fallback_model = None
        self.deep_seed_models = []
        self.upper_seed_models = []
        self.ensemble_seeds = [42, 100, 999, 2024, 1337]
        self.calib_config = {}
        self.threshold = DEFAULT_ALTITUDE_THRESHOLD
        self.deep_scale = 1.0
        self.upper_scale = 1.0
        
        self.load()

    def load_models(self):
        """Loads trained regime models (deep, upper, fallback) and calibration config."""
        self.load()

    def load_seed_ensembles(self):
        """Loads the 5-seed ensembles for deep & upper regimes."""
        if not self.deep_seed_models or not self.upper_seed_models:
            self.load()

    def load(self):
        """Load trained regime models, 5-fold seed ensembles, and calibration configuration."""
        candidate_deep = [
            os.path.join(self.model_dir, "xgb_deep_model.json"),
            os.path.join(self.model_dir, "xgb_leo_deep.json")
        ]
        candidate_upper = [
            os.path.join(self.model_dir, "xgb_upper_model.json"),
            os.path.join(self.model_dir, "xgb_leo_upper.json")
        ]
        calib_path = os.path.join(self.model_dir, "calibration_config.json")
        candidate_fallback = [
            get_model_path("Model_1_Single_Global_XGBoost", "xgb_single_global.json"),
            os.path.join(self.model_dir, "xgb_leo_model.json")
        ]
        
        for p in candidate_deep:
            if os.path.exists(p):
                try:
                    self.deep_model = XGBRegressor()
                    self.deep_model.load_model(p)
                    break
                except Exception as e:
                    print(f"Warning: Failed to load deep LEO model ({p}): {e}")
                
        for p in candidate_upper:
            if os.path.exists(p):
                try:
                    self.upper_model = XGBRegressor()
                    self.upper_model.load_model(p)
                    break
                except Exception as e:
                    print(f"Warning: Failed to load upper LEO model ({p}): {e}")
                
        # Load 5-seed ensembles for uncertainty estimation
        self.deep_seed_models = []
        self.upper_seed_models = []
        for s in self.ensemble_seeds:
            dp_s = os.path.join(self.model_dir, f"xgb_deep_seed_{s}.json")
            if os.path.exists(dp_s):
                try:
                    m = XGBRegressor()
                    m.load_model(dp_s)
                    self.deep_seed_models.append(m)
                except Exception as e:
                    print(f"Notice: Failed to load deep seed model {s}: {e}")
                    
            up_s = os.path.join(self.model_dir, f"xgb_upper_seed_{s}.json")
            if os.path.exists(up_s):
                try:
                    m = XGBRegressor()
                    m.load_model(up_s)
                    self.upper_seed_models.append(m)
                except Exception as e:
                    print(f"Notice: Failed to load upper seed model {s}: {e}")
                
        if os.path.exists(calib_path):
            try:
                with open(calib_path, "r", encoding="utf-8") as f:
                    self.calib_config = json.load(f)
                self.threshold = self.calib_config.get("altitude_threshold_km", DEFAULT_ALTITUDE_THRESHOLD)
                self.deep_scale = float(self.calib_config.get("deep_scale", 1.0))
                self.upper_scale = float(self.calib_config.get("upper_scale", 1.0))
            except Exception as e:
                print(f"Warning: Failed to load calibration config: {e}")
                
        if self.deep_model is None or self.upper_model is None:
            for fb_path in candidate_fallback:
                if os.path.exists(fb_path):
                    try:
                        self.fallback_model = XGBRegressor()
                        self.fallback_model.load_model(fb_path)
                        break
                    except Exception as e:
                        print(f"Warning: Failed to load fallback model ({fb_path}): {e}")

    @property
    def is_stratified(self) -> bool:
        return self.deep_model is not None and self.upper_model is not None

    def predict(self, df_features: pd.DataFrame, altitudes: np.ndarray = None, apply_calibration: bool = True) -> np.ndarray:
        """
        Predict decay rates using the appropriate regime model and calibration.
        """
        if altitudes is None:
            if 'altitude_km' in df_features.columns:
                altitudes = df_features['altitude_km'].values
            else:
                altitudes = np.full(len(df_features), 400.0)
        else:
            altitudes = np.asarray(altitudes)
                
        preds = np.zeros(len(df_features))
        
        if self.is_stratified:
            is_deep = altitudes < self.threshold
            
            if np.any(is_deep):
                p_deep = self.deep_model.predict(df_features[is_deep])
                if apply_calibration:
                    p_deep = p_deep * self.deep_scale
                preds[is_deep] = np.maximum(0.01, p_deep)
                
            if np.any(~is_deep):
                p_up = self.upper_model.predict(df_features[~is_deep])
                if apply_calibration:
                    p_up = p_up * self.upper_scale
                preds[~is_deep] = np.maximum(0.01, p_up)
        else:
            model = self.fallback_model if self.fallback_model is not None else self.deep_model
            if model is None:
                raise RuntimeError("No trained model weights found in models/ directory!")
            raw = model.predict(df_features)
            preds = np.maximum(0.01, raw)
            
        return preds

    def predict_with_uncertainty(self, df_features: pd.DataFrame, altitudes: np.ndarray = None, altitude_km: float = None, apply_calibration: bool = True) -> dict:
        """
        Run all 5 seed models (xgb_deep_seed_* or xgb_upper_seed_*) and apply calibration_config.json
        to get mean predicted decay, ensemble standard deviation, and conservative 90th percentile drag.
        
        Returns:
            dict containing:
                - 'decay_mean_m_day': np.ndarray, shape (N,)
                - 'decay_std_m_day':  np.ndarray, shape (N,)
                - 'decay_p90_m_day':  np.ndarray, shape (N,) (mean + 1.28 * std)
                - 'seed_predictions': np.ndarray, shape (num_seeds, N)
                - 'regime': str (Deep LEO or Upper LEO)
        """
        if altitudes is None:
            if altitude_km is not None:
                altitudes = np.full(len(df_features), float(altitude_km))
            elif 'altitude_km' in df_features.columns:
                altitudes = df_features['altitude_km'].values
            else:
                altitudes = np.full(len(df_features), 400.0)
        else:
            altitudes = np.asarray(altitudes)

        n_samples = len(df_features)
        num_seeds = max(len(self.deep_seed_models), len(self.upper_seed_models), 1)
        seed_preds = np.zeros((num_seeds, n_samples))

        is_deep = altitudes < self.threshold

        for s_idx in range(num_seeds):
            row_preds = np.zeros(n_samples)

            # Deep LEO instances
            if np.any(is_deep):
                deep_m = self.deep_seed_models[s_idx] if s_idx < len(self.deep_seed_models) else self.deep_model
                if deep_m is not None:
                    p = deep_m.predict(df_features[is_deep])
                    if apply_calibration:
                        p = p * self.deep_scale
                    row_preds[is_deep] = p

            # Upper LEO instances
            if np.any(~is_deep):
                upper_m = self.upper_seed_models[s_idx] if s_idx < len(self.upper_seed_models) else self.upper_model
                if upper_m is not None:
                    p = upper_m.predict(df_features[~is_deep])
                    if apply_calibration:
                        p = p * self.upper_scale
                    row_preds[~is_deep] = p

            seed_preds[s_idx] = np.maximum(0.01, row_preds)

        decay_mean = np.mean(seed_preds, axis=0)
        decay_std = np.std(seed_preds, axis=0) if num_seeds > 1 else 0.1 * decay_mean
        decay_p90 = decay_mean + 1.28 * decay_std
        regime_label = "Deep LEO (< 550 km)" if np.mean(altitudes) < self.threshold else "Upper LEO (>= 550 km)"

        return {
            'decay_mean_m_day': np.maximum(0.01, decay_mean),
            'decay_std_m_day': np.maximum(0.0, decay_std),
            'decay_p90_m_day': np.maximum(0.01, decay_p90),
            'seed_predictions': seed_preds,
            'regime': regime_label
        }

    def get_status(self) -> dict:
        return {
            "stratified": self.is_stratified,
            "deep_model_loaded": self.deep_model is not None,
            "upper_model_loaded": self.upper_model is not None,
            "threshold_km": self.threshold,
            "deep_scale": self.deep_scale,
            "upper_scale": self.upper_scale
        }


def is_active_payload(norad_id: int) -> bool:
    """Check if satellite is an active, regularly reboosted space station."""
    return int(norad_id) in ACTIVE_PAYLOAD_IDS


def get_satellite_tag(norad_id: int, altitude_km: float = None) -> str:
    """Return a descriptive classification tag for display in the UI."""
    if is_active_payload(norad_id):
        return "⚠️ Active Station (Reboosted)"
    if altitude_km is not None:
        if altitude_km < DEFAULT_ALTITUDE_THRESHOLD:
            return "Deep LEO (Passive)"
        else:
            return "Upper LEO (Passive)"
    return "Passive Orbit"


def propagate_future_geometry(tle_line1: str, tle_line2: str, target_dt: datetime) -> dict:
    """
    Propagate SGP4 orbit forward to target date and compute authentic future
    Solar Beta Angle (β) and Sun Exposure Factor (SEF).
    """
    sat = Satrec.twoline2rv(tle_line1, tle_line2)
    jd, fr = jday(target_dt.year, target_dt.month, target_dt.day,
                  target_dt.hour, target_dt.minute, target_dt.second)
    
    error_code, r, v = sat.sgp4(jd, fr)
    if error_code != 0:
        raise RuntimeError(f"SGP4 propagation error code: {error_code}")
        
    EARTH_RADIUS_KM = 6378.137
    r_mag = np.linalg.norm(r)
    altitude_km = float(r_mag - EARTH_RADIUS_KM)
    
    # Orbit normal vector in ECI: h = r x v
    h = np.cross(r, v)
    h_unit = h / np.linalg.norm(h)
    
    # Sun vector in ECI
    s_x, s_y, s_z = calculate_sun_vector(pd.Series([target_dt]))
    sun_vec = np.array([s_x[0], s_y[0], s_z[0]])
    
    # Beta angle: sin(beta) = sun . orbit_normal
    sin_beta = np.clip(np.dot(sun_vec, h_unit), -1.0, 1.0)
    beta_angle_deg = float(np.degrees(np.arcsin(sin_beta)))
    sun_exposure_factor = float(np.cos(np.radians(beta_angle_deg)))
    
    # SGP4 Mean elements
    m = getattr(sat, 'model', sat)
    inclination_deg = float(np.degrees(m.inclo))
    mu = 398600.4418
    n_rad_s = m.no_kozai / 60.0
    semi_major_axis_km = float((mu / (n_rad_s ** 2)) ** (1.0 / 3.0))
    eccentricity = float(m.ecco)
    perigee_alt_km = float(semi_major_axis_km * (1.0 - eccentricity) - EARTH_RADIUS_KM)
    
    return {
        'altitude_km': altitude_km,
        'semi_major_axis_km': semi_major_axis_km,
        'perigee_alt_km': perigee_alt_km,
        'eccentricity': eccentricity,
        'inclination_deg': inclination_deg,
        'beta_angle_deg': beta_angle_deg,
        'sun_exposure_factor': sun_exposure_factor
    }

"""
step7_maneuver_optimizer.py - 7-Day Model Predictive Control Maneuver & Attitude Optimizer
========================================================================================
Formulates and solves the optimal 7-day thrust and attitude drag-modulation schedule
for satellites in Low Earth Orbit (LEO) during severe space weather events.

Decision Variables:
- u_t_burn in [0, max_burn_m_day]: Altitude reboost via thrusters on day t (m/day -> Delta-v).
- alpha_t in [alpha_min, 1.0]: Attitude drag-modulation factor ("feathering").
  1.0 = nominal payload operations; 0.5 = panels edge-on to cut drag in half.

Coupled to the Stratified 5-Seed XGBoost Ensemble with 90th-percentile conservative drag:
  d_hat_90 = mu_drag + 1.28 * sigma_drag
  h_t = h_{t-1} - (alpha_t * d_hat_90) + u_t_burn

Objective Function:
  Minimizes:
  1. Propellant Delta-v cost, weighted by storm burn inefficiency (Kp >= 5) and solar power penalty
  2. Operational science downtime penalty: w_ops * (1 - alpha_t)^2
  3. Safety altitude floor violation penalty: w_band * max(0, h_floor - h_t)^2
  4. Terminal altitude recovery deficit penalty
"""

import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Astrodynamics & Physics Constants
MU_EARTH = 3.986004418e14  # Earth gravitational parameter (m^3/s^2)
R_EARTH_M = 6378137.0      # Earth equatorial radius (meters)
G0 = 9.80665               # Standard acceleration due to gravity (m/s^2)


class BStarManeuverOptimizer:
    """
    7-Day Model Predictive Control (MPC) Maneuver and Attitude Steering Optimizer.
    """
    def __init__(self, sat_mass_kg: float = 500.0, isp_sec: float = 220.0, max_burn_m_day: float = 500.0):
        """
        Parameters:
            sat_mass_kg: Wet mass of the spacecraft (kg).
            isp_sec: Thruster specific impulse (seconds; ~220s for hydrazine, ~1500s for Hall thruster).
            max_burn_m_day: Maximum allowable semi-major axis reboost per day (meters).
        """
        self.sat_mass_kg = float(sat_mass_kg)
        self.isp_sec = float(isp_sec)
        self.max_burn_m_day = float(max_burn_m_day)

    @staticmethod
    def delta_a_to_delta_v(delta_a_m: np.ndarray, altitude_km: float) -> np.ndarray:
        """
        Converts orbital altitude/semi-major axis reboost (meters) to required Delta-v (m/s)
        using the linearized Gauss variational / Vis-Viva orbital equation:
            Delta-v = (1/2) * n * Delta-a
        where n is the orbital mean motion (rad/s).
        """
        a_m = (float(altitude_km) * 1000.0) + R_EARTH_M
        mean_motion_n = np.sqrt(MU_EARTH / (a_m ** 3))
        return 0.5 * mean_motion_n * np.asarray(delta_a_m, dtype=float)

    def delta_v_to_fuel_kg(self, delta_v_m_s: float) -> float:
        """
        Applies Tsiolkovsky's Rocket Equation to compute propellant mass consumed (kg):
            Delta-m = m_0 * (1 - exp(-Delta-v / (I_sp * g_0)))
        """
        dv = float(delta_v_m_s)
        return self.sat_mass_kg * (1.0 - np.exp(-dv / (self.isp_sec * G0)))

    def optimize_7day_schedule(
        self,
        initial_alt_km: float,
        alt_floor_km: float,
        predicted_decay_p90_m: np.ndarray,  # Shape (H,) from 5-seed ensemble (e.g. 7 days)
        kp_forecast: np.ndarray,            # Shape (H,) predicted Kp_max
        allow_feathering: bool = True,
        sun_exposure_forecast: np.ndarray = None  # Shape (H,) optional SEF for solar power
    ) -> dict:
        """
        Solves for the optimal 7-day thrust (m/day) and attitude feathering schedule.
        
        Returns:
            dict containing:
                - optimal_burns_m_day: np.ndarray, Shape (H,)
                - optimal_feather_alpha: np.ndarray, Shape (H,)
                - optimal_delta_v_m_s: np.ndarray, Shape (H,)
                - total_delta_v_m_s: float
                - total_fuel_kg: float
                - uncontrolled_alt_km: np.ndarray, Shape (H,)
                - naive_alt_km: np.ndarray, Shape (H,)
                - naive_delta_v_m_s: float
                - naive_fuel_kg: float
                - optimal_alt_km: np.ndarray, Shape (H,)
                - fuel_saved_pct: float
                - science_availability_pct: float
                - alt_floor_km: float
        """
        predicted_decay_p90_m = np.asarray(predicted_decay_p90_m, dtype=float)
        kp_forecast = np.asarray(kp_forecast, dtype=float)
        H = len(predicted_decay_p90_m)
        h0_m = float(initial_alt_km) * 1000.0
        h_floor_m = float(alt_floor_km) * 1000.0

        # Decision vector x: [burn_0..burn_{H-1}, alpha_0..alpha_{H-1}]
        x0 = np.concatenate([np.zeros(H), np.ones(H)])
        bounds = [(0.0, self.max_burn_m_day) for _ in range(H)] + [
            (0.4 if allow_feathering else 1.0, 1.0) for _ in range(H)
        ]

        # Storm penalty multiplier: Discourages thruster burns into a swollen thermosphere (Kp >= 5)
        storm_burn_penalty = 1.0 + 0.25 * np.maximum(0.0, kp_forecast - 4.0) ** 2

        # Eclipse / Solar Power penalty: Electric propulsion needs direct solar array illumination
        if sun_exposure_forecast is not None:
            sef = np.asarray(sun_exposure_forecast, dtype=float)
            power_penalty = 1.0 + 0.3 * np.maximum(0.0, 0.7 - sef)
        else:
            power_penalty = 1.0

        def objective(x):
            burns_m = x[:H]
            alphas = x[H:]

            # Roll out altitude trajectory in meters
            net_daily_change_m = burns_m - (alphas * predicted_decay_p90_m)
            alt_trajectory_m = h0_m + np.cumsum(net_daily_change_m)

            # 1. Fuel cost (Delta-v weighted by storm inefficiency and solar power penalty)
            dv_array = self.delta_a_to_delta_v(burns_m, initial_alt_km)
            fuel_cost = np.sum(dv_array * storm_burn_penalty * power_penalty) * 100.0

            # 2. Attitude feathering penalty (science payload downtime)
            feather_cost = np.sum((1.0 - alphas) ** 2) * 15.0

            # 3. Hard safety floor violation penalty
            floor_violations_m = np.maximum(0.0, h_floor_m - alt_trajectory_m)
            safety_cost = np.sum(floor_violations_m ** 2) * 50.0

            # 4. Terminal altitude recovery deficit penalty
            terminal_deficit_m = np.maximum(0.0, h_floor_m + 50.0 - alt_trajectory_m[-1])
            terminal_cost = (terminal_deficit_m ** 2) * 10.0

            return fuel_cost + feather_cost + safety_cost + terminal_cost

        res = minimize(
            objective, 
            x0, 
            method="SLSQP", 
            bounds=bounds, 
            options={'maxiter': 250, 'ftol': 1e-6}
        )

        opt_burns_m = np.round(res.x[:H], 2)
        opt_alphas = np.round(res.x[H:], 3)
        opt_dv = self.delta_a_to_delta_v(opt_burns_m, initial_alt_km)

        # 1. Strategy 1: Uncontrolled Drift
        uncontrolled_alt_m = h0_m - np.cumsum(predicted_decay_p90_m)

        # 2. Strategy 2: Naive Station-Keeping (cancels each day's loss blindly, no feathering)
        naive_burns_m = np.copy(predicted_decay_p90_m)
        naive_dv = self.delta_a_to_delta_v(naive_burns_m, initial_alt_km)
        naive_alt_m = h0_m + np.cumsum(naive_burns_m - predicted_decay_p90_m)
        naive_fuel_kg = float(self.delta_v_to_fuel_kg(np.sum(naive_dv)))

        # 3. Strategy 3: B-Star MPC Optimal Schedule
        optimal_alt_m = h0_m + np.cumsum(opt_burns_m - (opt_alphas * predicted_decay_p90_m))
        opt_fuel_kg = float(self.delta_v_to_fuel_kg(np.sum(opt_dv)))

        fuel_saved_pct = float(max(0.0, (naive_fuel_kg - opt_fuel_kg) / max(naive_fuel_kg, 1e-4) * 100.0))
        science_avail_pct = float(np.mean(opt_alphas) * 100.0)

        return {
            "optimal_burns_m_day": opt_burns_m,
            "optimal_feather_alpha": opt_alphas,
            "optimal_delta_v_m_s": opt_dv,
            "total_delta_v_m_s": float(np.sum(opt_dv)),
            "total_fuel_kg": opt_fuel_kg,
            "uncontrolled_alt_km": uncontrolled_alt_m / 1000.0,
            "naive_alt_km": naive_alt_m / 1000.0,
            "naive_delta_v_m_s": float(np.sum(naive_dv)),
            "naive_fuel_kg": naive_fuel_kg,
            "optimal_alt_km": optimal_alt_m / 1000.0,
            "fuel_saved_pct": fuel_saved_pct,
            "science_availability_pct": science_avail_pct,
            "alt_floor_km": float(alt_floor_km)
        }

    def optimize_7day_horizon(
        self,
        initial_alt_km: float,
        corridor_floor_offset_m: float = 100.0,
        decay_mean_m_day: np.ndarray = None,
        decay_p90_m_day: np.ndarray = None,
        kp_max_forecast: np.ndarray = None,
        beta_angle_deg: np.ndarray = None,
        allow_feathering: bool = True,
        use_p90_robust: bool = True,
        **kwargs
    ) -> dict:
        """
        Solves the 7-day receding horizon MPC problem using 5-seed uncertainty bounds.
        Direct API compatibility method for Streamlit Mission Control integration.
        """
        if decay_p90_m_day is not None:
            decay_p90 = np.asarray(decay_p90_m_day, dtype=float)[:7]
        elif decay_mean_m_day is not None:
            decay_p90 = np.asarray(decay_mean_m_day, dtype=float)[:7] * 1.25
        else:
            decay_p90 = np.full(7, 25.0)

        if decay_mean_m_day is not None:
            decay_mean = np.asarray(decay_mean_m_day, dtype=float)[:7]
        else:
            decay_mean = decay_p90

        design_drag = decay_p90 if use_p90_robust else decay_mean
        H = len(design_drag)

        alt_floor_km = float(initial_alt_km) - (float(corridor_floor_offset_m) / 1000.0)
        kp_forecast = np.asarray(kp_max_forecast if kp_max_forecast is not None else np.full(H, 3.0), dtype=float)[:H]

        # Calculate sun exposure from beta angle if provided
        if beta_angle_deg is not None:
            beta_rad = np.radians(np.asarray(beta_angle_deg, dtype=float)[:H])
            sun_exposure = np.clip(np.cos(beta_rad) * 0.95, 0.4, 1.0)
        else:
            sun_exposure = None

        base_res = self.optimize_7day_schedule(
            initial_alt_km=initial_alt_km,
            alt_floor_km=alt_floor_km,
            predicted_decay_p90_m=design_drag,
            kp_forecast=kp_forecast,
            allow_feathering=allow_feathering,
            sun_exposure_forecast=sun_exposure
        )

        opt_fuel_g = base_res['total_fuel_kg'] * 1000.0
        naive_fuel_g = base_res['naive_fuel_kg'] * 1000.0
        dv_savings_pct = round(base_res['fuel_saved_pct'], 1)

        days_labels = [f"Day +{i+1}" for i in range(H)]

        return {
            "days": days_labels,
            "total_opt_dv_m_s": base_res['total_delta_v_m_s'],
            "dv_savings_pct": dv_savings_pct,
            "total_opt_fuel_g": opt_fuel_g,
            "total_naive_fuel_g": naive_fuel_g,
            "design_drag_m_day": design_drag,
            "optimal_burns_m_day": base_res['optimal_burns_m_day'],
            "optimal_dv_m_s": base_res['optimal_delta_v_m_s'],
            "optimal_feather_alpha": base_res['optimal_feather_alpha'],
            "uncontrolled_alt_km": base_res['uncontrolled_alt_km'],
            "naive_alt_km": base_res['naive_alt_km'],
            "optimal_alt_km": base_res['optimal_alt_km'],
            "alt_floor_km": alt_floor_km,
            "fuel_saved_pct": base_res['fuel_saved_pct'],
            "science_availability_pct": base_res['science_availability_pct']
        }


def run_demo():
    """Runs a simulated severe G4 solar storm 7-day optimization."""
    print("=" * 65)
    print("B-STAR 7-DAY MPC MANEUVER OPTIMIZER DEMONSTRATION")
    print("=" * 65)
    
    # Spacecraft parameters (e.g. Hubble / 500kg science payload)
    opt = BStarManeuverOptimizer(sat_mass_kg=500.0, isp_sec=220.0, max_burn_m_day=500.0)
    
    # 7-day severe storm profile
    h0_km = 470.0
    h_floor_km = 469.75  # Operational altitude floor (250m safety deadband)
    
    days = ["Day 1", "Day 2", "Day 3 (Peak)", "Day 4 (Peak)", "Day 5", "Day 6", "Day 7"]
    kp_sim = np.array([2.3, 4.0, 7.3, 8.0, 5.7, 3.2, 2.1])
    decay_p90 = np.array([25.0, 42.0, 95.0, 110.0, 65.0, 32.0, 22.0])
    sef_sim = np.array([0.85, 0.82, 0.75, 0.70, 0.78, 0.84, 0.86])

    print(f"\nInitial Altitude: {h0_km:.2f} km | Safety Floor: {h_floor_km:.2f} km")
    print(f"Total Cumulative Drag Expected: {np.sum(decay_p90):.1f} meters\n")
    
    res = opt.optimize_7day_schedule(
        initial_alt_km=h0_km,
        alt_floor_km=h_floor_km,
        predicted_decay_p90_m=decay_p90,
        kp_forecast=kp_sim,
        allow_feathering=True,
        sun_exposure_forecast=sef_sim
    )

    print(f"{'Day':<14} | {'Kp':<4} | {'P90 Drag':<10} | {'Opt Burn':<10} | {'Feather Alpha':<14} | {'Opt Alt (km)':<12} | {'Uncontrolled'}")
    print("-" * 88)
    for i in range(len(days)):
        print(f"{days[i]:<14} | {kp_sim[i]:<4.1f} | {decay_p90[i]:<10.1f} | {res['optimal_burns_m_day'][i]:<10.2f} | "
              f"{res['optimal_feather_alpha'][i]:<14.3f} | {res['optimal_alt_km'][i]:<12.3f} | {res['uncontrolled_alt_km'][i]:.3f} km")

    print("\n" + "=" * 65)
    print("STRATEGY PERFORMANCE COMPARISON")
    print("=" * 65)
    print(f"1. Uncontrolled Drift Terminal Altitude: {res['uncontrolled_alt_km'][-1]:.3f} km (VIOLATES FLOOR BY {(h_floor_km - res['uncontrolled_alt_km'][-1])*1000:.1f} m)")
    print(f"2. Naive Daily Stationkeeping:           {res['naive_delta_v_m_s']:.3f} m/s | Propellant: {res['naive_fuel_kg']:.4f} kg")
    print(f"3. B-Star MPC Optimal Schedule:          {res['total_delta_v_m_s']:.3f} m/s | Propellant: {res['total_fuel_kg']:.4f} kg")
    print(f"-> Fuel Saved:                           {res['fuel_saved_pct']:.1f}%")
    print(f"-> Operational Science Availability:     {res['science_availability_pct']:.1f}%")
    print("=" * 65)


if __name__ == "__main__":
    run_demo()

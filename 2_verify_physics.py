"""
2_verify_physics.py - Physics Hypothesis Verification
======================================================
Satellite Decay Prediction System: Proving the Diurnal Bulge Effect

PURPOSE:
Before training, we must PROVE the physics engine is working correctly.
This script validates the correlation between Solar Beta Angle and Drag.

HYPOTHESIS:
For ISS (25544), β cycles 0-75° every ~60 days:
  - β ≈ 0° (Noon Orbit) → Satellite passes through Diurnal Bulge → MAX Drag
  - β ≈ 75° (Twilight) → Satellite in terminator plane → MIN Drag

EXPECTED RESULT:
Negative correlation between sun_exposure_factor and decay_rate_m
(Higher exposure → Higher drag → More negative correlation)

Author: Principal Astrodynamics Engineer
Phase: Physics Validation (Pre-Training)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = "training_set_geometric.csv"
TARGET_SATELLITE = 25544  # ISS (The "Drag King")
OUTPUT_PLOT = "physics_verification_iss.png"


def load_satellite_data(norad_id: int) -> pd.DataFrame:
    """Load and filter data for a specific satellite."""
    print(f"📂 Loading data for NORAD ID {norad_id}...")
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for target satellite
    sat_df = df[df['norad_id'] == norad_id].copy()
    
    # Apply physics filters (clean regime only)
    initial = len(sat_df)
    sat_df = sat_df[
        (sat_df.get('is_circular', True) == True) &
        (sat_df.get('is_maneuver', False) == False) &
        (sat_df['decay_rate_m'] > 0.01) &
        (sat_df['decay_rate_m'] < 500)  # Filter extreme maneuvers
    ].sort_values('date')
    
    print(f"   ├── Raw: {initial} rows")
    print(f"   └── Clean Physics Regime: {len(sat_df)} rows")
    
    return sat_df


def calculate_correlations(df: pd.DataFrame) -> dict:
    """Calculate key physics correlations."""
    correlations = {}
    
    # Beta Angle vs Decay Rate
    if 'beta_angle_deg' in df.columns and 'decay_rate_m' in df.columns:
        valid = df[['beta_angle_deg', 'decay_rate_m']].dropna()
        r, p = stats.pearsonr(valid['beta_angle_deg'], valid['decay_rate_m'])
        correlations['beta_vs_decay'] = {'r': r, 'p': p}
    
    # Sun Exposure Factor vs Decay Rate
    if 'sun_exposure_factor' in df.columns and 'decay_rate_m' in df.columns:
        valid = df[['sun_exposure_factor', 'decay_rate_m']].dropna()
        r, p = stats.pearsonr(valid['sun_exposure_factor'], valid['decay_rate_m'])
        correlations['exposure_vs_decay'] = {'r': r, 'p': p}
    
    # F10.7 vs Decay Rate
    if 'f107_obs' in df.columns and 'decay_rate_m' in df.columns:
        valid = df[['f107_obs', 'decay_rate_m']].dropna()
        r, p = stats.pearsonr(valid['f107_obs'], valid['decay_rate_m'])
        correlations['f107_vs_decay'] = {'r': r, 'p': p}
    
    return correlations


def plot_physics_verification(df: pd.DataFrame, correlations: dict) -> None:
    """Create comprehensive physics verification visualization."""
    print("\n🎨 Generating Physics Verification Plot...")
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f'Physics Verification: ISS (25544) - Sun-Drag Relationship',
        fontsize=14, fontweight='bold', y=0.98
    )
    
    # --- Subplot 1: Time Series (Dual Axis) ---
    ax1 = plt.subplot(2, 2, 1)
    ax1_twin = ax1.twinx()
    
    # Plot Decay Rate (Left Y-axis)
    ln1 = ax1.plot(
        df['date'], df['decay_rate_m'],
        'b-', alpha=0.7, linewidth=0.8, label='Decay Rate (m/day)'
    )
    ax1.set_ylabel('Decay Rate (m/day)', color='blue', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, df['decay_rate_m'].quantile(0.99))
    
    # Plot Sun Exposure Factor (Right Y-axis)
    ln2 = ax1_twin.plot(
        df['date'], df['sun_exposure_factor'],
        'r-', alpha=0.7, linewidth=0.8, label='Sun Exposure Factor'
    )
    ax1_twin.set_ylabel('Sun Exposure Factor (cos β)', color='red', fontsize=11)
    ax1_twin.tick_params(axis='y', labelcolor='red')
    ax1_twin.set_ylim(0, 1.1)
    
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_title('Time Series: Decay Rate vs Sun Exposure', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Combined legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right')
    
    # --- Subplot 2: Beta Angle Distribution ---
    ax2 = plt.subplot(2, 2, 2)
    
    ax2.hist(df['beta_angle_deg'].dropna(), bins=50, color='#4ECDC4', 
             alpha=0.8, edgecolor='black')
    ax2.axvline(df['beta_angle_deg'].median(), color='red', linestyle='--', 
                lw=2, label=f'Median: {df["beta_angle_deg"].median():.1f}°')
    ax2.set_xlabel('Beta Angle (degrees)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('ISS Beta Angle Distribution (Geometric Variance)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # --- Subplot 3: Scatter - Sun Exposure vs Decay ---
    ax3 = plt.subplot(2, 2, 3)
    
    scatter = ax3.scatter(
        df['sun_exposure_factor'], df['decay_rate_m'],
        alpha=0.4, c=df['f107_obs'], cmap='plasma',
        s=20, edgecolors='black', linewidths=0.2
    )
    
    # Add regression line
    valid = df[['sun_exposure_factor', 'decay_rate_m']].dropna()
    z = np.polyfit(valid['sun_exposure_factor'], valid['decay_rate_m'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid['sun_exposure_factor'].min(), valid['sun_exposure_factor'].max(), 100)
    ax3.plot(x_line, p(x_line), 'r--', lw=2, label='Linear Fit')
    
    # Annotation
    corr = correlations.get('exposure_vs_decay', {})
    r_val = corr.get('r', 0)
    p_val = corr.get('p', 1)
    ax3.text(
        0.05, 0.95, 
        f'Pearson r = {r_val:.4f}\np-value = {p_val:.2e}',
        transform=ax3.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
    )
    
    ax3.set_xlabel('Sun Exposure Factor (cos β)', fontsize=11)
    ax3.set_ylabel('Decay Rate (m/day)', fontsize=11)
    ax3.set_title('Hypothesis Test: Sun Exposure → Drag', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='F10.7 (SFU)')
    
    # --- Subplot 4: Scatter - F10.7 vs Decay ---
    ax4 = plt.subplot(2, 2, 4)
    
    scatter2 = ax4.scatter(
        df['f107_obs'], df['decay_rate_m'],
        alpha=0.4, c=df['sun_exposure_factor'], cmap='coolwarm',
        s=20, edgecolors='black', linewidths=0.2
    )
    
    # Add regression line
    valid_f107 = df[['f107_obs', 'decay_rate_m']].dropna()
    z2 = np.polyfit(valid_f107['f107_obs'], valid_f107['decay_rate_m'], 1)
    p2 = np.poly1d(z2)
    x_line2 = np.linspace(valid_f107['f107_obs'].min(), valid_f107['f107_obs'].max(), 100)
    ax4.plot(x_line2, p2(x_line2), 'r--', lw=2, label='Linear Fit')
    
    # Annotation
    corr_f107 = correlations.get('f107_vs_decay', {})
    r_val_f107 = corr_f107.get('r', 0)
    p_val_f107 = corr_f107.get('p', 1)
    ax4.text(
        0.05, 0.95, 
        f'Pearson r = {r_val_f107:.4f}\np-value = {p_val_f107:.2e}',
        transform=ax4.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
    )
    
    ax4.set_xlabel('F10.7 Solar Flux (SFU)', fontsize=11)
    ax4.set_ylabel('Decay Rate (m/day)', fontsize=11)
    ax4.set_title('Baseline: F10.7 → Drag (Expected Positive)', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax4, label='Sun Exposure')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
    print(f"   └── Saved to {OUTPUT_PLOT}")


def main():
    print("=" * 78)
    print("🔬 PHYSICS HYPOTHESIS VERIFICATION")
    print("   Testing: Sun Exposure Factor → Drag Correlation (ISS)")
    print("=" * 78)
    
    # Load ISS data
    df = load_satellite_data(TARGET_SATELLITE)
    
    if len(df) == 0:
        print("❌ No data found for ISS. Run physics engine first.")
        return
    
    # Check required columns
    required = ['beta_angle_deg', 'sun_exposure_factor', 'decay_rate_m', 'f107_obs']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        print("   Run 3_physics_engine.py first to generate geometric features.")
        return
    
    # Calculate correlations
    print("\n📊 Calculating Physics Correlations...")
    correlations = calculate_correlations(df)
    
    # Report
    print("\n" + "=" * 60)
    print("📋 CORRELATION REPORT")
    print("=" * 60)
    
    for name, vals in correlations.items():
        r = vals['r']
        p = vals['p']
        sig = "✓ Significant" if p < 0.05 else "✗ Not Significant"
        direction = "Positive" if r > 0 else "Negative"
        print(f"   {name:25s}: r = {r:+.4f} ({direction}) | p = {p:.2e} | {sig}")
    
    # Interpret physics
    print("\n" + "=" * 60)
    print("🔍 PHYSICS INTERPRETATION")
    print("=" * 60)
    
    exp_corr = correlations.get('exposure_vs_decay', {})
    f107_corr = correlations.get('f107_vs_decay', {})
    
    if exp_corr:
        r_exp = exp_corr['r']
        if r_exp > 0:
            print(f"""
   ✅ SUN EXPOSURE → DECAY: POSITIVE CORRELATION (r = {r_exp:+.4f})
   
   INTERPRETATION:
   Higher Sun Exposure Factor (cos β → 1) means the orbit passes
   through the Diurnal Bulge (noon density spike), causing MORE drag.
   
   This CONFIRMS the Diurnal Bulge physics is working correctly!
   The satellite experiences higher drag when crossing the Sun-facing
   thermospheric expansion zone.
""")
        else:
            print(f"""
   ⚠ SUN EXPOSURE → DECAY: NEGATIVE CORRELATION (r = {r_exp:+.4f})
   
   This is UNEXPECTED. Possible explanations:
   1. ISS reboosts during high-exposure periods
   2. Maneuver detection may have missed some burns
   3. The physics model needs refinement
""")
    
    if f107_corr:
        r_f107 = f107_corr['r']
        if r_f107 > 0:
            print(f"""
   ✅ F10.7 → DECAY: POSITIVE CORRELATION (r = {r_f107:+.4f})
   
   INTERPRETATION:
   Higher solar flux (F10.7) heats the thermosphere, increasing
   density at orbital altitudes, causing MORE drag.
   
   This CONFIRMS the thermospheric physics is working correctly!
""")
    
    # Generate visualization
    plot_physics_verification(df, correlations)
    
    # Summary
    print("\n" + "=" * 78)
    print("✅ PHYSICS VERIFICATION COMPLETE")
    print("=" * 78)
    print(f"""
Summary:
  • Satellite: ISS (25544)
  • Data Points: {len(df):,}
  • Beta Angle Range: {df['beta_angle_deg'].min():.1f}° to {df['beta_angle_deg'].max():.1f}°
  • Sun Exposure Correlation: r = {correlations.get('exposure_vs_decay', {}).get('r', 0):+.4f}
  • F10.7 Correlation: r = {correlations.get('f107_vs_decay', {}).get('r', 0):+.4f}

Output: {OUTPUT_PLOT}

Next Steps:
  1. If correlations are positive → Proceed to retrain on mixed dataset
  2. If correlations are weak → Check for reboosts/maneuvers in ISS data
""")


if __name__ == "__main__":
    main()

"""
Visualize Satellite Orbital Decay vs. Space Weather
==================================================
This script generates a dual-axis plot showing correlation between
Kp Index (space weather) and Satellite Altitude decay.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from etl_pipeline import OrbitalPropagator, GOLDEN_DATASET

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

def generate_decay_plot(norad_id, sat_name):
    print(f"Generating visualization for {sat_name} (ID: {norad_id})...")
    
    # 1. Get Propagated Data using our ETL pipeline logic
    propagator = OrbitalPropagator(db_path="satellite_data.db")
    df = propagator.propagate_with_weather(norad_id)
    
    if df.empty:
        print(f"No data found for {sat_name}. Make sure ETL pipeline has run.")
        return

    # Filter for interesting period if data is too long (optional)
    # df = df.tail(365) # Last year
    
    print(f"Plotting {len(df)} days of data...")

    # 2. Create Dual-Axis Plot
    fig, ax1 = plt.subplots()

    # Plot Altitude (Primary Y-Axis)
    color = 'tab:blue'
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Altitude (km)', color=color, fontsize=12)
    ax1.plot(df['Date'], df['Altitude'], color=color, linewidth=2, label='Altitude')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Plot Kp Index (Secondary Y-Axis)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Kp Index (Geomagnetic Activity)', color=color, fontsize=12)
    
    # Use fill_between for Kp to look like a "storm" density
    # ax2.bar(df['Date'], df['Kp_Index'], color=color, alpha=0.3, width=1.0, label='Kp Index')
    ax2.plot(df['Date'], df['Kp_Index'], color=color, alpha=0.5, linestyle='--', linewidth=1)
    ax2.fill_between(df['Date'], 0, df['Kp_Index'], color=color, alpha=0.1)
    
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 9)  # Kp scale is 0-9

    # Title and Layout
    plt.title(f'Orbital Decay Correlation: {sat_name} vs Space Weather', fontsize=16, pad=20)
    
    # Add correlation text
    correlation = df['Altitude'].diff().corr(df['Kp_Index'])
    stats_text = (
        f"Total Decay: {(df['Altitude'].iloc[0] - df['Altitude'].iloc[-1]):.2f} km\n"
        f"Avg Decay: {df['Daily_Drop'].mean():.4f} km/day\n"
        f"Max Kp: {df['Kp_Index'].max()}"
    )
    plt.figtext(0.15, 0.8, stats_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    fig.tight_layout()
    
    # Save output
    filename = f"decay_plot_{sat_name.lower()}.png"
    plt.savefig(filename)
    print(f"✓ Saved plot to {filename}")
    plt.close()

if __name__ == "__main__":
    # Vanguard 2 is best for showing decay
    # LCS-1 (Mock data) is flat, so Vanguard 2 is the real proof
    
    # Check what we have
    for name, nid in GOLDEN_DATASET.items():
        try:
            generate_decay_plot(nid, name)
        except Exception as e:
            print(f"Failed to plot {name}: {e}")

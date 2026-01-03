import sqlite3
import pandas as pd
import numpy as np
from skyfield.api import EarthSatellite, load, wgs84
import logging

DB_NAME = "satellite_data.db"
TARGET_SATELLITES = [25544, 48274, 20580, 33053, 27386, 25994, 5, 11]

def calculate_orbital_state(row, ts):
    try:
        if not row.get('tle_line1') or not row.get('tle_line2'):
            return pd.Series([np.nan]*6)
        sat = EarthSatellite(row['tle_line1'], row['tle_line2'], str(row['norad_id']), ts)
        epoch = row['epoch']
        t = ts.utc(epoch.year, epoch.month, epoch.day, 0, 0, 0)
        geocentric = sat.at(t)
        altitude_km = wgs84.height_of(geocentric).km
        mean_motion_rad_min = sat.model.no_kozai
        MU = 398600.4418
        mean_motion_rad_sec = mean_motion_rad_min / 60.0
        if mean_motion_rad_sec > 0:
            semi_major_axis_km = (MU / (mean_motion_rad_sec ** 2)) ** (1/3)
        else:
            semi_major_axis_km = np.nan
        perigee_alt_km = (semi_major_axis_km * (1 - sat.model.ecco)) - 6378.137
        is_circular = sat.model.ecco < 0.01
        return pd.Series([altitude_km, semi_major_axis_km, sat.model.ecco, perigee_alt_km, mean_motion_rad_min, is_circular])
    except:
        return pd.Series([np.nan]*6)

def inspect():
    conn = sqlite3.connect(DB_NAME)
    placeholders = ','.join(['?'] * len(TARGET_SATELLITES))
    tle_df = pd.read_sql(f"SELECT * FROM tle_history WHERE norad_id IN ({placeholders})", conn, params=TARGET_SATELLITES)
    tle_df['epoch'] = pd.to_datetime(tle_df['epoch_datetime'])
    ts = load.timescale()
    orbital_cols = ['altitude_km', 'semi_major_axis_km', 'eccentricity', 'perigee_alt_km', 'mean_motion', 'is_circular']
    tle_df[orbital_cols] = tle_df.apply(lambda r: calculate_orbital_state(r, ts), axis=1)
    tle_df = tle_df.dropna(subset=['semi_major_axis_km'])
    tle_df['date_only'] = tle_df['epoch'].dt.date
    tle_df = tle_df.sort_values(['norad_id', 'epoch']).drop_duplicates(subset=['norad_id', 'date_only'], keep='first')
    tle_df['prev_sma'] = tle_df.groupby('norad_id')['semi_major_axis_km'].shift(1)
    tle_df['prev_date'] = tle_df.groupby('norad_id')['date_only'].shift(1)
    tle_df['days_diff'] = (pd.to_datetime(tle_df['date_only']) - pd.to_datetime(tle_df['prev_date'])).dt.days
    clean = tle_df[tle_df['days_diff'] == 1].copy()
    clean['decay_rate_m'] = (clean['prev_sma'] - clean['semi_major_axis_km']) * 1000
    
    # 3-Strike Logic
    cond_anti = clean['decay_rate_m'] <= 0
    cond_stale = clean['decay_rate_m'] < 0.01
    cond_crash = clean['decay_rate_m'] > 2000.0
    
    clean['is_maneuver'] = cond_anti | cond_stale | cond_crash
    clean['strike'] = ""
    clean.loc[cond_anti, 'strike'] += "Anti-Gravity "
    clean.loc[cond_stale, 'strike'] += "Stale/Vacuum "
    clean.loc[cond_crash, 'strike'] += "Crash "
    
    flagged = clean[clean['is_maneuver'] == True].copy()
    print(f"Total flagged maneuvers: {len(flagged)}")
    if not flagged.empty:
        print("\nSignificant flagged rows:")
        print(flagged[['date_only', 'norad_id', 'satellite_name', 'decay_rate_m', 'strike']].head(20).to_string(index=False))

if __name__ == "__main__":
    inspect()

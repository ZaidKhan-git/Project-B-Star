"""
Fetch F10.7 Solar Flux Data for Golden Epoch (2014-2026)
=========================================================
This script populates the solar_flux table with F10.7 observations.
"""
import sys
sys.path.insert(0, '.')

from etl_pipeline import SpaceWeatherIngestor

DB_PATH = "satellite_data.db"

print("=" * 70)
print("FETCHING F10.7 SOLAR FLUX DATA (2014-2026)")
print("=" * 70)

sw_ingestor = SpaceWeatherIngestor(db_path=DB_PATH)
result = sw_ingestor.fetch_historical_kp_and_f107(start_year=2014, end_year=2026)

print("")
print("=" * 70)
print("INGESTION COMPLETE")
print(f"Kp records: {result['kp_records']:,}")
print(f"F10.7 records: {result['f107_records']:,}")
print("=" * 70)


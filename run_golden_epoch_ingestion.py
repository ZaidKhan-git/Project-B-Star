"""
Smart Golden Epoch TLE Ingestion
Only fetches 2014-2024 (data before existing 2025-2026 cache)
"""
import sqlite3
import sys
sys.path.insert(0, '.')

from etl_pipeline import SpaceTrackIngestor, GOLDEN_DATASET

DB_PATH = "satellite_data.db"
SPACETRACK_USER = "mohdzaidk25@gmail.com"
SPACETRACK_PASS = "ThisisspacetrackpassworD"

print("=" * 60)
print("SMART GOLDEN EPOCH TLE INGESTION")
print("Fetching ONLY 2014-2024 (2025-2026 already cached)")
print("=" * 60)

# Check existing TLE coverage
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT MIN(epoch_datetime), MAX(epoch_datetime) FROM tle_history")
existing_range = cursor.fetchone()
print(f"\nExisting TLE Coverage: {existing_range[0]} to {existing_range[1]}")
conn.close()

tle_ingestor = SpaceTrackIngestor(
    username=SPACETRACK_USER,
    password=SPACETRACK_PASS,
    db_path=DB_PATH
)

# Only fetch 2014-2024 (years BEFORE existing data)
START_YEAR = 2014
END_YEAR = 2024  # Stop before 2025 (already cached)

total_all = 0
for name, norad_id in GOLDEN_DATASET.items():
    print(f"\n📡 {name} (NORAD ID: {norad_id})")
    try:
        # Use fetch_tle_golden_epoch but with end_year=2024
        total = tle_ingestor.fetch_tle_golden_epoch(
            norad_id=norad_id,
            start_year=START_YEAR,
            end_year=END_YEAR,
            force_refresh=True  # Force for historical years only
        )
        total_all += total
    except Exception as e:
        print(f"    ✗ Failed: {e}")

print(f"\n{'='*60}")
print(f"TOTAL NEW TLEs FETCHED: {total_all}")
print(f"{'='*60}")

# Verify final coverage
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*), MIN(epoch_datetime), MAX(epoch_datetime) FROM tle_history")
final_stats = cursor.fetchone()
print(f"\nFinal TLE Coverage: {final_stats[0]} records")
print(f"Date Range: {final_stats[1]} to {final_stats[2]}")
conn.close()

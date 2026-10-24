"""
1_ingest_data.py - Fetch Golden Epoch Data
==========================================
Fetches satellite data for the extended Golden Dataset (including SL-16 R/B).
Uses strict caching to avoid Space-Track bans.
"""

from etl_pipeline import SpaceTrackIngestor, GOLDEN_DATASET
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def main():
    print("=" * 60)
    print("🛰️ GOLDEN EPOCH DATA INGESTION")
    print("=" * 60)
    
    # Initialize ingestor with dummy creds (it uses hardcoded ones internally)
    ingestor = SpaceTrackIngestor("dummy", "dummy")
    
    start_date = "2014-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Target Period: {start_date} to {end_date}")
    print(f"Satellites: {len(GOLDEN_DATASET)}")
    
    for name, norad_id in GOLDEN_DATASET.items():
        print(f"\nProcessing {name} (ID: {norad_id})...")
        try:
            # fetch_history handles caching automatically
            df = ingestor.fetch_history(
                norad_id=norad_id,
                start_date=start_date,
                end_date=end_date
            )
            print(f"   ✅ Available records: {len(df):,}")
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

    print("\n" + "=" * 60)
    print("✅ INGESTION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()

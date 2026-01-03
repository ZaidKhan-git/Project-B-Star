"""
Feature Engineering Module for Space Weather Data
=================================================
This module implements lag-based feature generation for satellite drag prediction.

Physics Rationale:
- Atmospheric density (ρ) does not increase instantly when Kp rises
- Thermodynamic lag: 3-6 hours for atmosphere to heat and expand
- Cumulative effect: Prolonged storms cause sustained drag (24-72 hours)

Author: Zaid Khan
Date: 2026-01-03
Reference: SKILLS.md - Vectorization, Type Hinting, Logging Standards
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
import logging
from datetime import datetime, timezone

# Configure logging per SKILLS.md standards
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpaceWeatherFeatureEngineer:
    """
    Transforms raw NOAA space weather data into ML-ready feature set.
    
    Implements lag features to capture the thermodynamic delay between
    solar storms (high Kp) and atmospheric drag increase.
    
    Key Features Generated:
    - Kp_Current: Instantaneous geomagnetic activity
    - Kp_Lag_3h: 3-hour lag (density peak)
    - Kp_Lag_6h: 6-hour lag (sustained heat)
    - Kp_Lag_24h: 24-hour lag (cumulative effect)
    - Kp_Roll_24h_Mean: 24-hour rolling average
    - Kp_Roll_72h_Mean: 72-hour rolling average
    
    Attributes:
        db_path: Path to SQLite database containing space_weather table
        target_frequency: Resampling frequency (default: '1H' for hourly)
    """
    
    def __init__(self, db_path: str = "satellite_data.db", target_frequency: str = "1H"):
        """
        Initialize the Feature Engineer.
        
        Args:
            db_path: Path to SQLite database
            target_frequency: Pandas frequency string for resampling (default: '1H')
        """
        self.db_path = db_path
        self.target_frequency = target_frequency
        logger.info(f"Initialized SpaceWeatherFeatureEngineer with freq={target_frequency}")
    
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw space weather data from SQLite database.
        
        Returns:
            DataFrame with columns: time_tag, kp
            
        Raises:
            sqlite3.Error: If database read fails
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT time_tag, kp 
                    FROM space_weather 
                    ORDER BY time_tag ASC
                """
                df = pd.read_sql_query(query, conn)
                
            # Convert time_tag to datetime with UTC timezone
            df['time_tag'] = pd.to_datetime(df['time_tag'], utc=True)
            
            logger.info(f"Loaded {len(df)} raw space weather records")
            logger.info(f"Time range: {df['time_tag'].min()} to {df['time_tag'].max()}")
            
            return df
            
        except sqlite3.Error as e:
            logger.error(f"Database read failed: {e}")
            raise
    
    def resample_to_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample data to hourly frequency and handle missing values.
        
        Strategy:
        - Linear interpolation for small gaps (<6 hours)
        - Forward-fill for larger gaps (prevents look-ahead bias)
        - NEVER backward-fill (would use future information)
        
        Args:
            df: Raw dataframe with time_tag and kp columns
            
        Returns:
            Resampled dataframe with hourly frequency
        """
        # Set time_tag as index for resampling
        df = df.set_index('time_tag')
        
        # Resample to hourly frequency (mean aggregation for any duplicates)
        df_hourly = df.resample(self.target_frequency).mean()
        
        # Handle missing values (CRITICAL: No backward-fill to prevent look-ahead bias)
        # Strategy: Interpolate small gaps, forward-fill larger ones
        df_hourly['kp'] = df_hourly['kp'].interpolate(
            method='linear',
            limit=6,  # Only interpolate gaps up to 6 hours
            limit_direction='forward'
        ).ffill()  # Forward-fill any remaining gaps
        
        # Reset index to make time_tag a column again
        df_hourly = df_hourly.reset_index()
        
        logger.info(f"Resampled to {len(df_hourly)} hourly records")
        logger.info(f"Missing values after processing: {df_hourly['kp'].isna().sum()}")
        
        return df_hourly
    
    def generate_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate lag-based features using vectorized pandas operations.
        
        Physics Mapping:
        - Lag 3h: Captures immediate ionosphere response
        - Lag 6h: Captures thermosphere heating peak
        - Lag 24h: Captures cumulative daily effect
        - Roll 24h: Sustained heating over 1 day
        - Roll 72h: Long-term cumulative stress
        
        Args:
            df: Hourly-resampled dataframe with time_tag and kp columns
            
        Returns:
            DataFrame with original columns plus lag features
            
        Note:
            All operations are VECTORIZED (no Python loops) per SKILLS.md
        """
        # Create a copy to avoid modifying original
        df_features = df.copy()
        
        # Rename base column for clarity
        df_features['Kp_Current'] = df_features['kp']
        
        # VECTORIZED LAG GENERATION (pandas shift operation is C-optimized)
        df_features['Kp_Lag_3h'] = df_features['Kp_Current'].shift(3)
        df_features['Kp_Lag_6h'] = df_features['Kp_Current'].shift(6)
        df_features['Kp_Lag_24h'] = df_features['Kp_Current'].shift(24)
        
        # VECTORIZED ROLLING STATISTICS (pandas rolling window is C-optimized)
        df_features['Kp_Roll_24h_Mean'] = df_features['Kp_Current'].rolling(
            window=24,
            min_periods=12  # Require at least 50% data in window
        ).mean()
        
        df_features['Kp_Roll_72h_Mean'] = df_features['Kp_Current'].rolling(
            window=72,
            min_periods=36  # Require at least 50% data in window
        ).mean()
        
        # Drop the intermediate 'kp' column (we have Kp_Current)
        df_features = df_features.drop(columns=['kp'])
        
        logger.info(f"Generated {df_features.shape[1] - 1} feature columns")
        logger.info(f"Feature columns: {list(df_features.columns)}")
        
        return df_features
    
    def get_feature_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate summary statistics for quality assurance.
        
        Args:
            df: DataFrame with lag features
            
        Returns:
            DataFrame with descriptive statistics per feature
        """
        stats = df.describe().T
        stats['missing_count'] = df.isna().sum()
        stats['missing_pct'] = (df.isna().sum() / len(df) * 100).round(2)
        
        logger.info("Feature Statistics:")
        logger.info(f"\n{stats.to_string()}")
        
        return stats
    
    def process_full_pipeline(
        self, 
        drop_incomplete: bool = True,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Execute the complete feature engineering pipeline.
        
        Pipeline Steps:
        1. Load raw data from SQLite
        2. Resample to hourly frequency
        3. Generate lag features
        4. Optionally drop rows with NaN values
        5. Optionally save to CSV
        
        Args:
            drop_incomplete: If True, drop rows with any NaN values
            output_path: If provided, save result to this CSV path
            
        Returns:
            ML-ready DataFrame with engineered features
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Step 1: Load raw data
        df_raw = self.load_raw_data()
        
        # Step 2: Resample to hourly
        df_hourly = self.resample_to_hourly(df_raw)
        
        # Step 3: Generate lag features
        df_features = self.generate_lag_features(df_hourly)
        
        # Step 4: Quality check
        self.get_feature_statistics(df_features)
        
        # Step 5: Drop incomplete rows if requested
        if drop_incomplete:
            rows_before = len(df_features)
            df_features = df_features.dropna()
            rows_after = len(df_features)
            logger.info(f"Dropped {rows_before - rows_after} incomplete rows")
        
        # Step 6: Save to CSV if requested
        if output_path:
            df_features.to_csv(output_path, index=False)
            logger.info(f"Saved engineered features to {output_path}")
        
        logger.info(f"Pipeline complete. Final dataset: {df_features.shape}")
        
        return df_features
    
    def save_to_database(self, df: pd.DataFrame, table_name: str = "space_weather_features") -> None:
        """
        Save engineered features back to SQLite database.
        
        Args:
            df: DataFrame with engineered features
            table_name: Name of table to create/replace
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                df.to_sql(
                    table_name,
                    conn,
                    if_exists='replace',
                    index=False
                )
            logger.info(f"Saved {len(df)} rows to {table_name} table")
            
        except sqlite3.Error as e:
            logger.error(f"Database write failed: {e}")
            raise


# =============================================================================
# MAIN EXECUTION (Demo)
# =============================================================================
def main():
    """
    Demonstration of the SpaceWeatherFeatureEngineer pipeline.
    """
    print("=" * 60)
    print("SPACE WEATHER FEATURE ENGINEERING - DAY 2")
    print("=" * 60)
    
    # Initialize engineer
    engineer = SpaceWeatherFeatureEngineer(
        db_path="satellite_data.db",
        target_frequency="1H"
    )
    
    # Execute pipeline
    try:
        df_features = engineer.process_full_pipeline(
            drop_incomplete=True,
            output_path="space_weather_features.csv"
        )
        
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING COMPLETE")
        print("=" * 60)
        print(f"Output Shape: {df_features.shape}")
        print(f"Time Range: {df_features['time_tag'].min()} to {df_features['time_tag'].max()}")
        print("\nSample Output:")
        print(df_features.head(10).to_string(index=False))
        
        # Save to database for easy access
        engineer.save_to_database(df_features)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()

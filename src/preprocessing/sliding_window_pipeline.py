from pathlib import Path
from datetime import datetime, timedelta
import argparse

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, TimestampType
from src.utils.spark_session import make_spark


class SlidingWindowPreparer:
    """Prepares session-based recommendation data with sliding window protocol using Spark"""
    
    def __init__(self, 
                 input_path: str,
                 output_path: str,
                 start_date: str,
                 n_days: int = 30,
                 min_session_len: int = 2,
                 max_session_len: int = 50,
                 min_item_support: int = 5):
        """
        Args:
            input_path: Path to enriched_events parquet data
            output_path: Where to save processed slices
            start_date: First date to include (YYYY-MM-DD)
            n_days: Total days to use (default: 30)
            min_session_len: Min interactions per session (default: 2)
            max_session_len: Max interactions per session (default: 50)
            min_item_support: Min occurrences for item to be kept (default: 5)
        """
        self.spark = make_spark(memory_storage_fraction=0.3)
        self.input_path = input_path
        self.output_path = output_path
        self.start_date = start_date
        self.n_days = n_days
        self.min_session_len = min_session_len
        self.max_session_len = max_session_len
        self.min_item_support = min_item_support
        
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
    def log(self, message: str):
        """Print timestamped log message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def load_raw_data(self) -> DataFrame:
        """Load raw events from parquet partitions"""
        self.log(f"Loading {self.n_days} days starting from {self.start_date}")
        
        start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
        end_dt = start_dt + timedelta(days=self.n_days - 1)
        
        # Load parquet with partition filter
        df = self.spark.read.parquet(self.input_path)
        
        # Filter by date range
        df = df.filter(
            (F.col('dt') >= start_dt.strftime('%Y-%m-%d')) &
            (F.col('dt') <= end_dt.strftime('%Y-%m-%d'))
        )
        
        total_events = df.count()
        self.log(f"Loaded {total_events:,} events")
        
        return df
    
    def filter_interaction_events(self, df: DataFrame) -> DataFrame:
        """Keep only real user-item interactions (exclude RankingRendered)"""
        self.log("Filtering interaction events (excluding RankingRendered)")
        
        # Keep events that represent actual user interest
        interaction_types = [
            'ListingRendered',      # User viewed listing detail
            'GalleryClicked',       # User clicked on gallery/image
            'RankingClicked',       # User clicked item in ranking
            'LeadPanelClicked',     # User clicked contact panel
            'LeadClicked',          # User initiated contact
            'FavoriteClicked',      # User favorited item
            'ShareClicked',         # User shared item
        ]
        
        df_filtered = df.filter(F.col('event_type').isin(interaction_types))
        
        total_before = df.count()
        total_after = df_filtered.count()
        self.log(f"Kept {total_after:,} interaction events ({total_after/total_before*100:.2f}%)")
        
        return df_filtered
    
    def create_sessions(self, df: DataFrame) -> DataFrame:
        """Create session-based interaction sequences"""
        self.log("Creating session sequences")
        
        # Select and rename columns for RecBole compatibility
        sessions = df.select(
            F.col('session_id').alias('session_id'),
            F.col('unique_user_id').alias('user_id'),
            F.col('listing_id').cast(IntegerType()).alias('item_id'),
            F.col('event_ts').alias('timestamp'),
            F.col('event_type').alias('event_type')
        )
        
        # Add row number within each session (ordered by timestamp)
        window_spec = Window.partitionBy('session_id').orderBy('timestamp')
        sessions = sessions.withColumn('position', F.row_number().over(window_spec))
        
        unique_sessions = sessions.select('session_id').distinct().count()
        self.log(f"Created {unique_sessions:,} unique sessions")
        
        return sessions
    
    def filter_sessions(self, df: DataFrame) -> DataFrame:
        """Filter sessions by length (2-50 interactions)"""
        self.log(f"Filtering sessions: {self.min_session_len}-{self.max_session_len} interactions")
        
        # Count session sizes
        session_sizes = df.groupBy('session_id').agg(
            F.count('*').alias('session_size')
        )
        
        # Filter valid sessions
        valid_sessions = session_sizes.filter(
            (F.col('session_size') >= self.min_session_len) &
            (F.col('session_size') <= self.max_session_len)
        )
        
        # Join back to keep only valid sessions
        df_filtered = df.join(
            valid_sessions.select('session_id'),
            on='session_id',
            how='inner'
        )
        
        sessions_before = session_sizes.count()
        sessions_after = valid_sessions.count()
        events_before = df.count()
        events_after = df_filtered.count()
        
        self.log(f"Sessions: {sessions_before:,} → {sessions_after:,}")
        self.log(f"Events: {events_before:,} → {events_after:,}")
        
        return df_filtered
    
    def filter_rare_items(self, df: DataFrame) -> DataFrame:
        """Remove items with fewer than min_item_support occurrences"""
        self.log(f"Filtering items with <{self.min_item_support} occurrences")
        
        # Count item occurrences
        item_counts = df.groupBy('item_id').agg(
            F.count('*').alias('item_count')
        )
        
        # Filter valid items
        valid_items = item_counts.filter(
            F.col('item_count') >= self.min_item_support
        )
        
        # Join back to keep only valid items
        df_filtered = df.join(
            valid_items.select('item_id'),
            on='item_id',
            how='inner'
        )
        
        items_before = item_counts.count()
        items_after = valid_items.count()
        events_before = df.count()
        events_after = df_filtered.count()
        
        self.log(f"Items: {items_before:,} → {items_after:,}")
        self.log(f"Events: {events_before:,} → {events_after:,}")
        
        return df_filtered
    
    def create_sliding_window_slices(self, df: DataFrame) -> list:
        """
        Create 5 slices with sliding window protocol:
        Slice 1: Days 1-5 (train) + Day 6 (test)
        Slice 2: Days 7-11 (train) + Day 12 (test)
        ...
        Slice 5: Days 25-29 (train) + Day 30 (test)
        """
        self.log("Creating 5 sliding window slices (5 days train + 1 day test)")
        
        slices = []
        days_per_slice = 6  # 5 train + 1 test
        n_slices = self.n_days // days_per_slice
        
        start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
        
        for slice_idx in range(n_slices):
            slice_start = start_dt + timedelta(days=slice_idx * days_per_slice)
            train_start = slice_start
            train_end = slice_start + timedelta(days=4, hours=23, minutes=59, seconds=59)
            test_start = slice_start + timedelta(days=5)
            test_end = slice_start + timedelta(days=5, hours=23, minutes=59, seconds=59)
            
            # Filter train data
            train_df = df.filter(
                (F.col('timestamp') >= train_start) &
                (F.col('timestamp') <= train_end)
            ).cache()
            
            # Filter test data
            test_df = df.filter(
                (F.col('timestamp') >= test_start) &
                (F.col('timestamp') <= test_end)
            ).cache()
            
            train_count = train_df.count()
            train_sessions_count = train_df.select('session_id').distinct().count()
            test_count = test_df.count()
            test_sessions_count = test_df.select('session_id').distinct().count()
            
            self.log(f"\nSlice {slice_idx + 1}:")
            self.log(f"  Train: {train_start.date()} to {train_end.date()} | {train_count:,} events | {train_sessions_count:,} sessions")
            self.log(f"  Test:  {test_start.date()} to {test_end.date()} | {test_count:,} events | {test_sessions_count:,} sessions")
            
            slices.append({
                'slice_id': slice_idx + 1,
                'train': train_df,
                'test': test_df,
                'train_period': f"{train_start.date()}_{train_end.date()}",
                'test_period': f"{test_start.date()}_{test_end.date()}"
            })
        
        return slices
    
    def save_slices(self, slices: list):
        """Save each slice as separate train/test parquet files"""
        self.log(f"\nSaving slices to {self.output_path}")
        
        stats = []
        
        for slice_data in slices:
            slice_id = slice_data['slice_id']
            slice_dir = f"{self.output_path}/slice_{slice_id}"
            
            train_df = slice_data['train']
            test_df = slice_data['test']
            
            # Save train (coalesce to reduce number of files)
            train_df.coalesce(10).write.mode('overwrite').parquet(f"{slice_dir}/train")
            
            # Save test
            test_df.coalesce(5).write.mode('overwrite').parquet(f"{slice_dir}/test")
            
            # Collect metadata
            metadata = {
                'slice_id': slice_id,
                'train_period': slice_data['train_period'],
                'test_period': slice_data['test_period'],
                'train_events': train_df.count(),
                'test_events': test_df.count(),
                'train_sessions': train_df.select('session_id').distinct().count(),
                'test_sessions': test_df.select('session_id').distinct().count(),
                'train_users': train_df.select('user_id').distinct().count(),
                'test_users': test_df.select('user_id').distinct().count(),
                'train_items': train_df.select('item_id').distinct().count(),
                'test_items': test_df.select('item_id').distinct().count(),
            }
            
            # Save metadata as CSV
            import pandas as pd
            pd.DataFrame([metadata]).to_csv(f"{slice_dir}/metadata.csv", index=False)
            stats.append(metadata)
            
            self.log(f"  Saved slice {slice_id} to {slice_dir}")
            
            # Unpersist cached DataFrames
            train_df.unpersist()
            test_df.unpersist()
        
        # Save overall summary
        import pandas as pd
        summary_df = pd.DataFrame(stats)
        summary_df.to_csv(f"{self.output_path}/summary.csv", index=False)
        self.log(f"\nSummary saved to {self.output_path}/summary.csv")
        
        return summary_df
    
    def run(self):
        """Execute full pipeline"""
        self.log("="*80)
        self.log("SLIDING WINDOW DATA PREPARATION PIPELINE (PySpark)")
        self.log("="*80)
        
        # 1. Load raw data
        df = self.load_raw_data()
        
        # 2. Filter interaction events
        df = self.filter_interaction_events(df)
        
        # 3. Create sessions
        df = self.create_sessions(df)
        
        # 4. Filter by session length
        df = self.filter_sessions(df)
        
        # 5. Filter rare items
        df = self.filter_rare_items(df)
        
        # 6. Re-filter sessions after item filtering
        df = self.filter_sessions(df)
        
        # Cache for sliding window creation
        df = df.cache()
        df.count()  # Materialize cache
        
        # 7. Create sliding window slices
        slices = self.create_sliding_window_slices(df)
        
        # 8. Save slices
        summary = self.save_slices(slices)
        
        # Cleanup
        df.unpersist()
        
        self.log("\n" + "="*80)
        self.log("PIPELINE COMPLETED SUCCESSFULLY")
        self.log("="*80)
        print("\nFINAL SUMMARY:")
        print(summary.to_string(index=False))
        
        self.spark.stop()
        return summary


def main():
    parser = argparse.ArgumentParser(description='Prepare sliding window data for session-based RecSys')
    parser.add_argument('--input', type=str, 
                       default='/home/hygo2025/Documents/data/processed_data/enriched_events',
                       help='Path to enriched events')
    parser.add_argument('--output', type=str,
                       default='data/sliding_window',
                       help='Output directory for slices')
    parser.add_argument('--start-date', type=str,
                       default='2024-03-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--n-days', type=int, default=30,
                       help='Total days to process')
    parser.add_argument('--min-session-len', type=int, default=2,
                       help='Minimum session length')
    parser.add_argument('--max-session-len', type=int, default=50,
                       help='Maximum session length')
    parser.add_argument('--min-item-support', type=int, default=5,
                       help='Minimum item occurrences')
    
    args = parser.parse_args()
    
    preparer = SlidingWindowPreparer(
        input_path=args.input,
        output_path=args.output,
        start_date=args.start_date,
        n_days=args.n_days,
        min_session_len=args.min_session_len,
        max_session_len=args.max_session_len,
        min_item_support=args.min_item_support
    )
    
    preparer.run()


if __name__ == '__main__':
    main()

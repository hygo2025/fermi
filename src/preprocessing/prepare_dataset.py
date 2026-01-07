import argparse
import os
from datetime import datetime, timedelta

from pyspark.sql import functions as F
from pyspark.sql.types import LongType

from src.utils import make_spark
from src.utils import log

def load_date_range_spark(spark, events_path: str, start_date: str, end_date: str):
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    log(f"\nLoading data from {start_date} to {end_date} using Spark...")
    
    # Build partition filter
    partitions = []
    current = start
    while current <= end:
        partition = f"dt={current.strftime('%Y-%m-%d')}"
        partition_path = os.path.join(events_path, partition)
        if os.path.exists(partition_path):
            partitions.append(partition)
            log(f"  Found partition: {partition}")
        current += timedelta(days=1)
    
    if not partitions:
        raise ValueError(f"No data found for date range {start_date} to {end_date}")
    
    # Load all partitions at once (Spark handles this efficiently)
    df = spark.read.parquet(events_path)
    
    # Filter by date range
    df = df.filter(
        (F.col('dt') >= start_date) & 
        (F.col('dt') <= end_date) &
        (F.col('event_type') != 'RankingRendered') &
        (F.col('business_type') != 'SALE')
    )
    
    log(f"  Loaded data with Spark")
    
    return df


def prepare_session_rec_format_spark(df):
    required_cols = ['session_id', 'listing_id', 'event_ts']
    
    # Check required columns
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {df.columns}")
    
    # Select and rename columns
    df_prepared = df.select(
        F.col('session_id').alias('SessionId'),
        F.col('listing_id').alias('ItemId'),
        F.col('event_ts').alias('Time')
    )
    
    # Convert timestamp to Unix timestamp (seconds)
    df_prepared = df_prepared.withColumn(
        'Time',
        F.unix_timestamp(F.col('Time')).cast(LongType())
    )
    
    # Sort by session and time
    df_prepared = df_prepared.orderBy('SessionId', 'Time')
    
    return df_prepared


def save_session_rec_data_spark(df, output_path: str, name: str):
    """
    Save data in session-rec format using Spark.
    
    Saves as Parquet (single file) for fast loading.
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Save as single Parquet file (coalesce to 1 partition)
    parquet_file = os.path.join(output_path, f"{name}.parquet")
    
    log(f"  Saving Parquet: {parquet_file}")
    df.coalesce(1).write.mode('overwrite').parquet(parquet_file + '.tmp')
    
    # Move the actual parquet file to desired location
    import glob
    import shutil
    tmp_files = glob.glob(f"{parquet_file}.tmp/*.parquet")
    if tmp_files:
        shutil.move(tmp_files[0], parquet_file)
        shutil.rmtree(f"{parquet_file}.tmp")
    
    log(f"   Saved Parquet: {parquet_file}")
    
    # Compute and log stats
    n_events = df.count()
    n_sessions = df.select('SessionId').distinct().count()
    n_items = df.select('ItemId').distinct().count()
    
    time_stats = df.agg(
        F.min('Time').alias('min_time'),
        F.max('Time').alias('max_time')
    ).collect()[0]
    
    date_min = datetime.fromtimestamp(time_stats['min_time']).strftime('%Y-%m-%d')
    date_max = datetime.fromtimestamp(time_stats['max_time']).strftime('%Y-%m-%d')
    
    log(f"""
    Stats for {name}:
      Events: {n_events:,}
      Sessions: {n_sessions:,}
      Items: {n_items:,}
      Time range: {date_min} / {date_max}
    """)


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for session-rec benchmark using Spark')
    parser.add_argument('--source-path', type=str, 
                       default='/home/hygo2025/Documents/data/processed_data/events',
                       help='Path to source Parquet data')
    parser.add_argument('--output-path', type=str,
                       default='./session_rec_format',
                       help='Output path for processed data')
    parser.add_argument('--start-date', type=str, required=True,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--test-days', type=int, default=2,
                       help='Number of days for test set (default: 2)')
    parser.add_argument('--name', type=str, default='realestate',
                       help='Dataset name prefix')
    
    args = parser.parse_args()
    
    log("="*80)
    log("Session-Rec Dataset Preparation (Spark-optimized)")
    log("="*80)
    
    # Initialize Spark
    log("\nInitializing Spark session...")
    spark = make_spark(memory_storage_fraction=0.3)
    
    try:
        # Load full date range with Spark
        df = load_date_range_spark(spark, args.source_path, args.start_date, args.end_date)
        
        total_events = df.count()
        log(f"\n Loaded {total_events:,} total events")
        
        # Prepare session-rec format
        log("\nConverting to session-rec format with Spark...")
        df_prepared = prepare_session_rec_format_spark(df)
        
        # Cache for reuse
        df_prepared.cache()
        
        # Split train/test by date
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        test_start_date = end_date - timedelta(days=args.test_days - 1)
        test_start_ts = int(test_start_date.timestamp())
        
        log(f"\nSplitting data: Last {args.test_days} days for test")
        log(f"  Train: {args.start_date} to {(test_start_date - timedelta(days=1)).strftime('%Y-%m-%d')}")
        log(f"  Test: {test_start_date.strftime('%Y-%m-%d')} to {args.end_date}")
        
        train_df = df_prepared.filter(F.col('Time') < test_start_ts)
        test_df = df_prepared.filter(F.col('Time') >= test_start_ts)
        
        # Save datasets
        output_path = os.path.join(args.output_path, args.name)
        
        log("\nSaving train set...")
        save_session_rec_data_spark(train_df, output_path, f"{args.name}_train_full")
        
        log("\nSaving test set...")
        save_session_rec_data_spark(test_df, output_path, f"{args.name}_test")
        
        log("\n" + "="*80)
        log(" Dataset preparation complete!")
        log("="*80)
        log(f"\nOutput location: {output_path}")
        log(f"\nFormats generated:")
        log(f"  • Parquet files (.parquet) - Fast loading with Spark optimization")
        log(f"\nNext step: Run benchmark with config pointing to '{args.name}' dataset")
        
    finally:
        # Clean up Spark
        spark.stop()


if __name__ == '__main__':
    main()

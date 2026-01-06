import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.spark_session import make_spark
from src.data_preparation.pipelines.listings_pipeline import run_listings_pipeline
from src.data_preparation.pipelines.events_pipeline import run_events_pipeline
from src.data_preparation.pipelines.merge_events import run_merge_events_pipeline


def main():
    print("\n" + "="*60)
    print(" RAW DATA PREPARATION - CLASSIFIED ADS")
    print("="*60 + "\n")
    
    spark = make_spark()
    
    try:
        print("Step 1/3: Processing listings...")
        run_listings_pipeline(spark=spark)
        
        print("\nStep 2/3: Processing user events...")
        run_events_pipeline(spark=spark)
        
        print("\nStep 3/3: Merging data...")
        #run_merge_events_pipeline(spark=spark)
        
        print("\n" + "="*60)
        print(" PREPARATION COMPLETE")
        print("="*60)
        print("\nNext steps:")
        print("  1. make prepare-data      # Create sliding window")
        print("  2. make convert-recbole   # Convert to RecBole format")
        print("  3. make run-all           # Run experiments\n")
        
    except Exception as e:
        print(f"\nERRO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

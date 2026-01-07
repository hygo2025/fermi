import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.spark_session import make_spark
from src.data_preparation.pipelines.listings_pipeline import run_listings_pipeline
from src.data_preparation.pipelines.events_pipeline import run_events_pipeline
from src.utils import log

def main():
    log(" RAW DATA PREPARATION - CLASSIFIED ADS")
    log("="*60 + "\n")
    
    spark = make_spark()
    
    try:
        log("Step 1/3: Processing listings...")
        run_listings_pipeline(spark=spark)
        
        log("\nStep 2/3: Processing user events...")
        run_events_pipeline(spark=spark)
        
        log("\nStep 3/3: Merging data...")
        #run_merge_events_pipeline(spark=spark)
        
        log("\n" + "="*60)
        log(" PREPARATION COMPLETE")
        log("="*60)
        log("\nNext steps:")
        log("  1. make prepare-data      # Create sliding window")
        log("  2. make convert-recbole   # Convert to RecBole format")
        log("  3. make run-all           # Run experiments\n")
        
    except Exception as e:
        log(f"\nERRO: {e}")
        import traceback
        traceback.log_exc()
        sys.exit(1)
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

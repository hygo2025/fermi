import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.spark_session import make_spark
from src.data_preparation.pipelines.listings_pipeline import run_listings_pipeline
from src.data_preparation.pipelines.events_pipeline import run_events_pipeline
from src.utils import log

def main():
    log("="*60 + "")
    
    spark = make_spark()
    
    try:
        log("Step 1/3: Processing listings...")
        run_listings_pipeline(spark=spark)
        
        log("Step 2/3: Processing user events...")
        run_events_pipeline(spark=spark)
        
    except Exception as e:
        log(f"ERRO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

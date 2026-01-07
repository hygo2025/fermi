from typing import List

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from src.utils.enviroment import (
    listings_processed_path,
    events_processed_path,
    enriched_events_path, listing_id_mapping_path
)
from src.utils import log

def run_merge_events_pipeline(spark: SparkSession):
    log("\nExecutando pipeline completo de eventos...")
    listings = spark.read.option("mergeSchema", "true").parquet(listings_processed_path())
    events = spark.read.option("mergeSchema", "true").parquet(events_processed_path())
    log("\nIniciando merge_with_listings...")

    listings = (
        listings
        .withColumnRenamed("listing_id_numeric", "listing_id")
        .drop("month", "dt", "business_type")
    )

    df = events.join(listings, on="listing_id", how="inner")

    columns_to_drop: List[str] = [
        "anonymized_listing_id", "created_at", "updated_at"
    ]

    df = df.drop(*columns_to_drop)


    log(f"Salvando eventos enriquecidos em: {enriched_events_path()}")
    df.coalesce(4).write.mode("overwrite").partitionBy("dt").parquet(enriched_events_path())

    log("merge_with_listings concluído.")
    log("Pipeline completo concluído com sucesso.")

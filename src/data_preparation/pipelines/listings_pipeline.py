import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.functions import col

from src.utils.enviroment import listing_id_mapping_path, listings_raw_path, listings_processed_path
from src.utils.spark_utils import read_csv_data
from src.utils import log

def clean_data(df: DataFrame) -> DataFrame:
    for c in ['price', 'usable_areas', 'total_areas', 'ceiling_height']:
        if c in df.columns:
            df = df.withColumn(c, F.regexp_replace(F.col(c), r"[^0-9.]", "").cast("double"))

    for c in ['bathrooms', 'bedrooms', 'suites', 'parking_spaces', 'floors']:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast("integer"))

    if 'dt' in df.columns:
        df = df.withColumn('dt', F.to_date(F.col('dt')))
    if 'created_at' in df.columns:
        df = df.withColumn('created_at', F.to_timestamp(F.col('created_at')))
    if 'updated_at' in df.columns:
        df = df.withColumn('updated_at', F.to_timestamp(F.col('updated_at')))
    return df


def deduplicate_and_map_ids(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    df_active = df.filter(F.col("status") == "ACTIVE")

    window_spec = Window.partitionBy("anonymized_listing_id").orderBy(F.col("updated_at").desc())
    latest_df = (
        df_active.withColumn("rank", F.row_number().over(window_spec))
                 .filter(F.col("rank") == 1)
                 .drop("rank")
    )

    id_window = Window.orderBy(F.lit(1))
    distinct_ids = latest_df.select("anonymized_listing_id").distinct()
    mapping_table = distinct_ids.withColumn("listing_id_numeric", F.row_number().over(id_window))

    enriched_df = latest_df.join(mapping_table, "anonymized_listing_id", "inner")
    return enriched_df, mapping_table


def save_results(df_final: DataFrame, mapping_table: DataFrame):
    final_path = listings_processed_path()
    mapping_path = listing_id_mapping_path()

    df_final_persisted = None
    mapping_table_persisted = None
    try:
        df_final_persisted = df_final.persist()
        mapping_table_persisted = mapping_table.persist()

        print(f"\nSalvando listings processados em: {final_path}")
        df_final_persisted.coalesce(1).write.mode("overwrite").parquet(final_path)

        print(f"\nSalvando mapeamento de listings em: {mapping_path}")
        mapping_table_persisted.write.mode("overwrite").parquet(mapping_path)
    finally:
        if df_final_persisted:
            df_final_persisted.unpersist()
        if mapping_table_persisted:
            mapping_table_persisted.unpersist()


def run_listings_pipeline(spark: SparkSession):
    print("Iniciando pipeline de listings...")
    raw_path = listings_raw_path() + "/*.csv.gz"
    all_raw_listings = read_csv_data(spark, raw_path, multiline=True)
    # all_raw_listings = all_raw_listings.filter((col("state") == "Espírito Santo"))
    # all_raw_listings = all_raw_listings.filter(
    #     (col("city") == "Vitória") | (col("city") == "Serra") | (col("city") == "Vila Velha") | (col("city") == "Cariacica") | (col("city") == "Viana") | (col("city") == "Guarapari") | (col("city") == "Fundão")
    # )

    cleaned_listings = clean_data(all_raw_listings)
    final_df, mapping_table = deduplicate_and_map_ids(cleaned_listings)

    # final_df = final_df.drop("status", "floors", "ceiling_height")

    save_results(final_df, mapping_table)
    print("\nListings pipeline concluído.")

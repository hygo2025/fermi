import pyspark.sql.functions as F
import pygeohash as pgh
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.functions import col, md5, concat_ws, coalesce, lit

from src.utils.enviroment import get_config
from src.utils.spark_utils import read_csv_data
from src.utils import log


def create_canonical_id(df: DataFrame) -> DataFrame:

    config = get_config()
    geohash_precision = int(config.get("canonical_id", {}).get("geohash_precision", 7))
    geohash_udf = F.udf(
        lambda lat, lon: (
            pgh.encode(float(lat), float(lon), precision=geohash_precision)
            if lat is not None and lon is not None else None
        ),
        "string",
    )
    df = df.withColumn(
        "geo_hash",
        geohash_udf(
            col("lat_region").cast("double"),
            col("lon_region").cast("double"),
        )
    )


    df = df.withColumn(
        "geo_key",
        coalesce(
            col("geo_hash"),
            col("zip_code"),
            lit("UNKNOWN_GEO")
        )
    )



    df = df.withColumn(
        "area_bucket",
        coalesce(
            ((col("usable_areas") / 10).cast("int") * 10).cast("string"),
            lit("-1")
        )
    )


    df = df.withColumn(
        "bedrooms_normalized",
        coalesce(col("bedrooms").cast("string"), lit("0"))
    )

    df = df.withColumn(
        "unit_type_normalized",
        coalesce(col("unit_type"), lit("UNKNOWN"))
    )


    df = df.withColumn(
        "fingerprint",
        concat_ws(
            "|",
            col("geo_key"),
            col("area_bucket"),
            col("bedrooms_normalized"),
            col("unit_type_normalized")
        )
    )


    df = df.withColumn(
        "canonical_listing_id",
        md5(col("fingerprint"))
    )


    df = df.drop(
        "geo_hash", "geo_key",
        "area_bucket", "bedrooms_normalized",
        "unit_type_normalized", "fingerprint"
    )

    return df


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

    distinct_canonical = latest_df.select("canonical_listing_id").distinct()
    id_window = Window.orderBy("canonical_listing_id")
    canonical_to_numeric = distinct_canonical.withColumn(
        "listing_id_numeric",
        F.row_number().over(id_window)
    )


    mapping_table = (
        latest_df
        .select("anonymized_listing_id", "canonical_listing_id")
        .distinct()
        .join(canonical_to_numeric, "canonical_listing_id", "inner")
    )


    enriched_df = latest_df.join(
        mapping_table.select("anonymized_listing_id", "listing_id_numeric"),
        "anonymized_listing_id",
        "inner"
    )

    return enriched_df, mapping_table


def save_results(df_final: DataFrame, mapping_table: DataFrame):
    config = get_config()
    final_path = config['raw_data']['listings_processed_path']
    mapping_path = config['raw_data']['listing_id_mapping_path']

    df_final_persisted = None
    mapping_table_persisted = None
    try:
        df_final_persisted = df_final.persist()
        mapping_table_persisted = mapping_table.persist()

        log(f"Salvando listings processados em: {final_path}")
        df_final_persisted.coalesce(4).write.mode("overwrite").parquet(final_path)

        log(f"Salvando mapeamento de listings em: {mapping_path}")
        mapping_table_persisted.coalesce(1).write.mode("overwrite").parquet(mapping_path)
    finally:
        if df_final_persisted:
            df_final_persisted.unpersist()
        if mapping_table_persisted:
            mapping_table_persisted.unpersist()


def run_listings_pipeline(spark: SparkSession):
    log("Iniciando pipeline de listings...")
    config = get_config()
    raw_path = config['raw_data']['listings_raw_path'] + "/*.csv.gz"
    all_raw_listings = read_csv_data(spark, raw_path, multiline=True)




    cleaned_listings = clean_data(all_raw_listings)


    log("Criando canonical_listing_id para diminuir cold start...")
    canonicalized_listings = create_canonical_id(cleaned_listings)


    final_df, mapping_table = deduplicate_and_map_ids(canonicalized_listings)

    save_results(final_df, mapping_table)
    log("Listings pipeline concluído.")

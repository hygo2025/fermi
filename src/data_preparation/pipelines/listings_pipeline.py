import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.functions import col, md5, concat_ws, round as spark_round, coalesce, lit

from src.utils.enviroment import get_config
from src.utils.spark_utils import read_csv_data
from src.utils import log

def create_canonical_id(df: DataFrame) -> DataFrame:
    """
    Creates canonical ID for grouping physically identical properties.
    Uses fingerprint based on: location (lat/lon), area, bedrooms, suites, unit_type.
    """
    df = df.withColumn(
        "lat_normalized",
        spark_round(col("lat_region"), 4)
    ).withColumn(
        "lon_normalized", 
        spark_round(col("lon_region"), 4)
    )
    
    # Fallback: usar zip_code se lat/lon forem nulos
    df = df.withColumn(
        "geo_key",
        coalesce(
            concat_ws("_", col("lat_normalized"), col("lon_normalized")),
            col("zip_code"),
            lit("UNKNOWN_GEO")
        )
    )
    
    # 2. Bucketizar área útil em intervalos de 5m²
    # Fórmula: (usable_areas / 5).cast("int") * 5
    df = df.withColumn(
        "area_bucket",
        coalesce(
            ((col("usable_areas") / 5).cast("int") * 5).cast("string"),
            lit("-1")  # Nulos viram -1
        )
    )
    
    # 3. Tratar tipologia (bedrooms, suites, unit_type)
    df = df.withColumn(
        "bedrooms_normalized",
        coalesce(col("bedrooms").cast("string"), lit("0"))
    )
    
    
    if "suites" in df.columns:
        df = df.withColumn(
            "suites_normalized",
            coalesce(col("suites").cast("string"), lit("0"))
        )
    else:
        df = df.withColumn("suites_normalized", lit("0"))
    
    df = df.withColumn(
        "unit_type_normalized",
        coalesce(col("unit_type"), lit("UNKNOWN"))
    )
    
    # 4. Gerar fingerprint concatenando todos os componentes
    df = df.withColumn(
        "fingerprint",
        concat_ws(
            "|",
            col("geo_key"),
            col("area_bucket"),
            col("bedrooms_normalized"),
            col("suites_normalized"),
            col("unit_type_normalized")
        )
    )
    
    # 5. Criar canonical_listing_id como MD5 do fingerprint
    df = df.withColumn(
        "canonical_listing_id",
        md5(col("fingerprint"))
    )
    
    # Remover colunas auxiliares
    df = df.drop(
        "lat_normalized", "lon_normalized", "geo_key", 
        "area_bucket", "bedrooms_normalized", "suites_normalized",
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
    """
    Deduplicação e criação de mapeamentos:
    1. Filtra apenas listings ACTIVE

    3. Cria mapeamento: anonymized_listing_id -> listing_id_numeric + canonical_listing_id
    
    Returns:
        tuple: (df_enriquecido, tabela_mapeamento)
    """
    df_active = df.filter(F.col("status") == "ACTIVE")

    window_spec = Window.partitionBy("anonymized_listing_id").orderBy(F.col("updated_at").desc())
    latest_df = (
        df_active.withColumn("rank", F.row_number().over(window_spec))
                 .filter(F.col("rank") == 1)
                 .drop("rank")
    )

    distinct_canonical = latest_df.select("canonical_listing_id").distinct()
    id_window = Window.orderBy("canonical_listing_id")  # Ordenação determinística
    canonical_to_numeric = distinct_canonical.withColumn(
        "listing_id_numeric", 
        F.row_number().over(id_window)
    )
    
    # 2. Criar tabela de mapeamento completa: anonymized -> canonical -> numeric
    mapping_table = (
        latest_df
        .select("anonymized_listing_id", "canonical_listing_id")
        .distinct()
        .join(canonical_to_numeric, "canonical_listing_id", "inner")
    )
    
    # 3. Enriquecer df final com listing_id_numeric
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

    # 1. Limpeza de dados
    cleaned_listings = clean_data(all_raw_listings)
    
    # 2. Criar ID canônico (agrupamento de imóveis similares)
    log("Criando canonical_listing_id para diminuir cold start...")
    canonicalized_listings = create_canonical_id(cleaned_listings)
    
    # 3. Deduplicação e mapeamento de IDs
    final_df, mapping_table = deduplicate_and_map_ids(canonicalized_listings)

    save_results(final_df, mapping_table)
    log("Listings pipeline concluído.")

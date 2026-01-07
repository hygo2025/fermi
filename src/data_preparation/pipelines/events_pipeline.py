import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame, Window
from typing import List

from src.utils.enviroment import get_config
from src.utils.spark_utils import read_csv_data
from src.utils import log


def clean_event_data(df: DataFrame) -> DataFrame:
    for name in ["anonymized_user_id", "anonymized_anonymous_id", "anonymized_listing_id"]:
        if name in df.columns:
            df = df.withColumn(
                name,
                F.when(F.trim(F.col(name)) == "", None).otherwise(F.trim(F.col(name)))
            )

    if "collector_timestamp" in df.columns:
        df = df.withColumn(
            "event_ts",
            (F.col("collector_timestamp").cast("bigint") / 1000).cast("timestamp")
        )

    return df


def process_raw_events(spark: SparkSession) -> DataFrame:
    log("\nProcessing raw events...")

    config = get_config()
    sale_raw_path = config['raw_data']['events_raw_path'] + "/*.csv.gz"
    all_raw_events = read_csv_data(spark, sale_raw_path, multiline=False)

    listing_map = spark.read.parquet(config['raw_data']['listing_id_mapping_path'])
    joined_df = all_raw_events.join(listing_map, on="anonymized_listing_id", how="inner")

    return clean_event_data(joined_df)


def resolve_user_identities(events: DataFrame) -> DataFrame:
    log("\nResolving user identities...")

    num_partitions = 512
    collision_threshold = 7
    events = events.repartition(num_partitions, F.col("anonymized_listing_id"))

    grouped = events.groupBy("anonymized_anonymous_id").agg(
        F.collect_list("anonymized_user_id").alias("user_id_list")
    )

    counted_grouped = (
        grouped
        .withColumn("distinct_user_ids", F.array_distinct(F.col("user_id_list")))
        .withColumn(
            "user_id_entries",
            F.expr("""
                transform(
                    distinct_user_ids,
                    x -> struct(
                        x as key,
                        size(filter(user_id_list, y -> y = x)) as value
                    )
                )
            """)
        )
        .withColumn("user_id", F.map_from_entries(F.col("user_id_entries")))
        .drop("distinct_user_ids", "user_id_entries", "user_id_list")
    )

    filtered = counted_grouped.filter(
        F.size(F.map_keys(F.col("user_id"))) <= collision_threshold
    )

    user_sessions = filtered.select(
        F.col("anonymized_anonymous_id"),
        F.expr("""
            struct(
                anonymized_anonymous_id AS anonymous_id,
                (
                    sort_array(
                        transform(
                            arrays_zip(map_keys(user_id), map_values(user_id)),
                            x -> struct(x['0'] AS key, -1 * x['1'] AS valNeg)
                        ),
                        true
                    )[0]['key']
                ) AS id
            )
        """).alias("user_session")
    ).select(
        F.col("user_session.anonymous_id"),
        F.col("user_session.id")
    )

    log(f"\nTotal de usuários resolvidos: {user_sessions.count()}")
    return user_sessions


def join_events_with_sessions(events: DataFrame, users: DataFrame, listings: DataFrame) -> DataFrame:
    log("\nIniciando join_events_with_sessions...")

    listings = listings.withColumnRenamed("anonymized_listing_id", "listing_id")

    events = (
        events
        .withColumnRenamed("anonymized_user_id", "user_id")
        .withColumnRenamed("anonymized_anonymous_id", "anonymous_id")
        .withColumnRenamed("anonymized_listing_id", "listing_id")
        .drop("usage_types", "is_bot", "event_date", "month", "listing_id_numeric")
    )

    window_spec = Window.partitionBy("anonymous_id")
    events = events.withColumn(
        "session_user_id",
        F.first(F.col("user_id"), ignorenulls=True).over(window_spec)
    )

    events = events.join(
        users.select(F.col("anonymous_id").alias("user_anonymous_id"), "id"),
        events["anonymous_id"] == F.col("user_anonymous_id"),
        "left"
    )

    events = events.withColumn(
        "user_id",
        F.when(F.col("user_id").isNull(), F.col("id")).otherwise(F.col("user_id"))
    ).drop("id", "user_anonymous_id")

    events = events.drop("session_user_id")

    events = events.join(listings.select("listing_id"), on="listing_id", how="inner")

    return events


def create_numeric_keys(events: DataFrame) -> DataFrame:
    log("\nIniciando criação de chaves numéricas...")

    id_window = Window.orderBy(F.lit(1))

    # user_id (usuário logado) → user_logged_numeric_id
    distinct_logged_users = events.select("user_id").distinct()
    logged_map = (
        distinct_logged_users
        .withColumn("user_logged_numeric_id", F.concat(F.lit("U_"), F.row_number().over(id_window)))
    )
    events = events.join(logged_map, "user_id", "left")

    # anonymous_id → anonymous_numeric_id
    distinct_anonymous = events.select("anonymous_id").distinct()
    anonymous_map = (
        distinct_anonymous
        .withColumn("anonymous_numeric_id", F.concat(F.lit("A_"), F.row_number().over(id_window)))
    )
    events = events.join(anonymous_map, "anonymous_id", "left")

    # anonymized_session_id → session_numeric_id
    distinct_sessions = events.select("anonymized_session_id").distinct()
    session_map = (
        distinct_sessions
        .withColumn("session_numeric_id", F.concat(F.lit("S_"), F.row_number().over(id_window)))
    )
    events = events.join(session_map, "anonymized_session_id", "left")

    log(
        f"Usuários logados: {logged_map.count()}, "
        f"Anônimos: {anonymous_map.count()}, "
        f"Sessões: {session_map.count()}"
    )

    return events

def save_events(spark: SparkSession, events: DataFrame) -> None:
    log("\nIniciando salvamento dos eventos...")

    events = events.withColumn(
        "unique_user_id",
        F.coalesce(
            F.col("user_logged_numeric_id"),
            F.col("anonymous_numeric_id"),
        )
    )

    # Traz o mapeamento numérico de listings
    listing_map = (
        spark.read
        .parquet(listing_id_mapping_path())
        .select(
            F.col("anonymized_listing_id").alias("listing_id"),
            "listing_id_numeric"
        )
    )

    df = events.join(listing_map, on="listing_id", how="left")

    columns_to_drop: List[str] = [
        "listing_id", "anonymized_session_id", "user_id",
        "anonymous_id", "browser_family"
    ]

    df = df.drop(*columns_to_drop)

    df = (
        df.withColumnRenamed("user_logged_numeric_id", "user_id")
        .withColumnRenamed("anonymous_numeric_id", "anonymous_id")
        .withColumnRenamed("session_numeric_id", "session_id")
        .withColumnRenamed("unified_user_id", "unique_user_id")
        .withColumnRenamed("listing_id_numeric", "listing_id")
        .withColumn("dt", F.col("dt"))
    )

    first_columns = ["listing_id", "unique_user_id", "session_id", "user_id", "anonymous_id", "event_type", "dt"]
    remaining_columns = [c for c in df.columns if c not in first_columns]
    df = df.select(*first_columns, *remaining_columns)

    config = get_config()
    events_path = config['raw_data']['events_processed_path']
    log(f"\nSalvando eventos com chaves numéricas em: {events_path}")
    df.coalesce(8).write.mode("overwrite").partitionBy("dt").parquet(events_path)

    log("Eventos salvos com sucesso.")


def run_events_pipeline(spark: SparkSession):
    log("\nExecutando pipeline completo de eventos...")
    config = get_config()
    listings = spark.read.option("mergeSchema", "true").parquet(config['raw_data']['listings_processed_path'])

    events = process_raw_events(spark=spark)
    user_sessions = resolve_user_identities(events=events)
    joined = join_events_with_sessions(events=events, users=user_sessions, listings=listings)
    events = create_numeric_keys(events=joined)
    save_events(spark=spark, events=events)


    log("Pipeline completo concluído com sucesso.")

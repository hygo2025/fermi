#%%
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pathlib import Path
from datetime import datetime, timedelta
import argparse

from pyspark.sql import DataFrame, Window
#%%
import pandas as pd
from IPython.display import display

def show_pd(df, limit=10, truncate=False):
    pandas_df = df.limit(limit).toPandas()

    if not truncate:
        with pd.option_context(
            'display.max_rows', None,        # IMPORTANTE: Exibe todas as linhas do DF (não esconde o meio)
            'display.max_columns', None,     # Exibe todas as colunas (não esconde colunas do meio)
            'display.max_colwidth', None,    # Exibe todo o texto da célula (não corta strings longas)
            'display.max_seq_items', None    # IMPORTANTE: Exibe todos os itens se a célula tiver uma lista/array
        ):
            display(pandas_df)
    else:
        display(pandas_df)
#%%
def make_spark(
        memory_storage_fraction: float = 0.2,
) -> SparkSession:
    return (
        SparkSession.builder
        .appName("spark")
        .master("local[*]")
        .config("spark.driver.memory", "112g")
        .config("spark.sql.shuffle.partitions", 200)
        .config("spark.default.parallelism", 200)
        .config("spark.memory.storageFraction", memory_storage_fraction)
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "256m")
        .config("spark.sql.files.maxPartitionBytes", "256m")
        .config("spark.sql.files.openCostInBytes", "8m")
        .config("spark.shuffle.manager", "sort")
        .config("spark.sql.autoBroadcastJoinThreshold", "512m")# isso tem de ficar em um valor bem baixo talvez algo proximo a 10m
        .config("spark.sql.parquet.filterPushdown", "true")
        .config("spark.sql.parquet.enableVectorizedReader", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") # isso é um teste para deixar o toPandas mais rapido
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:MaxGCPauseMillis=200 -XX:+ExitOnOutOfMemoryError")
    ).getOrCreate()
#%%
def log(message: str):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
#%%
spark = make_spark()
#%%
EVENTS_PROCESSED_PATH="/home/hygo2025/Documents/data/processed_data/events"
LISTINGS_PROCESSED_PATH="/home/hygo2025/Documents/data/processed_data/listings"
SESSIONS_PROCESSED_PATH="/home/hygo2025/Documents/data/processed_data/sessions"
#%%
from pyspark.sql.types import IntegerType


def load_events_data(n_days: int, start_date: str, input_path: str) -> DataFrame:
    """Load raw events from parquet partitions"""
    log(f"Loading {n_days} days starting from {start_date}")

    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = start_dt + timedelta(days=n_days - 1)

    # Load parquet with partition filter
    df = spark.read.parquet(input_path)

    # Filter by date range
    df = df.filter(
        (F.col('dt') >= start_dt.strftime('%Y-%m-%d')) &
        (F.col('dt') <= end_dt.strftime('%Y-%m-%d'))
    )
    total_events = df.count()
    log(f"Loaded {total_events:_} events")

    df = df.filter((F.col("business_type") == "SALE"))
    total_events = df.count()
    log(f"Loaded filter business_type {total_events:_} events")

    return df

def filter_interaction_events(df: DataFrame) -> DataFrame:
    """Keep only real user-item interactions (exclude RankingRendered)"""
    log("Filtering interaction events (excluding RankingRendered,GalleryClicked)")

    # Keep events that represent actual user interest
    interaction_types = [
        'ListingRendered',      # User viewed listing detail
        #'RankingRendered',       # User viewed listing in ranking
        #'GalleryClicked',       # User clicked on gallery/image
        #'RankingClicked',       # User clicked item in ranking
        #'LeadPanelClicked',     # User clicked contact panel
        #'LeadClicked',          # User initiated contact
        #'FavoriteClicked',      # User favorited item
        #'ShareClicked',         # User shared item
    ]

    df_filtered = df.filter(F.col('event_type').isin(interaction_types))

    total_before = df.count()
    total_after = df_filtered.count()
    log(f"Kept {total_after:_} interaction events ({total_after/total_before*100:.2f}%)")

    return df_filtered

def create_sessions(df: DataFrame) -> DataFrame:
    """Create session-based interaction sequences"""
    log("Creating session sequences")

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
    log(f"Created {unique_sessions:,} unique sessions")

    return sessions

def filter_sessions(df: DataFrame, min_session_len: int, max_session_len: int) -> DataFrame:
    """Filter sessions by length (2-50 interactions)"""
    log(f"Filtering sessions: {min_session_len}-{max_session_len} interactions")

    # Count session sizes
    session_sizes = df.groupBy('session_id').agg(
        F.count('*').alias('session_size')
    )

    # Filter valid sessions
    valid_sessions = session_sizes.filter(
        (F.col('session_size') >= min_session_len) &
        (F.col('session_size') <= max_session_len)
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

    log(f"Sessions: {sessions_before:,} → {sessions_after:,}")
    log(f"Events: {events_before:,} → {events_after:,}")

    return df_filtered

def filter_rare_items(df: DataFrame, min_item_support: int) -> DataFrame:
    """Remove items with fewer than min_item_support occurrences"""
    log(f"Filtering items with <{min_item_support} occurrences")

    # Count item occurrences
    item_counts = df.groupBy('item_id').agg(
        F.count('*').alias('item_count')
    )

    # Filter valid items
    valid_items = item_counts.filter(
        F.col('item_count') >= min_item_support
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

    log(f"Items: {items_before:,} → {items_after:,}")
    log(f"Events: {events_before:,} → {events_after:,}")

    return df_filtered
#%%
listings = spark.read.option("mergeSchema", "true").parquet(LISTINGS_PROCESSED_PATH)
listings_before = listings.count()


listings = listings.filter(
    (F.col("city") == "Vitória") | (F.col("city") == "Serra") | (F.col("city") == "Vila Velha") | (F.col("city") == "Cariacica") | (F.col("city") == "Viana") | (F.col("city") == "Guarapari") | (F.col("city") == "Fundão")
)

listings_after = listings.count()

log(f"Loaded {listings_before:_} listings")
log(f"Filtered to {listings_after:_} listings in the target cities")

events = load_events_data(n_days = 30, start_date="2024-05-01", input_path=EVENTS_PROCESSED_PATH)
events_before = events.count()

events = events.join(
    listings,
    events.listing_id == listings.listing_id_numeric,
    "left_semi"
)

events_after = events.count()
log(f"Filtered events from {events_before:_}")
log(f"to {events_after:_} after joining with listings in target cities")

events = filter_interaction_events(events)
events = create_sessions(events)
events = filter_sessions(events, min_session_len=2, max_session_len=50)
events = filter_rare_items(events, min_item_support = 5)
events = filter_sessions(events, min_session_len=2, max_session_len=50)
#%%
events.printSchema()
#%%
show_pd(events)
#%%
events.filter(F.col("session_id") == "S_8128600").count()
#%%
show_pd(events.filter(F.col("session_id") == "S_8128600").sort(F.col("timestamp")), limit=200)
#%%
events.count()
#%%
events.coalesce(1).write.mode('overwrite').parquet(SESSIONS_PROCESSED_PATH)
#%%

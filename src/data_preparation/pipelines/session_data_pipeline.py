from pyspark.sql import functions as F, SparkSession
from pyspark.sql.window import Window

from src.utils import log, make_spark
from src.utils.enviroment import get_config


class SessionDataPipeline:
    def __init__(self,
                 spark: SparkSession,
                 recbole_format: bool = True,
                 output_path: str = None,
                 start_date: str = None,
                 end_date: str = None,
                 ):
        project_config = get_config()
        self.recbole_format = recbole_format
        raw_data_config = project_config['raw_data']
        data_prep_config = project_config['data_preparation']
        self.config = {
            'events_path': raw_data_config['events_path'],
            'listings_path': raw_data_config.get('listings_path'),
            'output_path': output_path or raw_data_config['output_path'],
            'dataset_name': project_config['dataset'],
            'start_date': start_date or data_prep_config['start_date'],
            'end_date': end_date or data_prep_config['end_date'],
            'min_session_length': data_prep_config['min_session_length'],
            'max_session_length': data_prep_config.get('max_session_length', 50),
            'min_item_freq': data_prep_config['min_item_freq']
        }
        self.spark = spark

    def load_events(self, start_date: str, end_date: str):
        events_path = self.config['events_path']
        log(f" Carregando eventos: {start_date} → {end_date}")
        log(f"   Path: {events_path}")
        df = self.spark.read.parquet(events_path)
        df = df.filter(
            (F.col('dt') >= start_date) &
            (F.col('dt') <= end_date)
        )
        count = df.count()
        log(f"    {count:_} eventos carregados")
        df = df.filter(F.col('business_type') == 'SALE')
        count_sale = df.count()
        log(f"    {count_sale:_} eventos após filtrar business_type=SALE")
        return df

    def filter_interaction_events(self, df):
        log(" Filtrando eventos de interação...")
        interaction_types = [
            'ListingRendered',
            'RankingClicked',
            'LeadPanelClicked',
            'LeadClicked',
            'FavoriteClicked',
            'ShareClicked',
        ]

        df_filtered = df.filter(F.col('event_type').isin(interaction_types))

        total_before = df.count()
        total_after = df_filtered.count()
        log(f"    {total_after:_} eventos de interação ({total_after / total_before * 100:.2f}%)")
        log(f"    Tipos mantidos: {interaction_types}")

        return df_filtered

    def filter_by_location(self, df):
        listings_path = self.config.get('listings_path')
        if not listings_path:
            log("     listings_path not configured, skipping location filter")
            return df
        log(" Filtrando por localização...")
        listings = self.spark.read.option("mergeSchema", "true").parquet(listings_path)
        listings_before = listings.count()
        listings_after = listings.count()
        log(f"    {listings_before:_} listings → {listings_after:_} nas localizacoes alvo")
        events_before = df.count()
        df = df.join(
            listings.select('listing_id_numeric'),
            df.listing_id == listings.listing_id_numeric,
            "left_semi"
        )
        events_after = df.count()
        log(f"    {events_before:_} eventos → {events_after:_} após filtro geográfico")
        return df

    def prepare_sessions(self, df, recbole_format=False):
        log(" Preparando sessões com deduplicação consecutiva...")
        df = df.withColumn('original_user_id', F.col('user_id'))
        df = df.withColumn('user_id', F.col('session_id'))
        df = df.withColumn('item_id', F.col('listing_id'))
        df = df.withColumn('timestamp', F.unix_timestamp(F.col('event_ts')))
        df = df.filter(
            F.col('user_id').isNotNull() &
            F.col('item_id').isNotNull() &
            F.col('timestamp').isNotNull()
        )
        df = df.withColumn("tie_breaker", F.monotonically_increasing_id())
        window_spec = Window.partitionBy("user_id").orderBy("timestamp", "tie_breaker")
        df = df.withColumn("prev_item_id", F.lag("item_id").over(window_spec))
        df_clean = df.filter(
            (F.col("item_id") != F.col("prev_item_id")) |
            (F.col("prev_item_id").isNull())
        )
        df_clean = df_clean.drop("prev_item_id", "tie_breaker")

        if recbole_format:
            df_clean = df_clean.select('user_id', 'item_id', 'timestamp')
        df_clean = df_clean.orderBy('user_id', 'timestamp')
        count_before = df.count()
        count_after = df_clean.count()
        log(f"    Deduplicação: {count_before:_} -> {count_after:_} interações (Mantidos retornos A->B->A)")
        return df_clean

    def filter_sessions_by_length(self, df, min_length: int, max_length: int):
        log(f" Filtrando sessões ({min_length}-{max_length} interações), truncando as longas...")
        session_sizes = df.groupBy('user_id').agg(
            F.count('*').alias('session_size')
        )
        df_with_size = df.join(session_sizes, on='user_id', how='inner')
        df_filtered = df_with_size.filter(F.col('session_size') >= min_length)
        window_spec = Window.partitionBy('user_id').orderBy(F.col('timestamp').desc())
        df_filtered = df_filtered.withColumn('rn', F.row_number().over(window_spec))
        df_filtered = df_filtered.filter(
            (F.col('session_size') <= max_length) | (F.col('rn') <= max_length)
        )
        df_filtered = df_filtered.drop('rn', 'session_size')
        df_filtered = df_filtered.orderBy('user_id', 'timestamp')

        sessions_before = session_sizes.count()
        sessions_after = df_filtered.select('user_id').distinct().count()
        events_before = df.count()
        events_after = df_filtered.count()

        log(f"    Sessões: {sessions_before:_} → {sessions_after:_}")
        log(f"    Eventos: {events_before:_} → {events_after:_}")

        return df_filtered

    def filter_rare_items(self, df, min_support: int):
        log(f" Filtrando itens raros (mín. {min_support} ocorrências)...")
        item_counts = df.groupBy('item_id').agg(
            F.count('*').alias('item_count')
        )
        valid_items = item_counts.filter(
            F.col('item_count') >= min_support
        )
        df_filtered = df.join(
            valid_items.select('item_id'),
            on='item_id',
            how='inner'
        )

        items_before = item_counts.count()
        items_after = valid_items.count()
        events_before = df.count()
        events_after = df_filtered.count()

        log(f"    Itens: {items_before:_} → {items_after:_}")
        log(f"    Eventos: {events_before:_} → {events_after:_}")

        return df_filtered

    def run(self):
        df = self.load_events(
            self.config['start_date'],
            self.config['end_date']
        )
        df = self.filter_by_location(df)
        df = self.filter_interaction_events(df)
        df = self.prepare_sessions(df, recbole_format=self.recbole_format)
        min_session_len = self.config.get('min_session_length', 2)
        max_session_len = self.config.get('max_session_length', 50)
        df = self.filter_sessions_by_length(df, min_session_len, max_session_len)
        min_item_support = self.config.get('min_item_freq', 2)
        df = self.filter_rare_items(df, min_item_support)
        df = self.filter_sessions_by_length(df, min_session_len, max_session_len)
        return df

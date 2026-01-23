from pyspark.sql import functions as F, SparkSession
from pyspark.sql.window import Window

from src.utils import log, make_spark
from src.utils.enviroment import get_config


class SessionDataPipeline:
    def __init__(self,
                 spark: SparkSession,
                 output_path: str = None,
                 start_date: str = None,
                 end_date: str = None,
                 ):
        project_config = get_config()

        # Build pipeline config
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
        """Carrega eventos brutos do período especificado"""
        events_path = self.config['events_path']

        log(f" Carregando eventos: {start_date} → {end_date}")
        log(f"   Path: {events_path}")

        df = self.spark.read.parquet(events_path)

        # Filtros básicos
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
            'ListingRendered',  # User viewed listing detail
            # 'RankingRendered',      # User viewed listing in ranking
            # 'GalleryClicked',       # User clicked on gallery/image
            'RankingClicked',  # User clicked item in ranking
            'LeadPanelClicked',  # User clicked contact panel
            'LeadClicked',  # User initiated contact
            'FavoriteClicked',  # User favorited item
            'ShareClicked',  # User shared item
        ]

        df_filtered = df.filter(F.col('event_type').isin(interaction_types))

        total_before = df.count()
        total_after = df_filtered.count()
        log(f"    {total_after:_} eventos de interação ({total_after / total_before * 100:.2f}%)")
        log(f"    Tipos mantidos: {interaction_types}")

        return df_filtered

    def filter_by_location(self, df):
        """Filtra eventos por localização"""
        listings_path = self.config.get('listings_path')

        if not listings_path:
            log("     listings_path não configurado, pulando filtro de localização")
            return df

        log(" Filtrando por localização...")

        # Carrega listings
        listings = self.spark.read.option("mergeSchema", "true").parquet(listings_path)
        listings_before = listings.count()

        # Filtra cidades da Grande Vitória/ES
        # target_cities = ['Vitória', 'Serra', 'Vila Velha', 'Cariacica', 'Viana', 'Guarapari', 'Fundão']
        # listings = listings.filter(F.col('city').isin(target_cities))

        # target_states = ['Espírito Santo']
        # listings = listings.filter(F.col('state').isin(target_states))

        listings_after = listings.count()

        log(f"    {listings_before:_} listings → {listings_after:_} nas localizacoes alvo")

        # Join com eventos (left_semi = mantém apenas eventos de listings válidos)
        events_before = df.count()
        df = df.join(
            listings.select('listing_id_numeric'),
            df.listing_id == listings.listing_id_numeric,
            "left_semi"
        )
        events_after = df.count()

        log(f"    {events_before:_} eventos → {events_after:_} após filtro geográfico")

        return df

    def prepare_sessions(self, df):
        """
        Prepara sessões removendo APENAS repetições consecutivas (A->A),
        mas mantendo retornos (A->B->A).
        """
        log(" Preparando sessões com deduplicação consecutiva...")

        # 1. Seleção Básica e Conversão de TS
        df = df.select(
            F.col('session_id').alias('user_id'),
            F.col('listing_id').alias('item_id'),
            F.col('event_ts').alias('timestamp')
        ).filter(
            F.col('user_id').isNotNull() &
            F.col('item_id').isNotNull() &
            F.col('timestamp').isNotNull()
        ).withColumn(
            'timestamp',
            F.unix_timestamp(F.col('timestamp'))
        )

        # 2. Definir Janela para olhar o item anterior
        # Adicionamos um tie_breaker para garantir ordem estável em eventos do mesmo segundo
        df = df.withColumn("tie_breaker", F.monotonically_increasing_id())

        window_spec = Window.partitionBy("user_id").orderBy("timestamp", "tie_breaker")

        # 3. Criar coluna com o item anterior (Lag)
        df = df.withColumn("prev_item_id", F.lag("item_id").over(window_spec))

        # 4. Filtrar: Manter apenas se o item atual for DIFERENTE do anterior
        # O primeiro item da sessão (prev is null) sempre fica.
        df_clean = df.filter(
            (F.col("item_id") != F.col("prev_item_id")) |
            (F.col("prev_item_id").isNull())
        )

        # 5. Limpeza final
        df_clean = df_clean.select('user_id', 'item_id', 'timestamp')

        # Ordenação final para garantir a sequência no arquivo .inter
        df_clean = df_clean.orderBy('user_id', 'timestamp')

        # Log de impacto
        count_before = df.count()
        count_after = df_clean.count()
        log(f"    Deduplicação: {count_before:_} -> {count_after:_} interações (Mantidos retornos A->B->A)")

        return df_clean

    def filter_sessions_by_length(self, df, min_length: int, max_length: int):
        """Filtra sessões por comprimento"""
        log(f" Filtrando sessões ({min_length}-{max_length} interações)...")

        # Conta tamanho das sessões
        session_sizes = df.groupBy('user_id').agg(
            F.count('*').alias('session_size')
        )

        # Filtra sessões válidas
        valid_sessions = session_sizes.filter(
            (F.col('session_size') >= min_length) &
            (F.col('session_size') <= max_length)
        )

        # Join para manter apenas sessões válidas
        df_filtered = df.join(
            valid_sessions.select('user_id'),
            on='user_id',
            how='inner'
        )

        sessions_before = session_sizes.count()
        sessions_after = valid_sessions.count()
        events_before = df.count()
        events_after = df_filtered.count()

        log(f"    Sessões: {sessions_before:_} → {sessions_after:_}")
        log(f"    Eventos: {events_before:_} → {events_after:_}")

        return df_filtered

    def filter_rare_items(self, df, min_support: int):
        """Remove itens com menos de min_support ocorrências"""
        log(f" Filtrando itens raros (mín. {min_support} ocorrências)...")

        # Conta ocorrências de itens
        item_counts = df.groupBy('item_id').agg(
            F.count('*').alias('item_count')
        )

        # Filtra itens válidos
        valid_items = item_counts.filter(
            F.col('item_count') >= min_support
        )

        # Join para manter apenas itens válidos
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
        """Executa pipeline completo"""
        # 1. Carrega eventos
        df = self.load_events(
            self.config['start_date'],
            self.config['end_date']
        )

        # 2. Filtra por localização (se configurado)
        df = self.filter_by_location(df)

        # 3. Filtra eventos de interação
        df = self.filter_interaction_events(df)

        # 4. Prepara sessões
        df = self.prepare_sessions(df)

        # 5. Filtra sessões por comprimento
        min_session_len = self.config.get('min_session_length', 2)
        max_session_len = self.config.get('max_session_length', 50)
        df = self.filter_sessions_by_length(df, min_session_len, max_session_len)

        # 6. Filtra itens raros
        min_item_support = self.config.get('min_item_freq', 2)
        df = self.filter_rare_items(df, min_item_support)

        # 7. Re-filtra sessões (após remover itens raros, algumas sessões podem ter ficado curtas)
        df = self.filter_sessions_by_length(df, min_session_len, max_session_len)

        return df
import argparse
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.utils import log, make_spark
from src.utils.enviroment import get_config


class RecBoleDataPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.spark = make_spark()

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
        
        # Filtra business_type = SALE (importante para o domínio)
        df = df.filter(F.col('business_type') == 'SALE')
        count_sale = df.count()
        log(f"    {count_sale:_} eventos após filtrar business_type=SALE")
        
        return df
    
    def filter_interaction_events(self, df):
        """Mantém apenas eventos de interação real (exclui RankingRendered)"""
        log(" Filtrando eventos de interação...")
        
        # Eventos que representam interesse real do usuário
        interaction_types = [
            'ListingRendered',  # User viewed listing detail
            # 'RankingRendered',       # User viewed listing in ranking
            # 'GalleryClicked',       # User clicked on gallery/image
            # 'RankingClicked',       # User clicked item in ranking
            # 'LeadPanelClicked',     # User clicked contact panel
            # 'LeadClicked',          # User initiated contact
            # 'FavoriteClicked',      # User favorited item
            # 'ShareClicked',         # User shared item
        ]
        
        df_filtered = df.filter(F.col('event_type').isin(interaction_types))
        
        total_before = df.count()
        total_after = df_filtered.count()
        log(f"    {total_after:_} eventos de interação ({total_after/total_before*100:.2f}%)")
        
        return df_filtered
    
    def filter_by_location(self, df):
        """Filtra eventos por localização (cidades da Grande Vitória/ES)"""
        listings_path = self.config.get('listings_path')
        
        if not listings_path:
            log("     listings_path não configurado, pulando filtro de localização")
            return df
        
        log(" Filtrando por localização...")
        
        # Carrega listings
        listings = self.spark.read.option("mergeSchema", "true").parquet(listings_path)
        listings_before = listings.count()
        
        # Filtra cidades da Grande Vitória/ES
        target_cities = ['Vitória', 'Serra', 'Vila Velha', 'Cariacica', 'Viana', 'Guarapari', 'Fundão']
        listings = listings.filter(F.col('city').isin(target_cities))
        listings_after = listings.count()
        
        log(f"    {listings_before:_} listings → {listings_after:_} nas cidades alvo")
        
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
        """Prepara dados em formato session-based"""
        log(" Preparando sessões...")
        
        # Seleciona e renomeia colunas
        df = df.select(
            F.col('session_id').alias('user_id'),  # RecBole agrupa por user_id
            F.col('listing_id').alias('item_id'),
            F.col('event_ts').alias('timestamp')
        ).filter(
            F.col('user_id').isNotNull() &
            F.col('item_id').isNotNull() &
            F.col('timestamp').isNotNull()
        )
        
        # Converte timestamp para Unix (segundos desde 1970-01-01)
        df = df.withColumn(
            'timestamp',
            F.unix_timestamp(F.col('timestamp'))
        )
        
        # Ordena por sessão e tempo
        df = df.orderBy('user_id', 'timestamp')
        
        # Adiciona posição dentro da sessão (para análise)
        window_spec = Window.partitionBy('user_id').orderBy('timestamp')
        df = df.withColumn('position', F.row_number().over(window_spec))
        
        unique_sessions = df.select('user_id').distinct().count()
        log(f"    {unique_sessions:_} sessões únicas criadas")
        
        return df
    
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
    
    def save_inter_file(self, df, output_path: Path):
        """Salva DataFrame como arquivo .inter do RecBole"""
        log(f" Salvando arquivo .inter...")
        log(f"   Path: {output_path}")
        
        # Converte para Pandas (cabe em memória após filtros)
        pdf = df.toPandas()
        
        # Garante tipos corretos
        pdf['user_id'] = pdf['user_id'].astype(str)
        pdf['item_id'] = pdf['item_id'].astype(str)
        pdf['timestamp'] = pdf['timestamp'].astype(float)
        
        # Ordena novamente (garantia)
        pdf = pdf.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        
        # Cria diretório
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Escreve arquivo com header RecBole
        with open(output_path, 'w') as f:
            # Header: field_name:type
            f.write("user_id:token\titem_id:token\ttimestamp:float")
            
            # Data (tab-separated)
            for _, row in pdf.iterrows():
                f.write(f"{row['user_id']}\t{row['item_id']}\t{row['timestamp']}")
        
        size_mb = output_path.stat().st_size / (1024 * 1024)
        log(f"    Arquivo salvo: {len(pdf):_} interações ({size_mb:.1f} MB)")
        
        return pdf
    
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
        min_item_support = self.config.get('min_item_freq', 5)
        df = self.filter_rare_items(df, min_item_support)
        
        # 7. Re-filtra sessões (após remover itens raros, algumas sessões podem ter ficado curtas)
        df = self.filter_sessions_by_length(df, min_session_len, max_session_len)
        
        # Remove coluna auxiliar 'position' antes de salvar
        df = df.select('user_id', 'item_id', 'timestamp')
        
        # Estatísticas finais
        count = df.count()
        n_users = df.select('user_id').distinct().count()
        n_items = df.select('item_id').distinct().count()
        
        log(" ESTATÍSTICAS FINAIS:")
        log(f"    {count:_} interações")
        log(f"    {n_users:_} sessões")
        log(f"    {n_items:_} itens únicos")
        
        # 8. Salva arquivo atômico .inter
        dataset_name = self.config.get('dataset_name', 'realestate')
        output_dir = Path(self.config['output_path']) / dataset_name
        inter_file = output_dir / f"{dataset_name}.inter"
        
        self.save_inter_file(df, inter_file)
        
        log("" + "=" * 80)
        log(" PIPELINE COMPLETO!")
        log("=" * 80)
        log(f"Dataset: {dataset_name}")
        log(f"Arquivo: {inter_file}")
        log("Próximo passo:")
        log(f"  make benchmark --dataset {dataset_name}")
        
        self.spark.stop()


def main():
    parser = argparse.ArgumentParser(description='Prepare RecBole dataset')
    parser.add_argument('--start-date', type=str, help='Override start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='Override end date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, help='Override output path')
    
    args = parser.parse_args()
    project_config = get_config()
    
    # Build pipeline config
    raw_data_config = project_config['raw_data']
    data_prep_config = project_config['data_preparation']
    
    config = {
        'events_path': raw_data_config['events_path'],
        'listings_path': raw_data_config.get('listings_path'),
        'output_path': args.output or raw_data_config['output_path'],
        'dataset_name': project_config['dataset'],
        'start_date': args.start_date or data_prep_config['start_date'],
        'end_date': args.end_date or data_prep_config['end_date'],
        'min_session_length': data_prep_config['min_session_length'],
        'max_session_length': data_prep_config.get('max_session_length', 50),
        'min_item_freq': data_prep_config['min_item_freq']
    }
    
    # Run pipeline
    pipeline = RecBoleDataPipeline(config)
    pipeline.run()


if __name__ == '__main__':
    main()

"""
FERMI - Data Preparation Pipeline
==================================
Gera um arquivo atômico .inter compatível com RecBole a partir dos dados brutos.

ESTRATÉGIA:
    1. Carrega eventos brutos (Spark)
    2. Filtra e limpa dados
    3. Gera um ÚNICO arquivo .inter (sem splits físicos)
    4. RecBole fará o split temporal em runtime (Leave-One-Out)

OUTPUT:
    {output_path}/{dataset_name}/{dataset_name}.inter
    
SCHEMA CRÍTICO:
    user_id:token    item_id:token    timestamp:float
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window


class RecBoleDataPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.spark = self._init_spark()
        
    def _init_spark(self) -> SparkSession:
        return (SparkSession.builder
                .appName("Fermi-RecBole-Pipeline")
                .config("spark.driver.memory", "8g")
                .config("spark.sql.shuffle.partitions", "200")
                .getOrCreate())
    
    def load_events(self, start_date: str, end_date: str):
        """Carrega eventos brutos do período especificado"""
        events_path = self.config['events_path']
        
        print(f"\n📂 Carregando eventos: {start_date} → {end_date}")
        print(f"   Path: {events_path}")
        
        df = self.spark.read.parquet(events_path)
        
        # Filtros básicos
        df = df.filter(
            (F.col('dt') >= start_date) &
            (F.col('dt') <= end_date) &
            (F.col('event_type') != 'RankingRendered') &
            (F.col('business_type') != 'SALE')
        )
        
        count = df.count()
        print(f"   ✓ {count:,} eventos carregados")
        
        return df
    
    def prepare_sessions(self, df):
        """Prepara dados em formato session-based"""
        print("\n🔧 Preparando sessões...")
        
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
        
        # Filtra sessões muito curtas
        min_length = self.config.get('min_session_length', 2)
        window = Window.partitionBy('user_id')
        df = df.withColumn('session_length', F.count('*').over(window))
        df = df.filter(F.col('session_length') >= min_length)
        
        # Filtra itens raros
        min_freq = self.config.get('min_item_freq', 5)
        item_window = Window.partitionBy('item_id')
        df = df.withColumn('item_freq', F.count('*').over(item_window))
        df = df.filter(F.col('item_freq') >= min_freq)
        
        # Remove colunas auxiliares
        df = df.select('user_id', 'item_id', 'timestamp')
        
        count = df.count()
        n_users = df.select('user_id').distinct().count()
        n_items = df.select('item_id').distinct().count()
        
        print(f"   ✓ {count:,} interações")
        print(f"   ✓ {n_users:,} sessões")
        print(f"   ✓ {n_items:,} itens únicos")
        
        return df
    
    def save_inter_file(self, df, output_path: Path):
        """Salva DataFrame como arquivo .inter do RecBole"""
        print(f"\n💾 Salvando arquivo .inter...")
        print(f"   Path: {output_path}")
        
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
            f.write("user_id:token\titem_id:token\ttimestamp:float\n")
            
            # Data (tab-separated)
            for _, row in pdf.iterrows():
                f.write(f"{row['user_id']}\t{row['item_id']}\t{row['timestamp']}\n")
        
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"   ✓ Arquivo salvo: {len(pdf):,} interações ({size_mb:.1f} MB)")
        
        return pdf
    
    def run(self):
        """Executa pipeline completo"""
        print("=" * 80)
        print("FERMI - DATA PREPARATION PIPELINE")
        print("=" * 80)
        
        # 1. Carrega eventos
        df = self.load_events(
            self.config['start_date'],
            self.config['end_date']
        )
        
        # 2. Prepara sessões
        df = self.prepare_sessions(df)
        
        # 3. Salva arquivo atômico .inter
        dataset_name = self.config.get('dataset_name', 'realestate')
        output_dir = Path(self.config['output_path']) / dataset_name
        inter_file = output_dir / f"{dataset_name}.inter"
        
        self.save_inter_file(df, inter_file)
        
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETO!")
        print("=" * 80)
        print(f"\nDataset: {dataset_name}")
        print(f"Arquivo: {inter_file}")
        print("\nPróximo passo:")
        print(f"  make benchmark --dataset {dataset_name}")
        
        self.spark.stop()


def main():
    parser = argparse.ArgumentParser(description='Prepare RecBole dataset')
    parser.add_argument('--config', type=str, default='config/project_config.yaml',
                        help='Path to project config')
    parser.add_argument('--start-date', type=str, help='Override start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='Override end date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, help='Override output path')
    
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        project_config = yaml.safe_load(f)
    
    # Build pipeline config
    raw_data_config = project_config['raw_data']
    data_prep_config = project_config['data_preparation']
    
    config = {
        'events_path': raw_data_config['events_path'],
        'output_path': args.output or raw_data_config['output_path'],
        'dataset_name': project_config['dataset'],
        'start_date': args.start_date or data_prep_config['start_date'],
        'end_date': args.end_date or data_prep_config['end_date'],
        'min_session_length': data_prep_config['min_session_length'],
        'min_item_freq': data_prep_config['min_item_freq']
    }
    
    # Run pipeline
    pipeline = RecBoleDataPipeline(config)
    pipeline.run()


if __name__ == '__main__':
    main()

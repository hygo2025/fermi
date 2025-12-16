"""
Script de preparação dos dados brutos.

Processa os dados brutos da plataforma de classificados (listings + events) e 
gera o dataset enriquecido usado como entrada para o sliding window.

Uso:
    python src/data_preparation/prepare_raw_data.py
    ou
    make prepare-raw-data
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.spark_session import make_spark
from src.data_preparation.pipelines.listings_pipeline import run_listings_pipeline
from src.data_preparation.pipelines.events_pipeline import run_events_pipeline
from src.data_preparation.pipelines.merge_events import run_merge_events_pipeline


def main():
    """Executa o pipeline de preparação de dados brutos."""
    print("\n" + "="*60)
    print(" PREPARAÇÃO DE DADOS BRUTOS - CLASSIFICADOS")
    print("="*60 + "\n")
    
    spark = make_spark()
    
    try:
        print("Etapa 1/3: Processando anúncios...")
        run_listings_pipeline(spark=spark)
        
        print("\nEtapa 2/3: Processando eventos de usuários...")
        run_events_pipeline(spark=spark)
        
        print("\nEtapa 3/3: Fazendo merge final...")
        run_merge_events_pipeline(spark=spark)
        
        print("\n" + "="*60)
        print(" PREPARAÇÃO CONCLUÍDA COM SUCESSO!")
        print("="*60)
        print("\nPróximos passos:")
        print("  1. make prepare-data      # Criar sliding window")
        print("  2. make convert-recbole   # Converter para RecBole")
        print("  3. make run-all           # Executar experimentos\n")
        
    except Exception as e:
        print(f"\nERRO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

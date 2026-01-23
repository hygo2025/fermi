import argparse
from pathlib import Path

import pandas as pd

from src.data_preparation.pipelines.session_data_pipeline import SessionDataPipeline
from src.utils import log, make_spark
from src.utils.enviroment import get_config


class RecBoleDataPipeline:
    def __init__(self, config: dict, session_data_pipeline: SessionDataPipeline):
        self.config = config
        self.session_data_pipeline = session_data_pipeline

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
            f.write("user_id:token\titem_id:token\ttimestamp:float\n")

            # Data (tab-separated)
            for _, row in pdf.iterrows():
                f.write(f"{row['user_id']}\t{row['item_id']}\t{row['timestamp']}\n")

        size_mb = output_path.stat().st_size / (1024 * 1024)
        log(f"    Arquivo salvo: {len(pdf):_} interações ({size_mb:.1f} MB)")

        return pdf  # Retorna pandas para reutilizar

    def save_sessions_for_api(self, pdf, output_path: Path):
        """Salva sessions em formato parquet para a API (recebe pandas DataFrame)"""
        log(f" Salvando sessions para API...")
        log(f"   Path: {output_path}")

        # Renomeia colunas para formato esperado pela API
        pdf_api = pdf.copy()
        pdf_api.rename(columns={'user_id': 'session_id'}, inplace=True)

        # Converte item_id para int
        pdf_api['item_id'] = pdf_api['item_id'].astype(int)

        # Adiciona user_id (mesmo que session_id)
        pdf_api['user_id'] = pdf_api['session_id']

        # Converte timestamp para datetime
        pdf_api['timestamp'] = pd.to_datetime(pdf_api['timestamp'], unit='s')

        # Salva parquet
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_api.to_parquet(output_path, index=False)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        log(f"    Arquivo salvo: {len(pdf_api):_} interações ({size_mb:.1f} MB)")
        log(f"    Sessões: {pdf_api['session_id'].nunique():_}")
        log(f"    Itens: {pdf_api['item_id'].nunique():_}")

    def run(self):
        """Executa pipeline completo"""
        df = self.session_data_pipeline.run()

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

        # 8. Salva arquivo atômico .inter (e retorna pandas df)
        dataset_name = self.config.get('dataset_name', 'realestate')
        output_dir = Path(self.config['output_path']) / dataset_name
        inter_file = output_dir / f"{dataset_name}.inter"

        pdf = self.save_inter_file(df, inter_file)

        # 9. Salva sessions para API (reutiliza pandas df)
        sessions_api_file = Path(self.config['output_path']) / 'sessions_for_api.parquet'
        self.save_sessions_for_api(pdf, sessions_api_file)

        log("" + "=" * 80)
        log(" PIPELINE COMPLETO!")
        log("=" * 80)
        log(f"Dataset: {dataset_name}")
        log(f"Arquivo: {inter_file}")
        log("Próximo passo:")
        log(f"  make benchmark --dataset {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description='Prepare RecBole dataset')
    parser.add_argument('--start-date', type=str, help='Override start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='Override end date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, help='Override output path')

    project_config = get_config()
    raw_data_config = project_config['raw_data']
    args = parser.parse_args()
    spark = make_spark()

    # Run pipeline
    pipeline = RecBoleDataPipeline(
        config={
            'output_path': args.output or raw_data_config['output_path'],
            'dataset_name': project_config['dataset'],
        },
        session_data_pipeline=SessionDataPipeline(
            spark=spark,
            output_path=args.output,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    )
    pipeline.run()
    spark.stop()


if __name__ == '__main__':
    main()

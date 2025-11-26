
from pyspark.sql import SparkSession

import config as cfg
from preprocessing import build_events_clean, build_model_df
from splits import save_splits
from utils.enviroment import events_processed_path
from utils.spark_session import make_spark


def load_raw_df(spark: SparkSession):
    data_path = events_processed_path()
    return spark.read.parquet(data_path)


def main():
    spark = make_spark()

    print("Lendo dados brutos...")
    df_raw = load_raw_df(spark)

    print("Rodando pipeline de limpeza (events_clean)...")
    events_clean = build_events_clean(df_raw)

    print("Salvando events_clean em parquet...")
    (
        events_clean.write
        .mode("overwrite")
        .parquet(cfg.CLEANED_EVENTS_PATH)
    )

    # 3) Construir model_df (SessionId, ItemId, Time, Date, ...)
    print("Construindo model_df...")
    model_df = build_model_df(events_clean)

    print("Salvando model_df em parquet...")
    (
        model_df.write
        .mode("overwrite")
        .parquet(cfg.MODEL_DATA_PATH)
    )

    # 4) Gerar splits temporais train/test
    print("Gerando splits temporais train/test...")
    save_splits(model_df, spark)

    print("Pipeline concluído com sucesso.")


if __name__ == "__main__":
    main()

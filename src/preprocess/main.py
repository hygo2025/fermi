from pyspark.sql import DataFrame
from pyspark.sql import SparkSession

from preprocessing import build_events_clean, build_model_df
from splits import save_splits
from utils.enviroment import get_path
from utils.paths import PathsEnum
from utils.spark_session import make_spark


def clean_events(spark: SparkSession, load_only: bool = False ) -> DataFrame:
    if load_only:
        print("Carregando events_clean de parquet...")
        return spark.read.parquet(get_path(PathsEnum.CLEAN_EVENTS))

    df_raw = spark.read.parquet(get_path(PathsEnum.EVENTS))
    print("Rodando pipeline de limpeza (events_clean)...")

    events_clean = build_events_clean(df_raw)
    print(events_clean.count())

    print("Salvando events_clean em parquet...")
    (
        events_clean
        .coalesce(8)
        .write
        .mode("overwrite")
        .parquet(get_path(PathsEnum.CLEAN_EVENTS))
    )

    return events_clean

def build_model(spark: SparkSession, events_clean: DataFrame, load_only: bool = False) -> DataFrame:
    if load_only:
        print("Carregando model_df de parquet...")
        return spark.read.parquet(get_path(PathsEnum.MODEL_DF))

    print("Construindo model_df...")
    model_df = build_model_df(events_clean)
    print("Salvando model_df em parquet...")
    (
        model_df
        .coalesce(8)
        .write
        .mode("overwrite")
        .parquet(get_path(PathsEnum.MODEL_DF))
    )

    return model_df

if __name__ == '__main__':
    spark = make_spark()
    df = spark.read.parquet(get_path(PathsEnum.LISTING))
    print(df.show(10, truncate=False))
    #
    # events_clean = clean_events(spark=spark, load_only=False)
    # model_df = build_model(spark=spark, events_clean=events_clean, load_only=False)
    #
    # print("Gerando splits temporais train/test...")
    # save_splits(model_df=model_df, split_path=get_path(PathsEnum.MODEL_SPLIT))
    #
    # split_0_path = f"{get_path(PathsEnum.MODEL_SPLIT)}/slice_0/test"
    # test_df = spark.read.option("header", True).csv(split_0_path)
    # session_count = test_df.select("SessionId").distinct().count()
    # print(f"Número de sessões no split de teste slice_0: {session_count}")
    #
    #
    # split_0_path = f"{get_path(PathsEnum.MODEL_SPLIT)}/slice_0/train"
    # train_df = spark.read.option("header", True).csv(split_0_path)
    # session_count = train_df.select("ItemId").distinct().count()
    # print(f"Número de sessões no split de treino slice_0: {session_count}")
    #
    # print("Pipeline concluído com sucesso.")

from datetime import timedelta

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from utils import config as cfg


def get_date_range(model_df: DataFrame):
    """
    Retorna (min_dt, max_dt) do DataFrame base.
    """
    row = model_df.agg(
        F.min("Date").alias("min_dt"),
        F.max("Date").alias("max_dt")
    ).collect()[0]

    return row["min_dt"], row["max_dt"]


def build_slices(min_dt):
    """
    Constrói a lista de janelas temporais (slices) com base em:
    - cfg.N_SLICES
    - cfg.SLICE_LEN_DAYS

    Cada slice é um dict com:
    - idx
    - slice_start, slice_end
    - train_start, train_end
    - test_day
    """
    slices = []
    current_start = min_dt

    for i in range(cfg.N_SLICES):
        slice_start = current_start
        slice_end = slice_start + timedelta(days=cfg.SLICE_LEN_DAYS - 1)
        train_start = slice_start
        train_end = slice_start + timedelta(days=cfg.SLICE_LEN_DAYS - 2)
        test_day = slice_end

        slices.append(
            {
                "idx": i,
                "slice_start": slice_start,
                "slice_end": slice_end,
                "train_start": train_start,
                "train_end": train_end,
                "test_day": test_day,
            }
        )

        current_start = current_start + timedelta(days=cfg.SLICE_LEN_DAYS)

    return slices


def save_splits(model_df: DataFrame, split_path: str = ""):
    """
    Gera e salva os splits de treino/teste em CSV, no formato:
    - SessionId, ItemId, Time

    Usa sliding window: vários slices, cada um com treino e teste separados.
    """
    model_df = model_df.persist()

    min_dt, max_dt = get_date_range(model_df)
    print(f"Date range in model_df: {min_dt} -> {max_dt}")

    slices = build_slices(min_dt)

    for s in slices:
        i = s["idx"]
        train_start = s["train_start"]
        train_end = s["train_end"]
        test_day = s["test_day"]

        print(f"Slice {i}: train {train_start}–{train_end}, test {test_day}")

        # Treino: intervalo de datas (5 dias)
        train_df = model_df.filter(
            (F.col("Date") >= F.lit(train_start))
            & (F.col("Date") <= F.lit(train_end))
        )

        # Teste: último dia do slice
        test_df_raw = model_df.filter(F.col("Date") == F.lit(test_day))

        # Opcional: manter apenas sessões que aparecem no treino
        train_sessions = train_df.select("SessionId").distinct()
        test_df = test_df_raw.join(train_sessions, "SessionId", "inner")

        # Seleciona colunas finais
        out_cols = ["SessionId", "ItemId", "Time"]

        train_out = train_df.select(*out_cols).orderBy("SessionId", "Time")
        test_out = test_df.select(*out_cols).orderBy("SessionId", "Time")

        # Caminhos de saída
        train_path = f"{split_path}/slice_{i}/train"
        test_path = f"{split_path}/slice_{i}/test"

        (
            train_out.write
            .mode("overwrite")
            .option("header", "true")
            .csv(train_path)
        )

        (
            test_out.write
            .mode("overwrite")
            .option("header", "true")
            .csv(test_path)
        )

        print(f"Saved slice {i}:")
        print(f"  train -> {train_path} - {train_out.count()} rows")
        print(f"  test  -> {test_path} - {test_out.count()} rows")

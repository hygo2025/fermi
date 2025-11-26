from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

import config as cfg


def cast_types(df: DataFrame) -> DataFrame:
    """
    Garante que colunas de data/hora estejam com tipos corretos.
    """
    return (
        df
        .withColumn("collector_ts", F.col("collector_ts").cast("timestamp"))
        .withColumn("event_ts", F.col("event_ts").cast("timestamp"))
        .withColumn("dt", F.col("dt").cast("date"))
    )


def filter_date_range(df: DataFrame) -> DataFrame:
    """
    Filtra o DataFrame para o intervalo [DATE_START, DATE_END].
    """
    return df.filter(
        (F.col("dt") >= F.lit(cfg.DATE_START)) &
        (F.col("dt") <= F.lit(cfg.DATE_END))
    )


def filter_events(df: DataFrame) -> DataFrame:
    """
    Mantém apenas eventos relevantes:
    - event_type em EVENT_TYPES_OF_INTEREST
    - listing_id e session_id não nulos
    """
    return (
        df
        .filter(F.col("event_type").isin(cfg.EVENT_TYPES_OF_INTEREST))
        .filter(F.col("listing_id").isNotNull())
        .filter(F.col("session_id").isNotNull())
    )


def truncate_sessions(df: DataFrame) -> DataFrame:
    """
    Ordena eventos por sessão e timestamp e trunca sessões
    com mais de MAX_SESSION_LENGTH eventos.
    """
    w = Window.partitionBy("session_id").orderBy(F.col("collector_ts"))

    df_ord = (
        df
        .withColumn("ts", F.col("collector_ts").cast("timestamp"))
        .withColumn("rn_in_session", F.row_number().over(w))
    )

    df_trunc = df_ord.filter(F.col("rn_in_session") <= cfg.MAX_SESSION_LENGTH)
    return df_trunc


def filter_short_sessions(df: DataFrame) -> DataFrame:
    """
    Remove sessões com menos de MIN_SESSION_LENGTH eventos.
    """
    sess_len = (
        df.groupBy("session_id")
        .agg(F.count("*").alias("session_len"))
    )

    valid_sessions = sess_len.filter(
        F.col("session_len") >= cfg.MIN_SESSION_LENGTH
    ).select("session_id")

    return df.join(valid_sessions, "session_id", "inner")


def filter_rare_items(df: DataFrame) -> DataFrame:
    """
    Remove items (listing_id) com menos de MIN_ITEM_INTERACTIONS interações.
    Depois de remover itens, é comum refiltrar sessões pequenas
    (isso é feito em outra função).
    """
    item_cnt = (
        df.groupBy("listing_id")
        .agg(F.count("*").alias("item_cnt"))
    )

    valid_items = item_cnt.filter(
        F.col("item_cnt") >= cfg.MIN_ITEM_INTERACTIONS
    ).select("listing_id")

    return df.join(valid_items, "listing_id", "inner")


def build_events_clean(df_raw: DataFrame) -> DataFrame:
    """
    Pipeline completo de limpeza, devolve o events_clean final.
    """
    df_cast = cast_types(df_raw)
    df_date = filter_date_range(df_cast)
    df_ev = filter_events(df_date)
    df_trunc = truncate_sessions(df_ev)
    df_sess = filter_short_sessions(df_trunc)
    df_items = filter_rare_items(df_sess)
    df_sess2 = filter_short_sessions(df_items)  # refiltra sessões após tirar itens raros

    return df_sess2


def build_model_df(events_clean: DataFrame) -> DataFrame:
    """
    Constrói o DataFrame final no formato padrão de sessão para recomendação:
    - SessionId, ItemId, Time, Date
    Também mantém UserId e BusinessType como extras.
    """
    model_df = (
        events_clean
        .select(
            F.col("session_id").alias("SessionId"),
            F.col("listing_id").alias("ItemId"),
            F.col("collector_ts").alias("Timestamp"),
            F.col("dt").alias("Date"),
            F.col("unique_user_id").alias("UserId"),
            F.col("business_type").alias("BusinessType"),
        )
        .withColumn("Time", F.col("Timestamp").cast("long"))
        .orderBy("SessionId", "Time")
    )

    return model_df

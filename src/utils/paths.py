from enum import Enum


class PathsEnum(Enum):
    EVENTS = "processed_data/events"
    LISTING = "processed_data/listings"
    ENRICHED_EVENTS = "processed_data/enriched_events"
    CLEAN_EVENTS = "models/processed/clean_events"
    MODEL_DF = "models/model_df"
    MODEL_SPLIT = "models/splits"
OUTPUT_BASE_PATH = "s3://meu-bucket/imoveisrec/"

# onde salvar o dataset limpo (todas as interações)
CLEANED_EVENTS_PATH = OUTPUT_BASE_PATH + "events_clean_parquet"
# onde salvar o dataset no formato SessionId, ItemId, Time
MODEL_DATA_PATH = OUTPUT_BASE_PATH + "model_df_parquet"
# onde salvar os CSVs de train/test
SPLITS_OUTPUT_PATH = OUTPUT_BASE_PATH + "splits/"

# -----------------------------
# Janela temporal usada
# -----------------------------
DATE_START = "2024-03-01"
DATE_END = "2024-03-30"

# -----------------------------
# Filtros de eventos
# -----------------------------
# tipos de evento que indicam interesse real no imóvel
EVENT_TYPES_OF_INTEREST = [
    "view_listing",
    "detail_view",
    "listing_click",
]

# -----------------------------
# Regras de sessão e itens
# -----------------------------
MAX_SESSION_LENGTH = 50     # tamanho máximo de sessão (número de eventos)
MIN_SESSION_LENGTH = 2      # mínimo de eventos por sessão
MIN_ITEM_INTERACTIONS = 5   # mínimo de interações por listing

# -----------------------------
# Splits temporais (sliding window)
# -----------------------------
N_SLICES = 5          # número de janelas
SLICE_LEN_DAYS = 6    # dias por janela (6 x 5 = 30)

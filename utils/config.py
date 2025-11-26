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
    "LeadClicked",
    "RankingClicked",
    #"GalleryClicked",
    "DecisionTreeFormClicked",
    "ShareClicked",
    "FavoriteClicked",
    #"RankingRendered",
    "LeadPanelClicked",
    "ListingRendered",
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

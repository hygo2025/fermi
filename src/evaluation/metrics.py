from typing import List, Optional, Set
import math


def rank_of_item(recommended: List[str], true_item: str) -> Optional[int]:
    """
    Retorna a posição (0-based) do true_item na lista recommended.
    Se não estiver presente, retorna None.
    """
    try:
        return recommended.index(true_item)
    except ValueError:
        return None


def hit_rate_at_k(recommended: List[str], true_item: str, k: int) -> float:
    """
    HitRate@K: 1 se o item verdadeiro estiver entre os K recomendados, 0 caso contrário.
    """
    r = rank_of_item(recommended[:k], true_item)
    return 1.0 if r is not None else 0.0


def mrr_at_k(recommended: List[str], true_item: str, k: int) -> float:
    """
    MRR@K: Reciprocal Rank truncado em K (0 se não recomendado).
    """
    r = rank_of_item(recommended[:k], true_item)
    if r is None:
        return 0.0
    return 1.0 / (r + 1)


def precision_at_k(recommended: List[str], true_items: Set[str], k: int) -> float:
    """
    Precision@K para um conjunto de itens verdadeiros (rest-of-session).
    """
    if k == 0:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for x in rec_k if x in true_items)
    return hits / k


def recall_at_k(recommended: List[str], true_items: Set[str], k: int) -> float:
    """
    Recall@K para um conjunto de itens verdadeiros.
    """
    if not true_items:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for x in rec_k if x in true_items)
    return hits / len(true_items)


def average_precision_at_k(recommended: List[str], true_items: Set[str], k: int) -> float:
    """
    AP@K (Average Precision): média das precisões nos pontos de acerto, truncado em K.
    """
    if not true_items:
        return 0.0

    ap = 0.0
    hits = 0
    for i, item in enumerate(recommended[:k]):
        if item in true_items:
            hits += 1
            precision_i = hits / (i + 1)
            ap += precision_i

    if hits == 0:
        return 0.0

    return ap / min(len(true_items), k)


def ndcg_at_k(recommended: List[str], true_items: Set[str], k: int) -> float:
    """
    NDCG@K assumindo ganho binário (1 se item relevante, 0 caso contrário).
    """
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in true_items:
            dcg += 1.0 / math.log2(i + 2)  # i+2 porque posição começa em 1

    # IDCG: ideal seria todos os relevantes no topo
    ideal_hits = min(len(true_items), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0

    return dcg / idcg

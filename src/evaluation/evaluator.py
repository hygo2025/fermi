from typing import Dict

import pandas as pd

from src.evaluation.metrics import (
    hit_rate_at_k,
    mrr_at_k,
    precision_at_k,
    recall_at_k,
    average_precision_at_k,
    ndcg_at_k,
)
from src.evaluation.models.baselines import NextItemRecommender


def evaluate_next_item(
        model: NextItemRecommender,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        k: int = 10
) -> Dict[str, float]:
    """
    Avaliação de next-item prediction.

    Para cada sessão no conjunto de teste:
      - ordena eventos por Time
      - para cada posição t >= 1:
          - histórico = itens até t-1
          - próximo item verdadeiro = item em t
          - gera recomendações para o histórico
          - atualiza métricas HitRate@K e MRR@K

    Retorna um dicionário com as métricas médias:
      - 'hitrate@K'
      - 'mrr@K'
    """
    # modelo é ajustado com train_df
    model.fit(train_df)

    # garante ordenação correta
    test_df = test_df.sort_values(by=["SessionId", "Time"])

    total_events = 0
    sum_hit = 0.0
    sum_mrr = 0.0

    # agrupa por sessão
    for session_id, group in test_df.groupby("SessionId"):
        items = group["ItemId"].astype(str).tolist()
        # se sessão tem só 1 evento, não dá pra fazer next-item
        if len(items) <= 1:
            continue

        # vamos avaliar do passo 1 até o penúltimo item
        for t in range(1, len(items)):
            history = items[:t]
            true_next = items[t]

            recs = model.recommend_next(history, k)

            sum_hit += hit_rate_at_k(recs, true_next, k)
            sum_mrr += mrr_at_k(recs, true_next, k)
            total_events += 1

    if total_events == 0:
        return {
            f"hitrate@{k}": 0.0,
            f"mrr@{k}": 0.0,
            "n_events": 0
        }

    return {
        f"hitrate@{k}": sum_hit / total_events,
        f"mrr@{k}": sum_mrr / total_events,
        "n_events": total_events
    }

def evaluate_rest_of_session(
    model: NextItemRecommender,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 10,
) -> Dict[str, float]:
    """
    Avaliação de REST-OF-SESSION.

    Para cada sessão no conjunto de teste:
      - ordena eventos por Time
      - para cada posição t >= 1:
          - histórico = itens até t-1
          - itens futuros verdadeiros = itens de t até o fim da sessão
          - gera recomendações para o histórico
          - atualiza métricas:
              - Precision@K
              - Recall@K
              - NDCG@K
              - MAP@K

    Retorna um dicionário com as métricas médias:
      - 'precision@K'
      - 'recall@K'
      - 'ndcg@K'
      - 'map@K'
      - 'n_cases' (número de prefixos avaliados)
    """
    # treina o modelo
    model.fit(train_df)

    # garante ordenação correta
    test_df = test_df.sort_values(by=["SessionId", "Time"])

    total_cases = 0
    sum_prec = 0.0
    sum_rec = 0.0
    sum_ndcg = 0.0
    sum_map = 0.0

    # agrupa por sessão
    for session_id, group in test_df.groupby("SessionId"):
        items = group["ItemId"].astype(str).tolist()

        # se sessão tem só 1 evento, não dá pra ter "rest-of-session"
        if len(items) <= 1:
            continue

        # vamos avaliar para cada prefixo da sessão
        for t in range(1, len(items)):
            history = items[:t]
            future_items = set(items[t:])  # conjunto dos itens futuros

            if not future_items:
                continue

            recs = model.recommend_next(history, k)

            sum_prec += precision_at_k(recs, future_items, k)
            sum_rec += recall_at_k(recs, future_items, k)
            sum_ndcg += ndcg_at_k(recs, future_items, k)
            sum_map += average_precision_at_k(recs, future_items, k)
            total_cases += 1

    if total_cases == 0:
        return {
            f"precision@{k}": 0.0,
            f"recall@{k}": 0.0,
            f"ndcg@{k}": 0.0,
            f"map@{k}": 0.0,
            "n_cases": 0,
        }

    return {
        f"precision@{k}": sum_prec / total_cases,
        f"recall@{k}": sum_rec / total_cases,
        f"ndcg@{k}": sum_ndcg / total_cases,
        f"map@{k}": sum_map / total_cases,
        "n_cases": total_cases,
    }

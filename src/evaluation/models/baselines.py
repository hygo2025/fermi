import random
from collections import Counter
from typing import List, Dict

import pandas as pd


class NextItemRecommender:
    """
    Interface básica para modelos de recomendação de próximo item.
    """
    def fit(self, train_df: pd.DataFrame):
        raise NotImplementedError

    def recommend_next(self, session_items: List[str], k: int) -> List[str]:
        raise NotImplementedError


class PopularityRecommender(NextItemRecommender):
    """
    Recomendador de popularidade global (não-personalizado).
    """
    def __init__(self):
        self.popular_items: List[str] = []

    def fit(self, train_df: pd.DataFrame):
        counts = Counter(train_df["ItemId"].tolist())
        self.popular_items = [item for item, _ in counts.most_common()]

    def recommend_next(self, session_items: List[str], k: int) -> List[str]:
        return self.popular_items[:k]


class SessionPopularityRecommender(NextItemRecommender):
    """
    "Session-popularity":
    recomenda os itens mais frequentes na sessão (histórico),
    com desempate pela popularidade global.
    """
    def __init__(self):
        self.global_popular: List[str] = []

    def fit(self, train_df: pd.DataFrame):
        counts = Counter(train_df["ItemId"].tolist())
        self.global_popular = [item for item, _ in counts.most_common()]

    def recommend_next(self, session_items: List[str], k: int) -> List[str]:
        if not session_items:
            # fallback para popularidade global se não houver histórico
            return self.global_popular[:k]

        # popularidade dentro da sessão
        sess_counts = Counter(session_items)
        # ordena por frequência na sessão e desempata por popularidade global
        # definimos uma chave de ordenação que usa a posição na lista global_popular
        global_rank: Dict[str, int] = {
            item: rank for rank, item in enumerate(self.global_popular)
        }

        sorted_items = sorted(
            sess_counts.keys(),
            key=lambda x: (-sess_counts[x], global_rank.get(x, 10**9))
        )

        # Se não houver itens suficientes, completa com global_popular
        recs = list(sorted_items)
        for item in self.global_popular:
            if item not in sess_counts and item not in recs:
                recs.append(item)
            if len(recs) >= k:
                break

        return recs[:k]


class RandomRecommender(NextItemRecommender):
    """
    Recomendador que sorteia itens uniformemente entre todos os itens vistos no treino.
    """
    def __init__(self, seed: int = 42):
        self.items: List[str] = []
        self.random = random.Random(seed)

    def fit(self, train_df: pd.DataFrame):
        self.items = sorted(set(train_df["ItemId"].astype(str).tolist()))

    def recommend_next(self, session_items: List[str], k: int) -> List[str]:
        if not self.items:
            return []
        # se tiver menos itens que k, faz sample com reposição
        if len(self.items) <= k:
            return self.random.sample(self.items, len(self.items))
        return self.random.sample(self.items, k)

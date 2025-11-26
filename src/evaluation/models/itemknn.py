from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import math
import pandas as pd

from src.evaluation.models.baselines import NextItemRecommender


class ItemKNNRecommender(NextItemRecommender):
    """
    ItemKNN para recomendação de próximo item em sessões.

    Ideia:
    - Construir uma matriz de coocorrência item-item a partir das sessões de treino.
    - Transformar em similaridade (estilo cosine) usando frequências.
    - Para recomendar:
        - pegar os últimos N itens do histórico da sessão
        - pegar vizinhos similares de cada um
        - somar scores (ponderando pela similaridade e, opcionalmente, pela recência)
    """

    def __init__(
        self,
        k_neighbors: int = 100,
        max_history: int = 5,
        use_recency_weight: bool = True,
    ):
        """
        :param k_neighbors: número máximo de vizinhos armazenados por item
        :param max_history: quantos itens recentes da sessão usar para recomendar
        :param use_recency_weight: se True, aplica peso maior para itens mais recentes
        """
        self.k_neighbors = k_neighbors
        self.max_history = max_history
        self.use_recency_weight = use_recency_weight

        # item -> lista de (neighbor_item, similarity)
        self.item_sims: Dict[str, List[Tuple[str, float]]] = {}

        # popularidade global (fallback)
        self.popular_items: List[str] = []

    def fit(self, train_df: pd.DataFrame):
        """
        Treina o modelo a partir do DataFrame de treino:
        colunas esperadas: SessionId, ItemId, Time
        """
        # Garantir tipos
        df = train_df.copy()
        df["SessionId"] = df["SessionId"].astype(str)
        df["ItemId"] = df["ItemId"].astype(str)

        # Popularidade global (fallback)
        item_counts = Counter(df["ItemId"].tolist())
        self.popular_items = [it for it, _ in item_counts.most_common()]

        # Agrupa por sessão para construir coocorrência
        df = df.sort_values(by=["SessionId", "Time"])
        sessions = df.groupby("SessionId")["ItemId"].apply(list)

        # Frequência de cada item (número de sessões em que aparece)
        item_session_freq = Counter()

        # Coocorrências item-item
        co_counts: Dict[str, Counter] = defaultdict(Counter)

        for sess_items in sessions:
            # usar itens únicos da sessão para coocorrência binária
            uniq_items = list(set(sess_items))
            # atualiza freq de presença em sessão
            for it in uniq_items:
                item_session_freq[it] += 1

            # atualiza coocorrências para pares (i, j), i != j
            for i in range(len(uniq_items)):
                it_i = uniq_items[i]
                for j in range(i + 1, len(uniq_items)):
                    it_j = uniq_items[j]
                    co_counts[it_i][it_j] += 1
                    co_counts[it_j][it_i] += 1  # simétrico

        # Constrói matriz de similaridade
        item_sims: Dict[str, List[Tuple[str, float]]] = {}

        for it_i, neighbors in co_counts.items():
            sims = []
            freq_i = item_session_freq[it_i]
            if freq_i == 0:
                continue

            for it_j, cij in neighbors.items():
                freq_j = item_session_freq[it_j]
                if freq_j == 0:
                    continue

                # Similaridade tipo cosine em espaço binário (itens vs sessões)
                sim = cij / math.sqrt(freq_i * freq_j)

                if sim > 0:
                    sims.append((it_j, sim))

            # Ordena vizinhos por similaridade desc e trunca
            sims.sort(key=lambda x: x[1], reverse=True)
            if self.k_neighbors is not None:
                sims = sims[: self.k_neighbors]

            item_sims[it_i] = sims

        self.item_sims = item_sims

    def _score_candidates_from_history(self, history: List[str], k: int) -> List[Tuple[str, float]]:
        """
        Dado um histórico de itens, calcula score dos candidatos.
        """
        if not history:
            return []

        # Considera só os últimos max_history itens
        recent_history = history[-self.max_history :]

        scores = defaultdict(float)
        used_items = set(history)  # para filtrar depois

        # iterar do último para o primeiro no histórico recente
        for idx, item in enumerate(reversed(recent_history)):
            neighbors = self.item_sims.get(item)
            if not neighbors:
                continue

            if self.use_recency_weight:
                # mais recente = peso maior
                # idx = 0 é o item mais recente
                weight = 1.0 / (idx + 1)
            else:
                weight = 1.0

            for neigh, sim in neighbors:
                if neigh in used_items:
                    continue
                scores[neigh] += sim * weight

        # ordena candidatos
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # se faltar candidato, completa com popularidade global
        if len(ranked) < k:
            existing = set(item for item, _ in ranked)
            for pop_item in self.popular_items:
                if pop_item not in existing and pop_item not in used_items:
                    ranked.append((pop_item, 0.0))
                    if len(ranked) >= k:
                        break

        return ranked[:k]

    def recommend_next(self, session_items: List[str], k: int) -> List[str]:
        """
        Recomendação de próximo item a partir do histórico da sessão.
        """
        # fallback: se não tiver histórico, devolve só popularidade global
        if not session_items:
            return self.popular_items[:k]

        scored = self._score_candidates_from_history(session_items, k)
        recs = [it for it, _ in scored]

        # se ainda sobrar vaga, completa com popularidade global
        if len(recs) < k:
            used = set(recs) | set(session_items)
            for pop_item in self.popular_items:
                if pop_item not in used:
                    recs.append(pop_item)
                if len(recs) >= k:
                    break

        return recs[:k]

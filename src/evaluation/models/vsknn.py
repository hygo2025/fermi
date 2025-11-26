import math
from collections import defaultdict, Counter
from typing import List, Dict

import pandas as pd

from src.evaluation.models.baselines import NextItemRecommender


class VSKNNRecommender(NextItemRecommender):
    """
    Vector-Enhanced Session-based KNN (V-SKNN).

    Ideia principal:
      - Parecido com SessionKNN, mas:
        * usa pesos por posição dos itens no histórico (recency maior peso)
        * acumula um "score de candidato" por sessão baseado nos itens em comum
        * usa esse score como similaridade sessão-sessão (com normalizações simples)
      - Em geral é mais forte que o SKNN simples.
    """

    def __init__(
        self,
        n_neighbors: int = 100,
        max_history: int = 20,
        max_sessions_per_item: int = 1000,
        max_candidate_sessions: int = 2000,
        use_session_length_normalization: bool = True,
        use_recency_weight_session: bool = True,
        use_recency_weight_items: bool = True,
    ):
        """
        :param n_neighbors: número de sessões vizinhas mais similares
        :param max_history: quantos itens do histórico considerar (da cauda)
        :param max_sessions_per_item: máximo de sessões associadas a cada item
        :param max_candidate_sessions: limite de sessões candidatas na busca
        :param use_session_length_normalization: normaliza score por tamanho da sessão
        :param use_recency_weight_session: se True, dá mais peso a sessões mais recentes
        :param use_recency_weight_items: se True, pondera itens do histórico pela posição
        """
        self.n_neighbors = n_neighbors
        self.max_history = max_history
        self.max_sessions_per_item = max_sessions_per_item
        self.max_candidate_sessions = max_candidate_sessions
        self.use_session_length_normalization = use_session_length_normalization
        self.use_recency_weight_session = use_recency_weight_session
        self.use_recency_weight_items = use_recency_weight_items

        # Estruturas internas
        self.sessions: List[List[str]] = []
        self.session_lengths: List[int] = []
        self.session_recency: List[int] = []
        self.item_sessions: Dict[str, List[int]] = defaultdict(list)

        # fallback de popularidade global
        self.global_popular: List[str] = []

    # ------------------------------------------------------------------
    # Treino
    # ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame):
        df = train_df.copy()
        df["SessionId"] = df["SessionId"].astype(str)
        df["ItemId"] = df["ItemId"].astype(str)

        # popularidade global
        counts = Counter(df["ItemId"].tolist())
        self.global_popular = [it for it, _ in counts.most_common()]

        # ordenar e agrupar
        df = df.sort_values(by=["SessionId", "Time"])
        grouped = df.groupby("SessionId")["ItemId"].apply(list)

        self.sessions = []
        self.session_lengths = []
        self.session_recency = []
        self.item_sessions = defaultdict(list)

        for idx, (_, items) in enumerate(grouped.items()):
            sess_items = items
            self.sessions.append(sess_items)
            self.session_lengths.append(len(sess_items))
            self.session_recency.append(idx)

            # mapeia item -> lista de sessões (limitada)
            for it in set(sess_items):
                if len(self.item_sessions[it]) < self.max_sessions_per_item:
                    self.item_sessions[it].append(idx)

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------
    def _get_candidate_sessions(self, history: List[str]) -> Dict[int, float]:
        """
        A partir do histórico, retorna:
          sessão_id -> score de candidato (baseado em itens em comum).
        """
        if not history:
            return {}

        recent_history = history[-self.max_history :]
        # vamos considerar posição a partir do fim (item mais recente = pos 0)
        candidate_scores = Counter()

        for idx, item in enumerate(reversed(recent_history)):
            sess_list = self.item_sessions.get(item)
            if not sess_list:
                continue

            # peso pelo recency do item no histórico
            if self.use_recency_weight_items:
                # mais recente -> peso maior
                w_item = 1.0 / (idx + 1)
            else:
                w_item = 1.0

            for s_id in sess_list:
                candidate_scores[s_id] += w_item

        if not candidate_scores:
            return {}

        # ordenar por score, e como desempate, recência da sessão
        candidates = list(candidate_scores.items())
        candidates.sort(
            key=lambda x: (x[1], self.session_recency[x[0]]),
            reverse=True,
        )

        if self.max_candidate_sessions is not None:
            candidates = candidates[: self.max_candidate_sessions]

        return dict(candidates)

    # ------------------------------------------------------------------
    def _compute_session_similarity(self, candidate_scores: Dict[int, float]) -> Dict[int, float]:
        """
        Converte scores de candidatos em similaridades sessão-sessão,
        aplicando normalizações e pesos de recência.
        """
        if not candidate_scores:
            return {}

        max_recency = max(self.session_recency) if self.session_recency else 1
        sims: Dict[int, float] = {}

        for s_id, base_score in candidate_scores.items():
            sim = base_score

            # normalização pelo tamanho da sessão vizinha
            if self.use_session_length_normalization:
                length = self.session_lengths[s_id]
                if length > 0:
                    sim = sim / math.log2(length + 1.0)

            # peso de recência da sessão no treino
            if self.use_recency_weight_session:
                rec = self.session_recency[s_id]
                rec_weight = 1.0 + (rec / (max_recency + 1e-9))
                sim *= rec_weight

            if sim > 0:
                sims[s_id] = sim

        return sims

    # ------------------------------------------------------------------
    # Recomendação
    # ------------------------------------------------------------------
    def recommend_next(self, session_items: List[str], k: int) -> List[str]:
        if not self.sessions:
            return self.global_popular[:k]

        if not session_items:
            return self.global_popular[:k]

        candidate_scores = self._get_candidate_sessions(session_items)
        if not candidate_scores:
            return self.global_popular[:k]

        sims = self._compute_session_similarity(candidate_scores)
        if not sims:
            return self.global_popular[:k]

        # pega top-n vizinhos
        neighbors = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        neighbors = neighbors[: self.n_neighbors]

        # acumula scores nos itens dos vizinhos
        scores = defaultdict(float)
        history_set = set(session_items)

        for s_id, sim in neighbors:
            sess_items = self.sessions[s_id]
            for it in sess_items:
                if it in history_set:
                    continue
                scores[it] += sim

        if not scores:
            # fallback: popularidade global
            recs = []
            for it in self.global_popular:
                if it not in history_set:
                    recs.append(it)
                if len(recs) >= k:
                    break
            return recs

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        recs = [it for it, _ in ranked[:k]]

        # completa com popularidade se faltar
        if len(recs) < k:
            used = set(recs) | history_set
            for it in self.global_popular:
                if it not in used:
                    recs.append(it)
                if len(recs) >= k:
                    break

        return recs[:k]

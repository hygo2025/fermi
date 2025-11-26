import math
from collections import defaultdict, Counter
from typing import List, Dict

import pandas as pd

from src.evaluation.models.baselines import NextItemRecommender


class SessionKNNRecommender(NextItemRecommender):
    """
    Session-based KNN (SKNN) para recomendação de próximo item.

    Ideia:
    - Cada sessão de treino é um "vizinho em potencial".
    - Para uma sessão alvo (histórico no teste), encontramos sessões
      que compartilham itens com ela.
    - Calculamos similaridade sessão-sessão.
    - Itens nos vizinhos são pontuados proporcionalmente à similaridade.

    Similaridade usada: tipo cosine em espaço binário de itens:
        sim(s, t) = |I_s ∩ I_t| / sqrt(|I_s| * |I_t|)
    """

    def __init__(
        self,
        n_neighbors: int = 100,
        max_history: int = 20,
        max_sessions_per_item: int = 1000,
        max_candidate_sessions: int = 2000,
        use_session_length_normalization: bool = True,
        use_recency_weight: bool = True,
    ):
        """
        :param n_neighbors: número de sessões vizinhas mais similares
                            a considerar na recomendação
        :param max_history: quantos itens do histórico considerar (da cauda)
        :param max_sessions_per_item: máximo de sessões por item (para limitar memória)
        :param max_candidate_sessions: máximo de sessões candidatas por recomendação
        :param use_session_length_normalization: se True, normaliza sim por tamanhos das sessões
        :param use_recency_weight: se True, dá mais peso para sessões mais recentes no treino
        """
        self.n_neighbors = n_neighbors
        self.max_history = max_history
        self.max_sessions_per_item = max_sessions_per_item
        self.max_candidate_sessions = max_candidate_sessions
        self.use_session_length_normalization = use_session_length_normalization
        self.use_recency_weight = use_recency_weight

        # Índices internos
        # lista de sessões (em ordem de "tempo" de treino)
        self.sessions: List[List[str]] = []
        # tamanho das sessões
        self.session_lengths: List[int] = []
        # item -> lista de ids de sessão onde aparece
        self.item_sessions: Dict[str, List[int]] = {}
        # sessão_id interno -> "recência" (índice crescente)
        self.session_recency: List[int] = []

        # popularidade global (fallback)
        self.global_popular: List[str] = []

    def fit(self, train_df: pd.DataFrame):
        """
        Treina o SKNN a partir do DataFrame de treino:
        colunas esperadas: SessionId, ItemId, Time
        """
        df = train_df.copy()
        df["SessionId"] = df["SessionId"].astype(str)
        df["ItemId"] = df["ItemId"].astype(str)

        # popularidade global (fallback)
        counts = Counter(df["ItemId"].tolist())
        self.global_popular = [it for it, _ in counts.most_common()]

        # ordenar por SessionId, Time
        df = df.sort_values(by=["SessionId", "Time"])

        # agrupar itens por sessão
        grouped = df.groupby("SessionId")["ItemId"].apply(list)

        self.sessions = []
        self.session_lengths = []
        self.session_recency = []
        self.item_sessions = defaultdict(list)

        # cada sessão vai receber um id interno incremental (recente = id maior)
        for idx, (_, items) in enumerate(grouped.items()):
            # pode opcionalmente tirar repetições dentro da sessão,
            # mas em geral mantemos a ordem original (faz sentido para SKNN)
            sess_items = items
            self.sessions.append(sess_items)
            self.session_lengths.append(len(sess_items))
            self.session_recency.append(idx)  # idx maior = mais recente

            # atualiza índice item -> sessões
            # (limitando número de sessões por item)
            unique_items = set(sess_items)
            for it in unique_items:
                if len(self.item_sessions[it]) < self.max_sessions_per_item:
                    self.item_sessions[it].append(idx)

    def _find_candidate_sessions(self, history: List[str]) -> List[int]:
        """
        A partir do histórico, encontra sessões candidatas (ids internos).
        """
        if not history:
            return []

        # considerar apenas últimos max_history itens
        recent_items = history[-self.max_history :]

        candidate_counts = Counter()

        for it in set(recent_items):
            sess_list = self.item_sessions.get(it)
            if not sess_list:
                continue
            for s_id in sess_list:
                candidate_counts[s_id] += 1

        if not candidate_counts:
            return []

        # ordenar candidatos por:
        #   - quantos itens em comum com o histórico (desc)
        #   - recência da sessão (desc)
        candidates = list(candidate_counts.items())
        candidates.sort(
            key=lambda x: (x[1], self.session_recency[x[0]]),
            reverse=True,
        )

        # limitar número de sessões candidatas
        if self.max_candidate_sessions is not None:
            candidates = candidates[: self.max_candidate_sessions]

        candidate_ids = [s_id for s_id, _ in candidates]
        return candidate_ids

    def _compute_similarity(
        self, history_items: List[str], sess_items: List[str]
    ) -> float:
        """
        Similaridade sessão-sessão (histórico vs sessão vizinha).
        Usando algo tipo cosine binário:
            sim = |I_h ∩ I_s| / sqrt(|I_h| * |I_s|)
        """
        set_h = set(history_items)
        set_s = set(sess_items)
        inter_size = len(set_h & set_s)

        if inter_size == 0:
            return 0.0

        if self.use_session_length_normalization:
            denom = math.sqrt(len(set_h) * len(set_s))
            if denom == 0:
                return 0.0
            return inter_size / denom
        else:
            # só conta interseção (menos sofisticado)
            return float(inter_size)

    def recommend_next(self, session_items: List[str], k: int) -> List[str]:
        """
        Recomendação de próximo item para uma sessão de teste (histórico).
        """
        if not self.sessions:
            # fallback se não foi treinado
            return self.global_popular[:k]

        if not session_items:
            # se não tem histórico, cai pra popularidade global
            return self.global_popular[:k]

        # encontra sessões candidatas
        candidate_ids = self._find_candidate_sessions(session_items)
        if not candidate_ids:
            # fallback
            return self.global_popular[:k]

        # calcula similaridade com cada candidato
        sims: Dict[int, float] = {}
        # recência máxima (para peso)
        max_recency = max(self.session_recency) if self.session_recency else 1

        for s_id in candidate_ids:
            sess_items = self.sessions[s_id]
            sim = self._compute_similarity(session_items, sess_items)
            if sim <= 0:
                continue

            if self.use_recency_weight:
                rec = self.session_recency[s_id]
                # quanto mais recente, maior o peso (normalização simples)
                rec_weight = 1.0 + (rec / (max_recency + 1e-9))
                sim *= rec_weight

            sims[s_id] = sim

        if not sims:
            return self.global_popular[:k]

        # pega top-n vizinhos
        neighbors = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        neighbors = neighbors[: self.n_neighbors]

        # acumula scores de itens nos vizinhos
        scores = defaultdict(float)
        history_set = set(session_items)

        for s_id, sim in neighbors:
            sess_items = self.sessions[s_id]
            for it in sess_items:
                if it in history_set:
                    continue
                scores[it] += sim

        if not scores:
            # se não temos nenhum candidato, volta pra popularidade global
            return self.global_popular[:k]

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        recs = [it for it, _ in ranked[:k]]

        # completa com popularidade global se faltar item
        if len(recs) < k:
            used = set(recs) | history_set
            for it in self.global_popular:
                if it not in used:
                    recs.append(it)
                if len(recs) >= k:
                    break

        return recs[:k]

from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import math
import pandas as pd

from src.evaluation.models.baselines import NextItemRecommender


# ======================================================================
# 1) MarkovRecommender - Cadeia de Markov de primeira ordem
# ======================================================================

class MarkovRecommender(NextItemRecommender):
    """
    Modelo de Cadeia de Markov de primeira ordem para próximo item.

    Ideia:
      - Aprende transições de item_i -> item_j a partir de pares consecutivos
        em cada sessão de treino.
      - Para recomendar, pega o último item do histórico e recomenda os itens
        com maior probabilidade de transição.
    """

    def __init__(self):
        # item -> lista de (next_item, probabilidade)
        self.transitions: Dict[str, List[Tuple[str, float]]] = {}
        # fallback de popularidade global
        self.global_popular: List[str] = []

    def fit(self, train_df: pd.DataFrame):
        df = train_df.copy()
        df["SessionId"] = df["SessionId"].astype(str)
        df["ItemId"] = df["ItemId"].astype(str)

        # popularidade global para fallback
        counts = Counter(df["ItemId"].tolist())
        self.global_popular = [it for it, _ in counts.most_common()]

        # ordenar por sessão e tempo
        df = df.sort_values(by=["SessionId", "Time"])
        grouped = df.groupby("SessionId")["ItemId"].apply(list)

        # contar transições
        trans_counts: Dict[str, Counter] = defaultdict(Counter)

        for _, items in grouped.items():
            if len(items) < 2:
                continue
            for t in range(len(items) - 1):
                i = items[t]
                j = items[t + 1]
                trans_counts[i][j] += 1

        # normalizar em probabilidades
        self.transitions = {}
        for i, cnts in trans_counts.items():
            total = sum(cnts.values())
            if total == 0:
                continue
            # lista (j, prob)
            self.transitions[i] = [(j, c / total) for j, c in cnts.items()]
            # ordena desc
            self.transitions[i].sort(key=lambda x: x[1], reverse=True)

    def recommend_next(self, session_items: List[str], k: int) -> List[str]:
        # se não tem histórico, volta pra popularidade global
        if not session_items:
            return self.global_popular[:k]

        last_item = session_items[-1]

        candidates: List[Tuple[str, float]] = []
        if last_item in self.transitions:
            candidates = self.transitions[last_item][:]

        # extrai só os itens
        recs = [it for it, _ in candidates]

        # completa com popularidade global se precisar
        used = set(recs) | set(session_items)
        for it in self.global_popular:
            if len(recs) >= k:
                break
            if it not in used:
                recs.append(it)

        return recs[:k]


# ======================================================================
# 2) SequentialRulesRecommender - Regras sequenciais
# ======================================================================

class SequentialRulesRecommender(NextItemRecommender):
    """
    Modelo de Sequential Rules para próximo item.

    Ideia:
      - Para cada sessão de treino, para todos os pares (i_t, i_{t+Δ}) com Δ>0
        até um limite max_steps, cria uma "regra" i_t -> i_{t+Δ} com peso
        decaindo com a distância.
      - Na recomendação, usa os últimos max_history itens do histórico,
        combina as regras de todos eles, acumulando scores nos consequentes.
    """

    def __init__(
        self,
        max_history: int = 10,
        max_steps_ahead: int = 5,
        decay: float = 0.7,
        min_score: float = 0.0,
    ):
        """
        :param max_history: quantos itens recentes do histórico considerar
        :param max_steps_ahead: quantos passos à frente considerar na mesma sessão
        :param decay: fator de decaimento por distância (ex: 0.7**(delta-1))
        :param min_score: limiar mínimo de score para guardar uma regra
        """
        self.max_history = max_history
        self.max_steps_ahead = max_steps_ahead
        self.decay = decay
        self.min_score = min_score

        # regras: antecedente -> dict(consequente -> score)
        self.rules: Dict[str, Dict[str, float]] = {}
        self.global_popular: List[str] = []

    def fit(self, train_df: pd.DataFrame):
        df = train_df.copy()
        df["SessionId"] = df["SessionId"].astype(str)
        df["ItemId"] = df["ItemId"].astype(str)

        # popularidade global para fallback
        counts = Counter(df["ItemId"].tolist())
        self.global_popular = [it for it, _ in counts.most_common()]

        # ordenar por sessão e tempo
        df = df.sort_values(by=["SessionId", "Time"])
        grouped = df.groupby("SessionId")["ItemId"].apply(list)

        rule_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for _, items in grouped.items():
            n = len(items)
            if n < 2:
                continue
            # para cada posição i, olhar até max_steps_ahead à frente
            for i in range(n - 1):
                antecedent = items[i]
                for j in range(i + 1, min(n, i + 1 + self.max_steps_ahead)):
                    consequent = items[j]
                    delta = j - i
                    # peso decai com a distância
                    w = self.decay ** (delta - 1)
                    rule_scores[antecedent][consequent] += w

        # aplica min_score e transforma em dict final
        self.rules = {}
        for ant, cons_dict in rule_scores.items():
            filtered = {
                c: s for c, s in cons_dict.items() if s >= self.min_score
            }
            if not filtered:
                continue
            self.rules[ant] = filtered

    def recommend_next(self, session_items: List[str], k: int) -> List[str]:
        if not session_items:
            return self.global_popular[:k]

        history = session_items[-self.max_history :]
        scores: Dict[str, float] = defaultdict(float)
        history_set = set(history)

        # iterar da cauda pra cabeça, dando mais peso aos itens mais recentes
        for pos, item in enumerate(reversed(history)):
            # pos=0 é o mais recente
            recency_weight = 1.0 / (pos + 1)
            conseq_dict = self.rules.get(item)
            if not conseq_dict:
                continue
            for c, s in conseq_dict.items():
                if c in history_set:
                    continue
                scores[c] += s * recency_weight

        if not scores:
            # fallback pra popularidade global
            recs = []
            for it in self.global_popular:
                if it not in history_set:
                    recs.append(it)
                if len(recs) >= k:
                    break
            return recs

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        recs = [it for it, _ in ranked[:k]]

        # completa com popularidade global se não tiver k itens
        if len(recs) < k:
            used = set(recs) | history_set
            for it in self.global_popular:
                if it not in used:
                    recs.append(it)
                if len(recs) >= k:
                    break

        return recs[:k]

class SecondOrderMarkovRecommender(NextItemRecommender):
    """
    Cadeia de Markov de segunda ordem.

    Usa transições do tipo:
        (i_{t-2}, i_{t-1}) -> i_t

    Se o histórico tiver apenas 1 item, cai para um modelo de primeira ordem
    interno (construído com os mesmos dados).
    """

    def __init__(self):
        # (prev2, prev1) -> lista de (next_item, prob)
        self.second_order: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}

        # Modelo de 1a ordem como fallback
        self.first_order: Dict[str, List[Tuple[str, float]]] = {}

        # popularidade global
        self.global_popular: List[str] = []

    def fit(self, train_df: pd.DataFrame):
        df = train_df.copy()
        df["SessionId"] = df["SessionId"].astype(str)
        df["ItemId"] = df["ItemId"].astype(str)

        # popularidade global
        counts = Counter(df["ItemId"].tolist())
        self.global_popular = [it for it, _ in counts.most_common()]

        # ordenar
        df = df.sort_values(by=["SessionId", "Time"])
        grouped = df.groupby("SessionId")["ItemId"].apply(list)

        # contagens de 1a e 2a ordem
        first_counts: Dict[str, Counter] = defaultdict(Counter)
        second_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)

        for _, items in grouped.items():
            n = len(items)
            if n < 2:
                continue
            # primeira ordem
            for t in range(n - 1):
                i = items[t]
                j = items[t + 1]
                first_counts[i][j] += 1

            # segunda ordem
            if n < 3:
                continue
            for t in range(2, n):
                prev2 = items[t - 2]
                prev1 = items[t - 1]
                cur = items[t]
                second_counts[(prev2, prev1)][cur] += 1

        # normaliza 1a ordem
        self.first_order = {}
        for i, cnts in first_counts.items():
            total = sum(cnts.values())
            if total == 0:
                continue
            lst = [(j, c / total) for j, c in cnts.items()]
            lst.sort(key=lambda x: x[1], reverse=True)
            self.first_order[i] = lst

        # normaliza 2a ordem
        self.second_order = {}
        for state, cnts in second_counts.items():
            total = sum(cnts.values())
            if total == 0:
                continue
            lst = [(j, c / total) for j, c in cnts.items()]
            lst.sort(key=lambda x: x[1], reverse=True)
            self.second_order[state] = lst

    def recommend_next(self, session_items: List[str], k: int) -> List[str]:
        if not session_items:
            return self.global_popular[:k]

        # tenta segunda ordem se tiver >= 2 itens
        if len(session_items) >= 2:
            prev1 = session_items[-1]
            prev2 = session_items[-2]
            state = (prev2, prev1)

            if state in self.second_order:
                candidates = self.second_order[state][:k]
                recs = [it for it, _ in candidates]
                # completa com fallback se faltar
                used = set(recs) | set(session_items)
                for it in self.global_popular:
                    if len(recs) >= k:
                        break
                    if it not in used:
                        recs.append(it)
                return recs

            # se não achou estado de 2a ordem, tenta 1a ordem
            if prev1 in self.first_order:
                candidates = self.first_order[prev1][:k]
                recs = [it for it, _ in candidates]
                used = set(recs) | set(session_items)
                for it in self.global_popular:
                    if len(recs) >= k:
                        break
                    if it not in used:
                        recs.append(it)
                return recs

        # se histórico é curto ou nenhum estado conhecido, cai em popularidade
        return self.global_popular[:k]
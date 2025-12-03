"""
Custom Metrics for Session-Based Recommendation
Following Domingues et al. (2024) methodology

Implements two evaluation scenarios:
1. Next Item Prediction - HitRate@K, MRR@K, Coverage@K, Popularity@K
2. Rest of Session - Precision@K, Recall@K, NDCG@K, MAP@K
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter


class SessionBasedMetrics:
    """Métricas para avaliação de recomendação baseada em sessão"""
    
    def __init__(self, k_values: List[int] = [5, 10, 20]):
        """
        Args:
            k_values: Lista de valores K para top-K (default: [5, 10, 20])
        """
        self.k_values = k_values
        self.item_popularity = None
        self.catalog_size = None
    
    def set_item_popularity(self, train_interactions: List[int]):
        """
        Calcula popularidade dos itens a partir dos dados de treino
        
        Args:
            train_interactions: Lista de item_ids do conjunto de treino
        """
        self.item_popularity = Counter(train_interactions)
        self.catalog_size = len(set(train_interactions))
    
    # ========================================================================
    # NEXT ITEM PREDICTION METRICS
    # ========================================================================
    
    def hit_rate(self, predictions: List[List[int]], targets: List[int], k: int) -> float:
        """
        HitRate@K - Taxa de acerto no top-K
        
        Args:
            predictions: Lista de listas de itens recomendados (top-K) para cada sessão
            targets: Lista de itens verdadeiros (próximo item)
            k: Tamanho da lista de recomendação
            
        Returns:
            HitRate@K (0-1)
        """
        hits = 0
        for pred, target in zip(predictions, targets):
            if target in pred[:k]:
                hits += 1
        return hits / len(targets) if len(targets) > 0 else 0.0
    
    def mrr(self, predictions: List[List[int]], targets: List[int], k: int) -> float:
        """
        MRR@K - Mean Reciprocal Rank
        
        Args:
            predictions: Lista de listas de itens recomendados (top-K)
            targets: Lista de itens verdadeiros
            k: Tamanho da lista de recomendação
            
        Returns:
            MRR@K
        """
        reciprocal_ranks = []
        for pred, target in zip(predictions, targets):
            pred_k = pred[:k]
            if target in pred_k:
                rank = pred_k.index(target) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        return np.mean(reciprocal_ranks) if len(reciprocal_ranks) > 0 else 0.0
    
    def coverage(self, predictions: List[List[int]], k: int) -> float:
        """
        Coverage@K - Proporção do catálogo recomendado
        
        Args:
            predictions: Lista de listas de itens recomendados
            k: Tamanho da lista de recomendação
            
        Returns:
            Coverage@K (0-1)
        """
        if self.catalog_size is None:
            raise ValueError("Item popularity not set. Call set_item_popularity() first.")
        
        recommended_items = set()
        for pred in predictions:
            recommended_items.update(pred[:k])
        
        return len(recommended_items) / self.catalog_size
    
    def popularity_bias(self, predictions: List[List[int]], k: int) -> float:
        """
        Popularity@K - Viés para itens populares (APC - Average Percentage of Coverage)
        
        Mede quanto os itens recomendados são populares em relação ao treino.
        Valores altos indicam viés para itens populares.
        
        Args:
            predictions: Lista de listas de itens recomendados
            k: Tamanho da lista de recomendação
            
        Returns:
            Popularity@K (0-1)
        """
        if self.item_popularity is None:
            raise ValueError("Item popularity not set. Call set_item_popularity() first.")
        
        total_popularity = sum(self.item_popularity.values())
        
        recommended_popularity = 0
        count = 0
        for pred in predictions:
            for item in pred[:k]:
                recommended_popularity += self.item_popularity.get(item, 0)
                count += 1
        
        if count == 0:
            return 0.0
        
        avg_recommended_popularity = recommended_popularity / count
        avg_catalog_popularity = total_popularity / self.catalog_size
        
        return avg_recommended_popularity / avg_catalog_popularity if avg_catalog_popularity > 0 else 0.0
    
    # ========================================================================
    # REST OF SESSION METRICS
    # ========================================================================
    
    def precision_at_k(self, predictions: List[List[int]], targets: List[List[int]], k: int) -> float:
        """
        Precision@K - Proporção de itens relevantes no top-K
        
        Args:
            predictions: Lista de listas de itens recomendados
            targets: Lista de listas de itens relevantes (resto da sessão)
            k: Tamanho da lista de recomendação
            
        Returns:
            Precision@K
        """
        precisions = []
        for pred, target in zip(predictions, targets):
            pred_k = pred[:k]
            target_set = set(target)
            
            relevant_in_topk = len([item for item in pred_k if item in target_set])
            precisions.append(relevant_in_topk / k if k > 0 else 0.0)
        
        return np.mean(precisions) if len(precisions) > 0 else 0.0
    
    def recall_at_k(self, predictions: List[List[int]], targets: List[List[int]], k: int) -> float:
        """
        Recall@K - Proporção de itens relevantes recuperados
        
        Args:
            predictions: Lista de listas de itens recomendados
            targets: Lista de listas de itens relevantes
            k: Tamanho da lista de recomendação
            
        Returns:
            Recall@K
        """
        recalls = []
        for pred, target in zip(predictions, targets):
            if len(target) == 0:
                continue
                
            pred_k = pred[:k]
            target_set = set(target)
            
            relevant_in_topk = len([item for item in pred_k if item in target_set])
            recalls.append(relevant_in_topk / len(target))
        
        return np.mean(recalls) if len(recalls) > 0 else 0.0
    
    def ndcg_at_k(self, predictions: List[List[int]], targets: List[List[int]], k: int) -> float:
        """
        NDCG@K - Normalized Discounted Cumulative Gain
        
        Args:
            predictions: Lista de listas de itens recomendados
            targets: Lista de listas de itens relevantes
            k: Tamanho da lista de recomendação
            
        Returns:
            NDCG@K
        """
        ndcgs = []
        for pred, target in zip(predictions, targets):
            if len(target) == 0:
                continue
            
            pred_k = pred[:k]
            target_set = set(target)
            
            # DCG
            dcg = 0.0
            for i, item in enumerate(pred_k):
                if item in target_set:
                    dcg += 1.0 / np.log2(i + 2)  # i+2 porque índice começa em 0
            
            # IDCG (ideal DCG - todos relevantes nas primeiras posições)
            idcg = 0.0
            for i in range(min(len(target), k)):
                idcg += 1.0 / np.log2(i + 2)
            
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        
        return np.mean(ndcgs) if len(ndcgs) > 0 else 0.0
    
    def map_at_k(self, predictions: List[List[int]], targets: List[List[int]], k: int) -> float:
        """
        MAP@K - Mean Average Precision
        
        Args:
            predictions: Lista de listas de itens recomendados
            targets: Lista de listas de itens relevantes
            k: Tamanho da lista de recomendação
            
        Returns:
            MAP@K
        """
        aps = []
        for pred, target in zip(predictions, targets):
            if len(target) == 0:
                continue
            
            pred_k = pred[:k]
            target_set = set(target)
            
            # Average Precision
            hits = 0
            sum_precisions = 0.0
            for i, item in enumerate(pred_k):
                if item in target_set:
                    hits += 1
                    precision_at_i = hits / (i + 1)
                    sum_precisions += precision_at_i
            
            ap = sum_precisions / min(len(target), k) if len(target) > 0 else 0.0
            aps.append(ap)
        
        return np.mean(aps) if len(aps) > 0 else 0.0
    
    # ========================================================================
    # EVALUATION WRAPPER
    # ========================================================================
    
    def evaluate_next_item(self, predictions: List[List[int]], targets: List[int]) -> Dict[str, float]:
        """
        Avalia cenário de Next Item Prediction
        
        Args:
            predictions: Lista de listas de itens recomendados (top-K)
            targets: Lista de itens verdadeiros (próximo item)
            
        Returns:
            Dict com todas as métricas para todos K values
        """
        results = {}
        
        for k in self.k_values:
            results[f'HitRate@{k}'] = self.hit_rate(predictions, targets, k)
            results[f'MRR@{k}'] = self.mrr(predictions, targets, k)
            results[f'Coverage@{k}'] = self.coverage(predictions, k)
            results[f'Popularity@{k}'] = self.popularity_bias(predictions, k)
        
        return results
    
    def evaluate_rest_of_session(self, predictions: List[List[int]], targets: List[List[int]]) -> Dict[str, float]:
        """
        Avalia cenário de Rest of Session
        
        Args:
            predictions: Lista de listas de itens recomendados
            targets: Lista de listas de itens relevantes (resto da sessão)
            
        Returns:
            Dict com todas as métricas para todos K values
        """
        results = {}
        
        for k in self.k_values:
            results[f'Precision@{k}'] = self.precision_at_k(predictions, targets, k)
            results[f'Recall@{k}'] = self.recall_at_k(predictions, targets, k)
            results[f'NDCG@{k}'] = self.ndcg_at_k(predictions, targets, k)
            results[f'MAP@{k}'] = self.map_at_k(predictions, targets, k)
        
        return results

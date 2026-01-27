import numpy as np
import torch
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType


class VSKNNRecommender(SequentialRecommender):
    """
    V-SKNN (Vector Multiplication Session-based KNN)
    
    Jannach, D., & Ludewig, M. (2017). 
    When recurrent neural networks meet the neighborhood for session-based recommendation.
    
    Parameters:
    - k: Number of nearest neighbors (default: 500)
    - sample_size: Sample recent sessions for efficiency (default: 5000, 0 = use all)
    - similarity: Similarity function ('cosine', 'jaccard')
    - weighting: Weighting scheme for current session ('same', 'div', 'linear', 'log', 'quadratic')
    - weighting_score: Weighting for neighbor scores ('same', 'div', 'linear', 'log', 'quadratic')
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(VSKNNRecommender, self).__init__(config, dataset)
        
        self.k = config['k'] if 'k' in config.final_config_dict else 500
        self.sample_size = config['sample_size'] if 'sample_size' in config.final_config_dict else 5000
        self.similarity = config['similarity'] if 'similarity' in config.final_config_dict else 'cosine'
        self.weighting = config['weighting'] if 'weighting' in config.final_config_dict else 'div'
        self.weighting_score = config['weighting_score'] if 'weighting_score' in config.final_config_dict else 'div'
        self.normalize = config['normalize'] if 'normalize' in config.final_config_dict else True
        
        self.n_items = dataset.num(self.ITEM_ID)
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))
        
        # Training data storage
        self.session_item_map = {}  # session_id -> list of item_ids
        self.item_session_map = {}  # item_id -> set of session_ids
        self.session_time = {}      # session_id -> timestamp
        
    def calculate_loss(self, interaction):
        """Build session-item mappings during training"""
        session_ids = interaction[self.USER_ID].cpu().numpy()
        item_seqs = interaction[self.ITEM_SEQ].cpu().numpy()
        item_seq_lens = interaction[self.ITEM_SEQ_LEN].cpu().numpy()
        
        for sess_id, seq, seq_len in zip(session_ids, item_seqs, item_seq_lens):
            if seq_len > 0:
                items = seq[:seq_len].tolist()
                self.session_item_map[int(sess_id)] = items
                
                for item_id in items:
                    if item_id not in self.item_session_map:
                        self.item_session_map[item_id] = set()
                    self.item_session_map[item_id].add(int(sess_id))
        
        return torch.abs(self.fake_loss).sum()

    def full_sort_predict(self, interaction):
        """Generate predictions for all items"""
        batch_size = interaction[self.ITEM_SEQ].size(0)
        scores = torch.zeros(batch_size, self.n_items, device=self.device)
        
        item_seqs = interaction[self.ITEM_SEQ].cpu().numpy()
        item_seq_lens = interaction[self.ITEM_SEQ_LEN].cpu().numpy()
        
        for idx, (seq, seq_len) in enumerate(zip(item_seqs, item_seq_lens)):
            if seq_len > 0:
                current_items = seq[:seq_len].tolist()
                item_scores = self._predict_for_session(current_items)
                scores[idx] = torch.tensor(item_scores, device=self.device)
        
        return scores
    
    def _predict_for_session(self, current_items):
        """KNN-based prediction for a single session"""
        # Find neighbor sessions
        neighbors = self._find_neighbors(current_items)
        
        if len(neighbors) == 0:
            return np.zeros(self.n_items)
        
        # Score items from neighbors
        scores = self._score_items(neighbors, current_items)
        
        if self.normalize and scores.max() > 0:
            scores = scores / scores.max()
        
        return scores
    
    def _find_neighbors(self, current_items):
        """Find k most similar sessions"""
        # Get candidate sessions (all sessions containing at least one current item)
        candidates = set()
        for item in current_items:
            if item in self.item_session_map:
                candidates.update(self.item_session_map[item])
        
        if len(candidates) == 0:
            return []
        
        # Sample if needed
        if self.sample_size > 0 and len(candidates) > self.sample_size:
            candidates = set(np.random.choice(list(candidates), self.sample_size, replace=False))
        
        # Calculate similarities
        similarities = []
        for sess_id in candidates:
            sim = self._calculate_similarity(current_items, self.session_item_map[sess_id])
            if sim > 0:
                similarities.append((sess_id, sim))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:self.k]
    
    def _calculate_similarity(self, items1, items2):
        """Calculate session similarity"""
        set1 = set(items1)
        set2 = set(items2)
        
        if self.similarity == 'jaccard':
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0
        
        elif self.similarity == 'cosine':
            # Vector-based cosine similarity
            intersection = set1 & set2
            if len(intersection) == 0:
                return 0
            
            # Apply positional weighting to current session
            weights1 = self._get_position_weights(len(items1))
            weights2 = np.ones(len(items2))
            
            # Build weighted vectors
            vec1 = np.zeros(self.n_items)
            vec2 = np.zeros(self.n_items)
            
            for pos, item in enumerate(items1):
                vec1[item] = weights1[pos]
            
            for pos, item in enumerate(items2):
                vec2[item] = weights2[pos]
            
            # Cosine similarity
            dot = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            return dot / (norm1 * norm2) if (norm1 * norm2) > 0 else 0
        
        return 0
    
    def _score_items(self, neighbors, current_items):
        """Score all items based on neighbor sessions"""
        scores = np.zeros(self.n_items)
        
        # Position weights for current session
        pos_weights = self._get_position_weights(len(current_items))
        
        for sess_id, similarity in neighbors:
            neighbor_items = self.session_item_map[sess_id]
            
            # Find which items in neighbor were also in current session
            for n_idx, n_item in enumerate(neighbor_items):
                # Calculate decay based on position in neighbor session
                decay = self._get_score_weight(n_idx, len(neighbor_items))
                
                # Find if this item was in current session
                match_weight = 1.0
                if n_item in current_items:
                    # Weight by position in current session
                    c_pos = len(current_items) - 1 - current_items[::-1].index(n_item)
                    match_weight = pos_weights[c_pos]
                
                scores[n_item] += similarity * decay * match_weight
        
        return scores
    
    def _get_position_weights(self, length):
        """Get positional weights for session items (recent items have higher weight)"""
        if self.weighting == 'same':
            return np.ones(length)
        elif self.weighting == 'div':
            return 1 / np.arange(length, 0, -1)
        elif self.weighting == 'linear':
            return np.arange(1, length + 1) / length
        elif self.weighting == 'log':
            return 1 / np.log2(np.arange(length, 0, -1) + 1)
        elif self.weighting == 'quadratic':
            return ((np.arange(1, length + 1) / length) ** 2)
        else:
            return np.ones(length)
    
    def _get_score_weight(self, position, length):
        """Get decay weight for scoring (used for neighbor items)"""
        if self.weighting_score == 'same':
            return 1.0
        elif self.weighting_score == 'div':
            return 1 / (length - position)
        elif self.weighting_score == 'linear':
            return (length - position) / length
        elif self.weighting_score == 'log':
            return 1 / np.log2(length - position + 1)
        elif self.weighting_score == 'quadratic':
            return ((length - position) / length) ** 2
        else:
            return 1.0
    
    def predict(self, interaction):
        """Predict for evaluation"""
        return self.full_sort_predict(interaction)
    
    def forward(self, item_seq, item_seq_len):
        """Forward pass (not used in inference, only for compatibility)"""
        batch_size = item_seq.size(0)
        return torch.zeros(batch_size, self.n_items, device=self.device)

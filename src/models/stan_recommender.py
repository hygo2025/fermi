import numpy as np
import torch
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType


class STANRecommender(SequentialRecommender):
    """
    STAN (Session-based Temporal Attention Network - KNN variant)
    
    Garg, D., et al. (2019). 
    Sequence and time aware neighborhood for session-based recommendations.
    
    Extends V-SKNN with temporal decay functions:
    - lambda_spw: Sequential position weighting (recent items more important)
    - lambda_snh: Session neighborhood decay (older sessions less important)
    - lambda_inh: Item neighborhood decay (items from older positions less important)
    
    Parameters:
    - k: Number of nearest neighbors (default: 500)
    - sample_size: Sample recent sessions (default: 5000)
    - similarity: Similarity function (default: 'cosine')
    - lambda_spw: Position weight decay (default: 1.02)
    - lambda_snh: Session time decay in days (default: 5)
    - lambda_inh: Item position decay (default: 2.05)
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(STANRecommender, self).__init__(config, dataset)
        
        self.k = config['k'] if 'k' in config.final_config_dict else 500
        self.sample_size = config['sample_size'] if 'sample_size' in config.final_config_dict else 5000
        self.similarity = config['similarity'] if 'similarity' in config.final_config_dict else 'cosine'
        
        # STAN temporal parameters
        self.lambda_spw = config['lambda_spw'] if 'lambda_spw' in config.final_config_dict else 1.02
        self.lambda_snh = (config['lambda_snh'] if 'lambda_snh' in config.final_config_dict else 5) * 24 * 3600  # days to seconds
        self.lambda_inh = config['lambda_inh'] if 'lambda_inh' in config.final_config_dict else 2.05
        
        self.n_items = dataset.num(self.ITEM_ID)
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))
        
        # Training data storage
        self.session_item_map = {}  # session_id -> list of item_ids
        self.item_session_map = {}  # item_id -> set of session_ids
        self.session_time = {}      # session_id -> timestamp
        self.min_time = float('inf')
        
    def calculate_loss(self, interaction):
        """Build session-item mappings with timestamps during training"""
        session_ids = interaction[self.USER_ID].cpu().numpy()
        item_seqs = interaction[self.ITEM_SEQ].cpu().numpy()
        item_seq_lens = interaction[self.ITEM_SEQ_LEN].cpu().numpy()
        
        # Get timestamps (use last timestamp of sequence as session time)
        if 'timestamp' in interaction.interaction:
            timestamps = interaction['timestamp'].cpu().numpy()
        else:
            # Fallback: use dummy timestamps
            timestamps = np.arange(len(session_ids))
        
        for sess_id, seq, seq_len, ts in zip(session_ids, item_seqs, item_seq_lens, timestamps):
            if seq_len > 0:
                items = seq[:seq_len].tolist()
                sess_id_int = int(sess_id)
                
                self.session_item_map[sess_id_int] = items
                self.session_time[sess_id_int] = float(ts)
                self.min_time = min(self.min_time, float(ts))
                
                for item_id in items:
                    if item_id not in self.item_session_map:
                        self.item_session_map[item_id] = set()
                    self.item_session_map[item_id].add(sess_id_int)
        
        return torch.abs(self.fake_loss).sum()

    def full_sort_predict(self, interaction):
        """Generate predictions for all items"""
        batch_size = interaction[self.ITEM_SEQ].size(0)
        scores = torch.zeros(batch_size, self.n_items, device=self.device)
        
        item_seqs = interaction[self.ITEM_SEQ].cpu().numpy()
        item_seq_lens = interaction[self.ITEM_SEQ_LEN].cpu().numpy()
        
        # Get current timestamps
        if 'timestamp' in interaction.interaction:
            timestamps = interaction['timestamp'].cpu().numpy()
        else:
            timestamps = np.full(batch_size, self.session_time.get(list(self.session_time.keys())[-1], 0) if self.session_time else 0)
        
        for idx, (seq, seq_len, ts) in enumerate(zip(item_seqs, item_seq_lens, timestamps)):
            if seq_len > 0:
                current_items = seq[:seq_len].tolist()
                item_scores = self._predict_for_session(current_items, float(ts))
                scores[idx] = torch.tensor(item_scores, device=self.device)
        
        return scores
    
    def _predict_for_session(self, current_items, current_time):
        """STAN-based prediction for a single session"""
        # Find neighbor sessions
        neighbors = self._find_neighbors(current_items, current_time)
        
        if len(neighbors) == 0:
            return np.zeros(self.n_items)
        
        # Score items from neighbors with temporal weighting
        scores = self._score_items(neighbors, current_items, current_time)
        
        # Normalize
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores
    
    def _find_neighbors(self, current_items, current_time):
        """Find k most similar sessions with temporal decay"""
        # Get candidate sessions
        candidates = set()
        for item in current_items:
            if item in self.item_session_map:
                candidates.update(self.item_session_map[item])
        
        if len(candidates) == 0:
            return []
        
        # Sample if needed (prefer recent sessions)
        if self.sample_size > 0 and len(candidates) > self.sample_size:
            # Sort by recency and sample
            candidate_list = sorted(
                candidates, 
                key=lambda x: self.session_time.get(x, 0), 
                reverse=True
            )[:self.sample_size]
            candidates = set(candidate_list)
        
        # Calculate similarities with temporal decay
        similarities = []
        for sess_id in candidates:
            # Base similarity
            sim = self._calculate_similarity(current_items, self.session_item_map[sess_id])
            
            if sim > 0:
                # Apply session temporal decay (lambda_snh)
                time_diff = abs(current_time - self.session_time.get(sess_id, current_time))
                temporal_decay = np.exp(-time_diff / self.lambda_snh) if self.lambda_snh > 0 else 1.0
                
                similarities.append((sess_id, sim * temporal_decay))
        
        # Sort by weighted similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:self.k]
    
    def _calculate_similarity(self, items1, items2):
        """Calculate session similarity (cosine by default)"""
        set1 = set(items1)
        set2 = set(items2)
        
        intersection = set1 & set2
        if len(intersection) == 0:
            return 0
        
        if self.similarity == 'jaccard':
            union = len(set1 | set2)
            return len(intersection) / union if union > 0 else 0
        
        elif self.similarity == 'cosine':
            # Weighted cosine with sequential position weighting (lambda_spw)
            vec1 = np.zeros(self.n_items)
            vec2 = np.zeros(self.n_items)
            
            for pos, item in enumerate(items1):
                # Recent items have exponentially higher weight
                weight = self.lambda_spw ** pos
                vec1[item] = weight
            
            for pos, item in enumerate(items2):
                weight = self.lambda_spw ** pos
                vec2[item] = weight
            
            dot = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            return dot / (norm1 * norm2) if (norm1 * norm2) > 0 else 0
        
        return 0
    
    def _score_items(self, neighbors, current_items, current_time):
        """Score items with STAN temporal weighting"""
        scores = np.zeros(self.n_items)
        
        for sess_id, similarity in neighbors:
            neighbor_items = self.session_item_map[sess_id]
            
            for n_idx, n_item in enumerate(neighbor_items):
                # Item position decay (lambda_inh) - recent positions more important
                position_weight = self.lambda_inh ** (len(neighbor_items) - n_idx - 1)
                
                # Check if item appears in current session
                match_weight = 1.0
                if n_item in current_items:
                    # Weight by position in current session (lambda_spw)
                    c_pos = len(current_items) - 1 - current_items[::-1].index(n_item)
                    match_weight = self.lambda_spw ** c_pos
                
                scores[n_item] += similarity * position_weight * match_weight
        
        return scores
    
    def predict(self, interaction):
        """Predict for evaluation"""
        return self.full_sort_predict(interaction)
    
    def forward(self, item_seq, item_seq_len):
        """Forward pass (for compatibility)"""
        batch_size = item_seq.size(0)
        return torch.zeros(batch_size, self.n_items, device=self.device)

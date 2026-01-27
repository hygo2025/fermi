import numpy as np
import torch
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType


class VSTANRecommender(SequentialRecommender):
    """
    V-STAN (Vector Session-based Temporal Attention Network)
    
    Combines V-SKNN's vector-based similarity with STAN's temporal attention mechanisms.
    Adds IDF (Inverse Document Frequency) weighting for items.
    
    Ludewig, M., et al. (2018).
    Evaluation of session-based recommendation algorithms.
    
    Parameters:
    - k: Number of nearest neighbors (default: 500)
    - sample_size: Sample recent sessions (default: 5000)
    - similarity: Similarity function ('cosine', 'jaccard') - default: 'cosine'
    - lambda_spw: Sequential position weight (default: 1.02)
    - lambda_snh: Session temporal decay in days (default: 5)
    - lambda_inh: Item position decay (default: 2.05)
    - lambda_ipw: Item position weight in current session (default: 1.02)
    - lambda_idf: IDF weighting strength (default: 5.0, None to disable)
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(VSTANRecommender, self).__init__(config, dataset)
        
        self.k = config['k'] if 'k' in config.final_config_dict else 500
        self.sample_size = config['sample_size'] if 'sample_size' in config.final_config_dict else 5000
        self.similarity = config['similarity'] if 'similarity' in config.final_config_dict else 'cosine'
        
        # STAN temporal parameters
        self.lambda_spw = config['lambda_spw'] if 'lambda_spw' in config.final_config_dict else 1.02
        self.lambda_snh = (config['lambda_snh'] if 'lambda_snh' in config.final_config_dict else 5) * 24 * 3600  # days to seconds
        self.lambda_inh = config['lambda_inh'] if 'lambda_inh' in config.final_config_dict else 2.05
        
        # V-STAN specific
        self.lambda_ipw = config['lambda_ipw'] if 'lambda_ipw' in config.final_config_dict else 1.02
        self.lambda_idf = config['lambda_idf'] if 'lambda_idf' in config.final_config_dict else 5.0
        
        self.n_items = dataset.num(self.ITEM_ID)
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))
        
        # Training data storage
        self.session_item_map = {}
        self.item_session_map = {}
        self.session_time = {}
        self.min_time = float('inf')
        
        # IDF weights
        self.idf = None
        self.item_freq = np.zeros(self.n_items)
        
    def calculate_loss(self, interaction):
        """Build session-item mappings and compute IDF weights"""
        session_ids = interaction[self.USER_ID].cpu().numpy()
        item_seqs = interaction[self.ITEM_SEQ].cpu().numpy()
        item_seq_lens = interaction[self.ITEM_SEQ_LEN].cpu().numpy()
        
        # Get timestamps
        if 'timestamp' in interaction.interaction:
            timestamps = interaction['timestamp'].cpu().numpy()
        else:
            timestamps = np.arange(len(session_ids))
        
        for sess_id, seq, seq_len, ts in zip(session_ids, item_seqs, item_seq_lens, timestamps):
            if seq_len > 0:
                items = seq[:seq_len].tolist()
                sess_id_int = int(sess_id)
                
                # Store as list (preserving order for V-STAN)
                self.session_item_map[sess_id_int] = items
                self.session_time[sess_id_int] = float(ts)
                self.min_time = min(self.min_time, float(ts))
                
                # Track item frequencies for IDF
                for item_id in items:
                    self.item_freq[item_id] += 1
                    
                    if item_id not in self.item_session_map:
                        self.item_session_map[item_id] = set()
                    self.item_session_map[item_id].add(sess_id_int)
        
        # Compute IDF weights
        if self.lambda_idf is not None:
            n_sessions = len(self.session_item_map)
            self.idf = np.log(n_sessions / (self.item_freq + 1))
        
        return torch.abs(self.fake_loss).sum()

    def full_sort_predict(self, interaction):
        """Generate predictions for all items"""
        batch_size = interaction[self.ITEM_SEQ].size(0)
        scores = torch.zeros(batch_size, self.n_items, device=self.device)
        
        item_seqs = interaction[self.ITEM_SEQ].cpu().numpy()
        item_seq_lens = interaction[self.ITEM_SEQ_LEN].cpu().numpy()
        
        # Get timestamps
        if 'timestamp' in interaction.interaction:
            timestamps = interaction['timestamp'].cpu().numpy()
        else:
            timestamps = np.full(batch_size, max(self.session_time.values()) if self.session_time else 0)
        
        for idx, (seq, seq_len, ts) in enumerate(zip(item_seqs, item_seq_lens, timestamps)):
            if seq_len > 0:
                current_items = seq[:seq_len].tolist()
                item_scores = self._predict_for_session(current_items, float(ts))
                scores[idx] = torch.tensor(item_scores, device=self.device)
        
        return scores
    
    def _predict_for_session(self, current_items, current_time):
        """V-STAN prediction for a single session"""
        neighbors = self._find_neighbors(current_items, current_time)
        
        if len(neighbors) == 0:
            return np.zeros(self.n_items)
        
        scores = self._score_items(neighbors, current_items, current_time)
        
        # Apply IDF weighting
        if self.idf is not None:
            scores = scores * (self.idf ** self.lambda_idf)
        
        # Normalize
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores
    
    def _find_neighbors(self, current_items, current_time):
        """Find k most similar sessions with vector-based similarity"""
        # Get candidates
        candidates = set()
        for item in current_items:
            if item in self.item_session_map:
                candidates.update(self.item_session_map[item])
        
        if len(candidates) == 0:
            return []
        
        # Sample recent sessions
        if self.sample_size > 0 and len(candidates) > self.sample_size:
            candidate_list = sorted(
                candidates, 
                key=lambda x: self.session_time.get(x, 0), 
                reverse=True
            )[:self.sample_size]
            candidates = set(candidate_list)
        
        # Calculate vector-based similarities with temporal decay
        similarities = []
        for sess_id in candidates:
            neighbor_items = self.session_item_map[sess_id]
            
            # Vector similarity with positional weighting
            sim = self._vector_similarity(current_items, neighbor_items)
            
            if sim > 0:
                # Session temporal decay
                time_diff = abs(current_time - self.session_time.get(sess_id, current_time))
                temporal_decay = np.exp(-time_diff / self.lambda_snh) if self.lambda_snh > 0 else 1.0
                
                similarities.append((sess_id, sim * temporal_decay))
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:self.k]
    
    def _vector_similarity(self, items1, items2):
        """Vector-based cosine similarity with positional decay"""
        if self.similarity == 'jaccard':
            # Simple Jaccard
            set1 = set(items1)
            set2 = set(items2)
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0
        
        # Cosine with position weighting
        vec1 = np.zeros(self.n_items)
        vec2 = np.zeros(self.n_items)
        
        # Build weighted vectors
        for pos, item in enumerate(items1):
            # Item position weight (lambda_ipw) - more recent = higher weight
            weight = self.lambda_ipw ** pos
            vec1[item] += weight
        
        for pos, item in enumerate(items2):
            weight = self.lambda_ipw ** pos
            vec2[item] += weight
        
        # Cosine
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        return dot / (norm1 * norm2) if (norm1 * norm2) > 0 else 0
    
    def _score_items(self, neighbors, current_items, current_time):
        """Score items with V-STAN weighting scheme"""
        scores = np.zeros(self.n_items)
        
        # Position weights for current session
        current_pos_map = {}
        for pos, item in enumerate(current_items):
            if item not in current_pos_map:
                current_pos_map[item] = []
            current_pos_map[item].append(pos)
        
        for sess_id, similarity in neighbors:
            neighbor_items = self.session_item_map[sess_id]
            
            for n_idx, n_item in enumerate(neighbor_items):
                # Base score from similarity
                score = similarity
                
                # Item position decay in neighbor (lambda_inh)
                # More recent positions in neighbor have higher weight
                position_decay = self.lambda_inh ** (len(neighbor_items) - n_idx - 1)
                score *= position_decay
                
                # If item appears in current session, boost based on position
                if n_item in current_pos_map:
                    # Use most recent position
                    c_pos = max(current_pos_map[n_item])
                    # Sequential position weight (lambda_spw)
                    spw = self.lambda_spw ** c_pos
                    score *= spw
                
                scores[n_item] += score
        
        return scores
    
    def predict(self, interaction):
        """Predict for evaluation"""
        return self.full_sort_predict(interaction)
    
    def forward(self, item_seq, item_seq_len):
        """Forward pass (for compatibility)"""
        batch_size = item_seq.size(0)
        return torch.zeros(batch_size, self.n_items, device=self.device)

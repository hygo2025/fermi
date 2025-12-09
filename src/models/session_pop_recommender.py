import torch
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType


class SPOPRecommender(SequentialRecommender):
    """
    SPOP (Session Popularity) baseline
    Recommends most popular items within current session
    Falls back to global popularity for ties
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(SPOPRecommender, self).__init__(config, dataset)
        self.n_items = dataset.num(self.ITEM_ID)
        self.global_item_cnt = torch.zeros(self.n_items, dtype=torch.long)
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))

    def forward(self, item_seq, item_seq_len):
        batch_size = item_seq.size(0)
        scores = torch.zeros(batch_size, self.n_items)

        # For each session, count item frequencies
        for i, seq in enumerate(item_seq):
            session_cnt = torch.zeros(self.n_items, dtype=torch.long)
            for item_id in seq[:item_seq_len[i]]:
                if item_id != 0:
                    session_cnt[item_id] += 1

            # Combine session popularity with global as tiebreaker
            # Normalize global counts to small values
            global_normalized = self.global_item_cnt.float() / (self.global_item_cnt.max() + 1e-8) * 0.01
            scores[i] = session_cnt.float() + global_normalized

        return scores

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]

        # Build global popularity (vectorized)
        items_flat = item_seq.flatten()
        items_flat = items_flat[items_flat != 0]

        if len(items_flat) > 0:
            counts = torch.bincount(items_flat, minlength=self.n_items)
            self.global_item_cnt += counts.cpu()

        return torch.abs(self.fake_loss).sum()

    def predict(self, interaction):
        batch_size = len(interaction)
        item_seq = interaction.get(self.ITEM_SEQ)

        if item_seq is None:
            return self.global_item_cnt.float().unsqueeze(0).repeat(batch_size, 1)

        scores = torch.zeros(batch_size, self.n_items)
        for i, seq in enumerate(item_seq):
            session_cnt = torch.zeros(self.n_items, dtype=torch.long)
            for item_id in seq:
                if item_id != 0:
                    session_cnt[item_id] += 1
            global_normalized = self.global_item_cnt.float() / (self.global_item_cnt.max() + 1e-8) * 0.01
            scores[i] = session_cnt.float() + global_normalized

        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        batch_size = item_seq.size(0)
        scores = torch.zeros(batch_size, self.n_items).to(self.device)

        # For each session in batch
        for i in range(batch_size):
            session_cnt = torch.zeros(self.n_items, dtype=torch.long, device=self.device)
            seq_len = item_seq_len[i].item()

            for j in range(seq_len):
                item_id = item_seq[i][j].item()
                if item_id != 0:
                    session_cnt[item_id] += 1

            # Combine session + global popularity
            global_normalized = self.global_item_cnt.float().to(self.device) / (
                        self.global_item_cnt.max() + 1e-8) * 0.01
            scores[i] = session_cnt.float() + global_normalized

        return scores
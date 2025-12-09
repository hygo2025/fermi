import torch
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType


class RPOPRecommender(SequentialRecommender):
    """
    RPOP (Recent Popularity) baseline
    Recommends items popular in recent time window
    Uses session-level recency (current batch as proxy for recent items)
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(RPOPRecommender, self).__init__(config, dataset)
        self.n_items = dataset.num(self.ITEM_ID)
        self.recent_item_cnt = torch.zeros(self.n_items, dtype=torch.long)
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))

    def forward(self, item_seq, item_seq_len):
        batch_size = item_seq.size(0)
        scores = self.recent_item_cnt.float().unsqueeze(0).repeat(batch_size, 1)
        return scores

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]

        # Reset recent counts (only current batch matters)
        self.recent_item_cnt.zero_()

        # Count recent item frequencies (vectorized)
        items_flat = item_seq.flatten()
        items_flat = items_flat[items_flat != 0]

        if len(items_flat) > 0:
            counts = torch.bincount(items_flat, minlength=self.n_items)
            self.recent_item_cnt += counts.cpu()

        return torch.abs(self.fake_loss).sum()

    def predict(self, interaction):
        batch_size = len(interaction)
        return self.recent_item_cnt.float().unsqueeze(0).repeat(batch_size, 1)

    def full_sort_predict(self, interaction):
        batch_size = interaction[self.ITEM_SEQ].size(0)

        # Update recent counts with current batch
        item_seq = interaction[self.ITEM_SEQ]
        self.recent_item_cnt.zero_()
        for seq in item_seq:
            for item_id in seq:
                if item_id != 0:
                    self.recent_item_cnt[item_id] += 1

        scores = self.recent_item_cnt.float().unsqueeze(0).repeat(batch_size, 1)
        return scores.to(self.device)
import torch
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType


class POPRecommender(SequentialRecommender):
    """
    POP (Popularity) baseline
    Recommends items with highest global frequency in training set
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(POPRecommender, self).__init__(config, dataset)
        self.n_items = dataset.num(self.ITEM_ID)
        self.item_cnt = torch.zeros(self.n_items, dtype=torch.long)
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))

    def forward(self, item_seq, item_seq_len):
        batch_size = item_seq.size(0)
        scores = self.item_cnt.float().unsqueeze(0).repeat(batch_size, 1)
        return scores

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]

        # Count item frequencies (vectorized)
        items_flat = item_seq.flatten()
        items_flat = items_flat[items_flat != 0]  # Remove padding

        # Use bincount for fast counting
        if len(items_flat) > 0:
            counts = torch.bincount(items_flat, minlength=self.n_items)
            self.item_cnt += counts.cpu()

        return torch.abs(self.fake_loss).sum()

    def predict(self, interaction):
        batch_size = len(interaction)
        return self.item_cnt.float().unsqueeze(0).repeat(batch_size, 1)

    def full_sort_predict(self, interaction):
        batch_size = interaction[self.ITEM_SEQ].size(0)
        scores = self.item_cnt.float().unsqueeze(0).repeat(batch_size, 1)
        return scores.to(self.device)
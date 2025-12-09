import torch
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType


class RandomRecommender(SequentialRecommender):
    """Random baseline - recommends items uniformly at random"""

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(RandomRecommender, self).__init__(config, dataset)
        self.n_items = dataset.num(self.ITEM_ID)
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))

    def forward(self, item_seq, item_seq_len):
        return torch.randn(item_seq.size(0), self.n_items)

    def calculate_loss(self, interaction):
        return torch.abs(self.fake_loss).sum()

    def predict(self, interaction):
        return torch.randn(interaction.shape[0], self.n_items)

    def full_sort_predict(self, interaction):
        batch_size = interaction[self.ITEM_SEQ].size(0)
        return torch.randn(batch_size, self.n_items).to(self.device)
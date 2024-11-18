import torch
from torch.utils.data import Dataset

class NormalDataset(Dataset):
    def __init__(self, number_of_nodes, number_of_samples):
        super().__init__()
        self.distribution = torch.distributions.Normal(
            loc=torch.zeros(number_of_nodes), 
            scale=torch.ones(number_of_nodes))
        self.number_of_nodes = number_of_nodes
        self.n_sample = number_of_samples
        self.sample()

    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        return self.features[index]
    
    def sample(self):
        self.features = self.distribution.sample((self.n_sample,))
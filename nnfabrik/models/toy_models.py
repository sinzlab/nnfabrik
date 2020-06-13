import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim=5):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, out_dim)
        self.nl = nn.ReLU()

    def forward(self, x):
        out = self.nl(self.fc1(x))
        return self.nl(self.fc2(out))


def toy_model_fn(dataloaders, seed, h_dim=5):
    
    # get the input and output dimension for the model
    in_dim = dataloaders.dataset.tensors[0].shape[1]
    out_dim = dataloaders.dataset.tensors[1].shape[1]
    
    torch.manual_seed(seed) # for reproducibility (almost)
    model = ToyModel(in_dim, out_dim, h_dim=h_dim)
    
    return model


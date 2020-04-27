import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, kern=16, kern2=32):
        super().__init__()
        self.kern = kern
        self.lin1 = nn.Linear(128, kern)
        self.lin2 = nn.Linear(kern, 16)

    def forward(self, x):
        return self.lin2(self.lin1(x))
    
def toy_model(dataloaders, seed, kern=16, kern2=32):
    return ToyModel(kern=kern, kern2=kern2)


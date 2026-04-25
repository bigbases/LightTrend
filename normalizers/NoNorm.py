import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        
    def normalize(self, batch_x):
        return batch_x, None
    
    def de_normalize(self, batch_y, statistics):
        return batch_y


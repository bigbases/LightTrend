import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        if configs.affine: # configs.affine: use affine layers or not
            self.gamma = nn.Parameter(torch.ones(configs.enc_in))
            self.beta = nn.Parameter(torch.zeros(configs.enc_in))
        else:
            self.gamma, self.beta = 1, 0
        
    def normalize(self, batch_x):
        # batch_x: B*L*D || dec_inp: B*?*D (for xxformers)
        avg = torch.mean(batch_x, axis=1, keepdim=True).detach() # b*1*d
        var = torch.var(batch_x, axis=1, keepdim=True).detach()  # b*1*d
        temp = (batch_x - avg)/torch.sqrt(var + 1e-8)
        batch_x = temp.mul(self.gamma) + self.beta
        return batch_x, (avg, var)
    
    def de_normalize(self, batch_x, statistics):
        # batch_x: B*H*D (forecasts)
        avg, var = statistics
        batch_y = ((batch_x - self.beta) / self.gamma) * torch.sqrt(var + 1e-8) + avg
        return batch_y


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Quantize(nn.Module):
    def __init__(
        self,
        hidden_dim,
        entryV,
        num_codebook,
        tau_min,
    ):
        super().__init__()

        self.linear = nn.Linear( hidden_dim, entryV * num_codebook )
        var_dim = hidden_dim // num_codebook
        self.vars = nn.Parameter(torch.FloatTensor(1, 1 * entryV, var_dim))
        self.entryV = entryV
        self.num_codebook = num_codebook
        self.tau_min = tau_min


    def forward(self, x, tau ):
        
        bsz, tsz, fsz = x.shape
        
        x = self.linear( x )
        x2 = torch.mean( torch.mean( x, dim = 0 ), dim = 0 )

        tau = torch.max( tau, torch.tensor(self.tau_min) )
        y = F.gumbel_softmax( x, tau, hard = True, dim = 2 )
        #y = F.softmax( x, dim = 2 )
        y2 = F.softmax( x2, dim = 0 )

        y = y.view( bsz * tsz, -1 )
        y2 = y2.view( -1 )
        
        vars = self.vars
        vars = vars.repeat(1, self.num_codebook, 1)
        y = y.unsqueeze(-1) * vars
        y = y.view(bsz * tsz, self.num_codebook, self.entryV, -1)
        y = y.sum(-2)
        y = y.view(bsz, tsz, -1)
        y2 = y2.view( self.num_codebook, -1 )

        return y, y2



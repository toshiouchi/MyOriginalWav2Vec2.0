# -*- coding: utf-8 -*-

#
# Transformer エンコーダ部の実装です．
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from attention import ResidualAttentionBlock

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PositionalConvEmbbeding(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding, groups):
        super().__init__()
        self.conv = nn.Conv1d( in_dim, out_dim, kernel_size, stride = 1, padding = padding, groups = groups)

    def forward( self, x ):
        x = self.conv( x.transpose(1,2))
        x = F.gelu(x[:,:,:-1])
        return x.transpose(1,2)

class Encoder(nn.Module):
    def __init__(
        self,
        embed_dim=80,
        PCE_kernel_size=128,
        PCE_groups = 16,
        enc_hidden_dim = 512,
        num_enc_layers = 6,
        enc_num_heads = 4,
        enc_kernel_size = [5,1],
        enc_filter_size = 2048,
        enc_input_maxlen = 3000,
        enc_dropout_rate = 0.1,
    ):
        super(Encoder, self).__init__()
        #Positional convolution embedding
        self.positional_embedding = PositionalConvEmbbeding( \
                 in_dim = enc_hidden_dim, out_dim = enc_hidden_dim, \
                 kernel_size = PCE_kernel_size, padding = PCE_kernel_size // 2 , groups = PCE_groups )
        
        # Attention Block
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(enc_hidden_dim, enc_num_heads, cross_attention = False, kernel_size = enc_kernel_size, filter_size = enc_filter_size ) for _ in range(num_enc_layers)]
        )

        self.ln1 = nn.LayerNorm( enc_hidden_dim )
        
        self.input_maxlen = enc_input_maxlen
        
    def forward(self, x, in_lens ):

        out = x
        
        # positional embbeding
        x = out + self.positional_embedding( out )
        x = self.ln1( x )

        # attention block
        for i, block in enumerate( self.blocks ):
            x = block(x, x, mask = None)
       
        return x  # (batch_size, input_seq_len, d_model)

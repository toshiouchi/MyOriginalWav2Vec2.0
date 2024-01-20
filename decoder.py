# -*- coding: utf-8 -*-

#
# Transformer デコーダ部の実装です．
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch
import torch.nn as nn
from attention import ResidualAttentionBlock

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# プロット用モジュール(matplotlib)をインポート
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Decoder(nn.Module):
    def __init__(
        self,
        dec_num_layers=6,
        dec_input_maxlen=300,
        decoder_hidden_dim=512,
        dec_num_heads = 4,
        dec_kernel_size = [5,1],
        dec_filter_size = 2048,
        dec_dropout_rate = 0.1,
    ):
        super().__init__()
        self.num_heads = dec_num_heads

        #  Attention  Block
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(decoder_hidden_dim, dec_num_heads, cross_attention=True, kernel_size = dec_kernel_size, filter_size = dec_filter_size ) for _ in range(dec_num_layers)]
        )
        
        # position embedding
        self.pos_emb = nn.Embedding(dec_input_maxlen, decoder_hidden_dim)
        
        self.dec_input_maxlen = dec_input_maxlen

    def forward(self, encoder_outs, decoder_targets=None):

        # position embedding
        maxlen = decoder_targets.size()[1]
        positions = torch.arange(start=0, end=self.dec_input_maxlen, step=1, device=torch.device(device)).to(torch.long)
        positions = self.pos_emb(positions)[:maxlen,:]
        x = decoder_targets + positions

        #attention block
        for i, block in enumerate( self.blocks ):
            x = block(x, encoder_outs, mask=None)
        
        return x


# -*- coding: utf-8 -*-

#
# Attention Blockです．

# Pytorchを用いた処理に必要なモジュールをインポート

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

class Linear(nn.Linear):
    def forward(self, x):
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)
        self.layer_norm = nn.LayerNorm(n_state)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        x,
        xa = None,
        mask = None
    ):
        #print( " size of x:{}".format( x.size() ))
        residual = x
        q = self.query(x)

        if xa is None:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            k = self.key( xa )
            v = self.value( xa )

        wv, qk = self.qkv_attention(q, k, v, mask)

        return self.out(wv)

    def qkv_attention(
        self, q, k, v, mask = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        output = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        qk_detach =  qk.detach()
        
        return output, qk_detach

class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
        )
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False, kernel_size = [9,1], filter_size = 2048):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head ) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        self.pos_ffn = PositionwiseFeedForward(n_state, d_hid = filter_size, kernel_size = kernel_size, dropout = 0.1  )

    def forward(
        self,
        x,
        xa,
        mask = None
    ):
        xx = self.attn(self.attn_ln(x), mask=mask)
        x = x + xx
        if self.cross_attn:
            xx = self.cross_attn(self.cross_attn_ln(x), xa, mask = None)
            x = x + xx

        x = self.pos_ffn( x )

        return x


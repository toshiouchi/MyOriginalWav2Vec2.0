# -*- coding: utf-8 -*-

#
# Transformer エンコーダ部の実装です．
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch
import torch.nn as nn
import numpy as np
#from my_dataset import SequenceDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FeatureExtractor(nn.Module):
    def __init__(
        self,
        fe_conv_layer=7,
        fe_conv_channel=[512,512,512,512,512,512,512],
        fe_conv_kernel=[10,3,3,3,3,2,2],
        fe_conv_stride=[5,2,2,2,2,2,2],
        fe_conv_dropout_rate = 0.0,
        fe_out_dim = 768,
    ):
        super(FeatureExtractor, self).__init__()

        # 1 次元畳み込みの重ね合わせ：局所的な時間依存関係のモデル化
        convs = nn.ModuleList()
        
        convs += [
            nn.Conv1d(
                1,
                fe_conv_channel[0],
                fe_conv_kernel[0],
                stride = fe_conv_stride[0],
                bias = False,
            ),
            nn.GroupNorm( 1, fe_conv_channel[0] ),
            nn.GELU()
        ]
        for layer in range( 1, fe_conv_layer - 2 ):
            convs += [
                nn.Conv1d(
                    fe_conv_channel[layer-1],
                    fe_conv_channel[layer],
                    fe_conv_kernel[layer],
                    stride = fe_conv_stride[layer],
                    bias = False,
                ),
                nn.GroupNorm( 1, fe_conv_channel[layer] ),
                nn.GELU(),
            ]
        for layer in range( fe_conv_layer -2, fe_conv_layer):
            convs += [
                nn.Conv1d(
                    fe_conv_channel[layer-1],
                    fe_conv_channel[layer],
                    fe_conv_kernel[layer],
                    stride = fe_conv_stride[layer],
                    bias = False,
                ),
                nn.GroupNorm( 1, fe_conv_channel[layer] ),
                nn.GELU(),
            ]

        self.convs = nn.Sequential(*convs)
        self.ln = nn.LayerNorm( fe_out_dim,eps=1e-5,elementwise_affine=True )
        self.linear = nn.Linear( fe_conv_channel[layer], fe_out_dim, bias = True )
        self.dropout = nn.Dropout(p=fe_conv_dropout_rate)
        
        self.fe_conv_kernel = fe_conv_kernel
        self.fe_conv_stride = fe_conv_stride
        
    def forward(self, x, x_len ):

        # conv 層
        out = self.convs(x.transpose(1, 2)).transpose(1, 2)
        out_len = x_len
        for kernel, stride in zip( self.fe_conv_kernel, self.fe_conv_stride ):
            out_len = torch.round( ( out_len  - kernel ) / stride  + 1 ).long()

        y = self.ln( out )
        y = self.linear( y )
        out = self.dropout( y )

        return out, out_len  # (batch_size, input_seq_len, d_model)
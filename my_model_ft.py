# -*- coding: utf-8 -*-

#
# モデル構造を定義します
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch
import torch.nn as nn
import time

# 作成したEncoder, Decoderクラスをインポート
from encoder import Encoder
from decoder import Decoder

# 作成した初期化関数をインポート
from initialize import lecun_initialization
from feature_extractor import FeatureExtractor
from quantize import Quantize
import gc

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#def dump_garbage():
#    """ どんなゴミがあるか見せる """
#    # 強制収集
#    print('GARBAGE:')
#    gc.collect() # 検出した到達不可オブジェクトの数を返します。
#    print2('GARBAGE OBJECTS:')
#    # 到達不能であることが検出されたが、解放する事ができないオブジェクトのリスト
#    # （回収不能オブジェクト）
#    for x in gc.garbage:
#        s = str(x)
#        if len(s) > 80: s = s[:77]+'...'
#        print2(type(x),'\n ',s)
#    print2('END GARBAGE OBJECTS:')

class MyE2EModel(nn.Module):
    ''' 変数の意味
    dim_in:             ダミー変数
    dim_out:            ダミー変数
    fe_conv_layer       Feature Extractor のレイヤー数
    fe_conv_channel     Feature Extractor の Convolution channel 数（出力）
    fe_conv_kernel      Feature Extractor の Convolution kenrnel size
    fe_conv_stride      Feature Extractor の Convolution stride
    fe_conf_drop_out_rate Feature Extractor の Dropout Rate
    fe_out_dim          Feature Extractor の最後にある線形層の出力次元
    PCE_kernel_size:    Positional Convolution Embedding の kernel_size
    PCE_groups:         Positional Convolution Embedding の groups 数
    enc_num_layers:     エンコーダー層数
    enc_att_hidden_dim: エンコーダーのアテンションの隠れ層数
    enc_num_heads:      エンコーダーのhead数
    enc_input_maxlen:   エンコーダーの入力の時間数の最大フレーム値 3000
    enc_att_kernel_size:エンコーダートランスフォーマーのカーネルサイズ
    enc_att_filter_size:エンコーダートランスフォーマーのフィルター数
    enc_dropout_rate:   エンコーダーのドロップアウト
    dec_num_layers:     デコーダー層数
    ds_rate             ダウンサンプリングの割合
    n_mask              マスクの先頭の割合
    n_consec            マスクの連続する数
    entryV              コードブックのエントリー数
    num_codebook        コードブックの数
    tau                 Gumbel Softmax の温度の初期値
    temprature_multi    Gumbel Softmax の温度の1エポック経過したあとの倍数
    tau_min             Gumbel Softmax の温度の最小値
    dec_att_hidden_dim: デコーダーのアテンションの隠れ層数
    dec_num_heads:      デコーダーのhead数
    dec_input_maxlen:   デコーダーの入力の時間数の最大フレーム値 300
    dec_att_kernel_size:デコーダートランスフォーマーのカーネルサイズ
    dec_att_filter_size:デコーダートランスフォーマーのフィルター数
    dec_dropout_rage   :デコーダーのドロップアウト
    sos_id:             ダミー変数
    '''


    def __init__(self, dim_in, dim_out,
                 fe_conv_layer, fe_conv_channel, fe_conv_kernel, fe_conv_stride, fe_conv_dropout_rate, fe_out_dim,
                 PCE_kernel_size, PCE_groups,
                 enc_num_layers, enc_att_hidden_dim, enc_num_heads, enc_input_maxlen,  enc_att_kernel_size, enc_att_filter_size, enc_dropout_rate,
                 ds_rate,n_mask,n_consec, entryV, num_codebook, tau, temprature_multi,tau_min,
                 dec_num_layers, dec_att_hidden_dim, dec_num_heads, dec_target_maxlen, dec_att_kernel_size, dec_att_filter_size, dec_dropout_rate,
                 sos_id,
                 ):
        super(MyE2EModel, self).__init__()

        self.fe = FeatureExtractor(
            fe_conv_layer=fe_conv_layer,
            fe_conv_channel=fe_conv_channel,
            fe_conv_kernel=fe_conv_kernel,
            fe_conv_stride=fe_conv_stride,
            fe_conv_dropout_rate = fe_conv_dropout_rate,
            fe_out_dim=fe_out_dim,
        )
        self.quantize = Quantize(
            hidden_dim = enc_att_hidden_dim,
            entryV = entryV,
            num_codebook = num_codebook,
            tau_min = tau_min,
        )
        
        # エンコーダを作成
        self.encoder = Encoder(
            PCE_kernel_size=PCE_kernel_size,
            PCE_groups=PCE_groups,
            embed_dim = dim_in,
            num_enc_layers = enc_num_layers,
            enc_hidden_dim = enc_att_hidden_dim,
            enc_num_heads = enc_num_heads,
            enc_input_maxlen = enc_input_maxlen,
            enc_kernel_size = enc_att_kernel_size,
            enc_filter_size = enc_att_filter_size,
            enc_dropout_rate = enc_dropout_rate,
        )
        
        # デコーダを作成
        self.decoder = Decoder(
            dec_num_layers = dec_num_layers,
            dec_input_maxlen = dec_target_maxlen,
            decoder_hidden_dim = dec_att_hidden_dim,
            dec_num_heads = dec_num_heads,
            dec_kernel_size = dec_att_kernel_size,
            dec_filter_size = dec_att_filter_size,
            dec_dropout_rate = dec_dropout_rate,
        )


        #　デコーダーのあとに、n * t * hidden を n * t * num_vocab にする線形層。
        self.classifier = nn.Linear( dec_att_hidden_dim, dec_att_hidden_dim )
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm( dec_att_hidden_dim )
        
        self.dec_target_maxlen = dec_target_maxlen
        self.sos_id = sos_id
        self.ds_rate = ds_rate
        self.n_mask = n_mask
        self.n_consec = n_consec

        # LeCunのパラメータ初期化を実行
        #lecun_initialization(self)

    #@profile
    def forward(self,
                input_sequence,
                input_lengths,
                tau):
        ''' ネットワーク計算(forward処理)の関数
        input_sequence: 各発話の入力系列 [B x Tin x D]
        input_lengths:  各発話の系列長(フレーム数) [B]
          []の中はテンソルのサイズ
          B:    ミニバッチ内の発話数(ミニバッチサイズ)
          Tin:  入力テンソルの系列長(ゼロ埋め部分含む)
          D:    入力次元数(dim_in)
          Tout: 正解ラベル系列の系列長(ゼロ埋め部分含む)
        '''
        # Feature Extractor
        input_sequence2, input_lengths2 = self.fe( input_sequence, input_lengths )
        
        # エンコーダに入力する。mask なし入力。
        enc_out = self.encoder(input_sequence2,input_lengths2)
        enc_lengths = input_lengths2
        
        dec_input, outputs_lens = self.downsample( enc_out, input_lengths2 )
        
        # デコーダに入力する
        dec_out = self.decoder(enc_out, dec_input)
        dec_out = self.gelu( dec_out )
        dec_out = self.ln( dec_out )

        # n * T * hidden → n * T * num_vocab ( Fine Tuning の時は、一階層上のソースで修正する）
        outputs = self.classifier( dec_out )

        return outputs, outputs_lens

    def downsample(self, enc_out, input_lengths):
        
        max_label_length = int( round( enc_out.size(1) * self.ds_rate ) )
        
        polated_lengths = torch.round( torch.ones( (enc_out.size(0)), device=torch.device(device) ) * enc_out.size(1) * self.ds_rate ).long()

        outputs_lens = torch.ceil( input_lengths * self.ds_rate ).long()

        x = enc_out
        out_lens = polated_lengths

        for i in range( x.size(0) ):
            x0 = torch.unsqueeze( x[i], dim = 0 )
            x0 = x0.permute( 0,2,1 )
            x_out = torch.nn.functional.interpolate(x0, size = (out_lens[i]), mode='nearest-exact')
            z = torch.zeros( (x_out.size(0), x_out.size(1), max_label_length), device=torch.device(device) )
            if z.size(2) > x_out.size(2):
            	z[:,:,:x_out.size(2)] = x_out[:,:,:]
            else:
                z[:,:,:] = x_out[:,:,:z.size(2)]
            x_out = z.permute( 0, 2, 1 )
            if i == 0:
                y = x_out
            if i > 0:
                y = torch.cat( (y, x_out), dim = 0 )

        return y, outputs_lens

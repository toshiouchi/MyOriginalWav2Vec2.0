# -*- coding: utf-8 -*-

#
# Pytorchで用いるDatasetの定義
#

# PytorchのDatasetモジュールをインポート
import torch
from torch.utils.data import Dataset

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# sysモジュールをインポート
import sys
import gc

import scipy.io.wavfile as sw
import scipy.stats
import numpy as np
#import functools
#print = functools.partial(print, flush=True)

#def dump_garbage():
#    """ どんなゴミがあるか見せる """
#    # 強制収集
#    print('GARBAGE:')
#    gc.collect() # 検出した到達不可オブジェクトの数を返します。
#    print('GARBAGE OBJECTS:')
#    # 到達不能であることが検出されたが、解放する事ができないオブジェクトのリスト
#    # （回収不能オブジェクト）
#    for x in gc.garbage:
#        s = str(x)
#        if len(s) > 80: s = s[:77]+'...'
#        print(type(x),'\n ',s)
#    print('END GARBAGE OBJECTS:')

    

def wavread(wavefile, norm=True):
    """
    read wavfile like matlab audioread() function.
    Parameters
    ----------
    wavefile: str
        wavefile path to read
    norm: bool, default:True
        audio data normalization settings
        call audio_normalize() function
    Returns
    ----------
    y: np.ndarray
        audio data
    fs: int
        sampling rate
    """
    fs, y = sw.read(wavefile)
    if norm:
        y = np.float32( scipy.stats.zscore(y) )

    return (y, fs)

class SequenceDataset(Dataset):
    ''' ミニバッチデータを作成するクラス
        torch.utils.data.Datasetクラスを継承し，
        以下の関数を定義する
        __len__: 総サンプル数を出力する関数
        __getitem__: 1サンプルのデータを出力する関数
    feat_scp:  特徴量リストファイル
    pad_index: バッチ化の際にフレーム数を合わせる
               ためにpaddingする整数値
    splice:    前後(splice)フレームを特徴量を結合する
               splice=1とすると，前後1フレーム分結合
               するので次元数は3倍になる．
               splice=0の場合は何もしない
    '''
    def __init__(self, 
                 feat_scp, 
                 pad_index=0,
                 splice=0):
        # 発話の数
        self.num_utts = 0
        # 各発話のID
        self.id_list = []
        # 各発話の特徴量ファイルへのパスを記したリスト
        self.feat_list = []
        # 各発話の特徴量フレーム数を記したリスト
        self.feat_len_list = []
        # 特徴量の次元数
        self.feat_dim = 1
        # フレーム数の最大値
        self.max_feat_len = 0
        # フレーム埋めに用いる整数値
        self.pad_index = pad_index
        # splice:前後nフレームの特徴量を結合
        self.splice = splice

        # 特徴量リスト，ラベルを1行ずつ
        # 読み込みながら情報を取得する
        with open(feat_scp, mode='r') as file_f:
            for line_feats in file_f:
                # 各行をスペースで区切り，
                # リスト型の変数にする
                #print( "line_feats:",line_feats )
                parts_feats = line_feats.split()


                # 発話IDをリストに追加
                self.id_list.append(parts_feats[0])
                # 特徴量ファイルのパスをリストに追加
                self.feat_list.append(parts_feats[1])
                (audio,sr) = wavread( parts_feats[1] )
                # フレーム数をリストに追加
                feat_len = np.int64(len( audio ))
                self.feat_len_list.append(feat_len)

                # 発話数をカウント
                self.num_utts += 1
                
        # フレーム数の最大値を得る
        self.max_feat_len = \
            np.max(self.feat_len_list)

    def __len__(self):
        ''' 学習データの総サンプル数を返す関数
        本実装では発話単位でバッチを作成するため，
        総サンプル数=発話数である．
        '''
        return self.num_utts

    def __getitem__(self, idx):

        #gc.enable() # 自動ガベージコレクションを有効にします。
        #gc.set_debug(gc.DEBUG_LEAK) # メモリリークをデバッグするときに指定

        ''' サンプルデータを返す関数
        本実装では発話単位でバッチを作成するため，
        idx=発話番号である．
        '''
        # 特徴量系列のフレーム数
        feat_len = self.feat_len_list[idx]

        # 特徴量データを特徴量ファイルから読み込む
        (feat, sr ) = wavread( self.feat_list[idx] )

        # splicing: 前後 n フレームの特徴量を結合する
        org_feat = feat.copy()
        for n in range(-self.splice, self.splice+1):
            # 元々の特徴量を n フレームずらす
            tmp = np.roll(org_feat, n, axis=0)
            if n < 0:
                # 前にずらした場合は
                # 終端nフレームを0にする
                tmp[n:] = 0
            elif n > 0:
                # 後ろにずらした場合は
                # 始端nフレームを0にする
                tmp[:n] = 0
            else:
                continue
            # ずらした特徴量を次元方向に
            # 結合する
            feat = np.hstack([feat,tmp])

        # 特徴量データのフレーム数を最大フレーム数に
        # 合わせるため，0で埋める
        pad_len = self.max_feat_len - feat_len
        feat = np.pad(feat,
                      [(0, pad_len)],
                      mode='constant',
                      constant_values=0)

        # 発話ID
        utt_id = self.id_list[idx]

        feat = torch.unsqueeze( torch.tensor( feat, requires_grad = False ), dim = 1 )

        #dump_garbage()

        # 特徴量、フレーム数，
        # 発話IDを返す
        return (feat, 
               feat_len,
               utt_id)

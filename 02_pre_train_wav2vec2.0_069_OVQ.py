# -*- coding: utf-8 -*-

#
# RNN Attention Encoder-Decoderモデルを学習します．
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import random


# 作成したDatasetクラスをインポート
from my_dataset_pre import SequenceDataset

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# プロット用モジュール(matplotlib)をインポート
import matplotlib.pyplot as plt

# 認識エラー率を計算するモジュールをインポート
import levenshtein

# モデルの定義をインポート
from my_model import MyE2EModel

# json形式の入出力を行うモジュールをインポート
import json

# os, sys モジュールをインポート
import os
import sys
import gc
import psutil

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

class Wav2vec2Loss(nn.Module):
    def __init__(self, 
        contrastive_loss_temperature = 0.1,
        num_contrastive_loss_negative_samples = 25,
        num_code_vector_groups = 2,
        num_code_vectors_per_group = 320,
        loss_alpha = 0.1,
        ):
        super().__init__()
        self.k = contrastive_loss_temperature
        self.K = num_contrastive_loss_negative_samples
        self.cos = nn.CosineSimilarity(dim=-1)
        self.G = num_code_vector_groups
        self.V = num_code_vectors_per_group
        self.a = loss_alpha

    def forward(self, encoder_out, quantized_features, perplexity, time_mask_indices):
        target_encoder_out = encoder_out[time_mask_indices]
        labels = quantized_features[time_mask_indices]

        # Number of targets per batch
        num_targets_per_batch = [int(time_mask_indices[i].sum()) for i in range(time_mask_indices.size(0))]

        # Make negative samples
        negative_samples = self.negative_sampler(labels, num_targets_per_batch)
        #negative_samples = labels
        negative_samples = torch.cat([labels.unsqueeze(1), negative_samples], dim=1)

        contrastive_loss, pos_sim, neg_sim = self.contrastive_loss(target_encoder_out, labels, negative_samples)
        diversity_loss = self.diversity_loss(perplexity)

        loss = contrastive_loss + self.a * diversity_loss

        return loss, contrastive_loss.item(), diversity_loss.item(), pos_sim, neg_sim

    def contrastive_loss(
            self,
            targets: torch.Tensor,
            labels: torch.Tensor,
            negative_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            targets (torch.Tensor): with shape `(N, D)`
            labels (torch.Tensor): with shape `(N, D)`
            negative_samples (torch.Tensor): with shape `(N, K, D)`

        Returns:
            torch.Tensor with shape `(1)`
        """

        sim = self.cos( targets, labels )
        similarity = torch.exp( sim / self.k)
        sim_mean = torch.mean( sim )

        neg_sim = self.cos( targets.unsqueeze(1), negative_samples )
        negative_similarity = torch.sum(torch.exp(( neg_sim / self.k)), dim=1)
        only_neg_sim_sum = torch.sum( neg_sim, dim = 1 ) - sim
        neg_sim_2mean = only_neg_sim_sum / ( neg_sim.size(1) - 1 )
        neg_sim_2mean_1mean = torch.mean( neg_sim_2mean, dim = 0 )
        contrastive_loss = -torch.log(similarity / negative_similarity).mean()

        return contrastive_loss, sim_mean.item(), neg_sim_2mean_1mean.item()

    def diversity_loss(self, perplexity: torch.Tensor) -> torch.Tensor:
        """
        Args:
            perplexity (torch.Tensor): with shape `(G, V)`

        Returns:
            torch.Tensor with shape `(1)`
        """
        log_perplexity = torch.log(perplexity)
        entropy = torch.sum(perplexity*log_perplexity, dim=-1)
        diversity_loss = torch.sum(entropy) / (self.G * self.V)

        return diversity_loss

    def negative_sampler(self, label: torch.Tensor, num_targets_per_batch: list[int]):
        """
        Args:
            label (torch.Tensor): with shape `(N, D)`
            num_targets_per_batch (list[int]): Number of targets per batch.

        Returns:
            torch.Tensor with shape `(N, K, D)'

        """
        negative_samples = []
        start_idx = 0
        self.K = min( num_targets_per_batch ) - 1
        for num_targets in num_targets_per_batch:
            negative_sample_candidate_indices = torch.arange(
                num_targets, device=label.device
            ).unsqueeze(0).repeat(num_targets, 1)

            diagonal = torch.eye(num_targets)

            # Pull yourself from the list of candidates. `(N, N)` -> `(N, N-1)`
            negative_sample_candidate_indices = negative_sample_candidate_indices[diagonal == 0].view(num_targets, -1)
            negative_sample_candidate_indices += start_idx
            
            #print( "num_targets_per_batch:", num_targets_per_batch )
            #print( "num_targeets:", num_targets )

            where_negative_sample = (
                torch.tensor([i for i in range(num_targets) for _ in range(self.K)]),
                torch.tensor(
                    [random.sample(list(range(num_targets - 1)), k=self.K) for _ in range(num_targets)]).flatten()
            )

            # `(K * N)`
            negative_sample_indices = negative_sample_candidate_indices[where_negative_sample]

            negative_samples.append(label[negative_sample_indices])
            start_idx += num_targets

        negative_samples = torch.cat(negative_samples).view(label.size(0), self.K, -1)

        return negative_samples

def main():

    #torch.autograd.set_detect_anomaly(True)
    
    #
    # 設定ここから
    #

    # トークンの単位
    # phone:音素  kana:かな  char:キャラクター
    unit = 'char'

    # 学習データの特徴量(feats.scp)が存在するディレクトリ
    feat_dir_train = '../01compute_features/wav/train_large'
    # 開発データの特徴量(Feats.scp)が存在するディレクトリ
    feat_dir_dev = '../01compute_features/wav/dev'


    # 実験ディレクトリ
    # train_set_name = 'train_small' or 'train_large'
    train_set_name = os.path.basename(feat_dir_train) 
    exp_dir = './exp_' + os.path.basename(feat_dir_train) 

    # 学習/開発データの特徴量リストファイル
    feat_scp_train = os.path.join(feat_dir_train, 'feats0.scp')
    feat_scp_dev = os.path.join(feat_dir_dev, 'feats0.scp')
    
    # 学習結果を出力するディレクトリ
    output_dir = os.path.join(exp_dir, unit+'_model_wav2vec2.0_069_2')

    # ミニバッチに含める発話数
    batch_size = 8

    # 最大エポック数
    max_num_epoch = 100
    
    # feature_extractor の conv 層の設定
    fe_conv_layer = 7
    fe_conv_channel = [512,512,512,512,512,512,512]
    fe_conv_kernel = [10,3,3,3,3,2,2]
    fe_conv_stride = [5,2,2,2,2,2,2]
    fe_conv_dropout = 0.1
    fe_out_dim = 512

    # Encoder の Attention に入力するための conv 層の設定
    PCE_kernel_size = 128
    PCE_groups = 16

    # Encoderの設定
    # レイヤー数
    enc_num_layers = 6
    # encoder の head の数
    enc_num_heads = 8
    # Encoder の Attention block の次元数
    enc_att_hidden_dim = 512
    # encoder 入力の時間の最大数
    enc_input_maxlen = 3000
    # Encoder の Attention Bolock の kernel_size
    enc_att_kernel_size = [5,1]
    # Encoder の Attention Block の filter_size
    enc_att_filter_size = 2048
    # Encoder の dropout
    enc_dropout = 0.1
    
    #ダウンサンプリングの割合
    ds_rate = 0.25
    
    # マスクの数と連続フレーム数
    n_mask = 0.065
    n_consec = 10
    
    # コードブックG のエントリの数V
    entryV = 320
    #コードブックの数
    num_codebook = 2

    #Gumbel Softmax の温度
    tau = 2.0
    temprature_multi = 0.999995
    # GumbleSoftmax の最小温度
    tau_min = 0.5

    # Decoderの設定
    # attnesion blockのレイヤー数
    dec_num_layers = 6
    # decoder の head の数
    dec_num_heads = 8
    # Decoder の Attention block の次元数
    dec_att_hidden_dim = 512
    # decoder 入力( decoder targets, encoder_outs ではない）の時間の最大数
    dec_target_maxlen = 1000
    # Deccoder の Attention Bolock の kernel_size
    dec_att_kernel_size = [5,1]
    # Decoder の Attention Block の filter_size
    dec_att_filter_size = 2048
    # Decoder の dropout
    dec_dropout = 0.1

    # 初期学習率
    initial_learning_rate = 1e-5

    # Clipping Gradientの閾値
    clip_grad_threshold = 5.0
    #clip_grad_threshold = 1.0

    # 学習率の減衰やEarly stoppingの
    # 判定を開始するエポック数
    # (= 最低限このエポックまではどれだけ
    # validation結果が悪くても学習を続ける)
    lr_decay_start_epoch = 7

    # 学習率を減衰する割合
    # (減衰後学習率 <- 現在の学習率*lr_decay_factor)
    # 1.0以上なら，減衰させない
    lr_decay_factor = 0.5

    # Early stoppingの閾値
    # 最低損失値を更新しない場合が
    # 何エポック続けば学習を打ち切るか
    early_stop_threshold = 3

    # 学習過程で，認識エラー率を計算するか否か
    # 認識エラー率の計算は時間がかかるので注意
    # (ここではvalidationフェーズのみTrue(計算する)にしている)
    evaluate_error = {'train': True, 'validation': True}

    #
    # 設定ここまで
    #
    
    # Attention重み行列情報の保存先
    out_att_dir = os.path.join(output_dir, 'att_matrix')
    
    # 出力ディレクトリが存在しない場合は作成する
    os.makedirs(out_att_dir, exist_ok=True)

    # 設定を辞書形式にする
    config = {'fe_conv_layer' : fe_conv_layer,
              'fe_conv_channel' : fe_conv_channel,
              'fe_conv_kernel' : fe_conv_kernel,
              'fe_conv_stride' : fe_conv_stride,
              'fe_conv_dropout_rate' : fe_conv_dropout,
              'fe_out_dim' : fe_out_dim,
              'PCE_kernel_size' : PCE_kernel_size,
              'PCE_groups' : PCE_groups,
              'enc_num_layers': enc_num_layers,
              'enc_num_heads': enc_num_heads,
              'enc_input_maxlen' : enc_input_maxlen,
              'enc_att_hidden_dim': enc_att_hidden_dim,
              'enc_att_kernel_size': enc_att_kernel_size,
              'enc_att_filter_size': enc_att_filter_size,
              'downsampling_rate': ds_rate,
              'n_mask': n_mask,
              'n_consec': n_consec,
              'entryV': entryV,
              'num_codebook': num_codebook,
              'tau': tau,
              'temprature_multi': temprature_multi,
              'tau_min': tau_min,
              'enc_dropout_rate': enc_dropout,
              'dec_num_layers': dec_num_layers,
              'dec_num_heads': dec_num_heads,
              'dec_target_maxlen': dec_target_maxlen,
              'dec_att_hidden_dim': dec_att_hidden_dim,
              'dec_att_kernel_size': dec_att_kernel_size,
              'dec_att_filter_size': dec_att_filter_size,
              'dec_dropout_rate': dec_dropout,
              'batch_size': batch_size,
              'max_num_epoch': max_num_epoch,
              'clip_grad_threshold': clip_grad_threshold,
              'initial_learning_rate': initial_learning_rate,
              'lr_decay_start_epoch': lr_decay_start_epoch, 
              'lr_decay_factor': lr_decay_factor,
              'early_stop_threshold': early_stop_threshold
             }

    # 設定をJSON形式で保存する
    conf_file = os.path.join(output_dir, 'config.json')
    with open(conf_file, mode='w', encoding='utf-8' ) as f:
        json.dump(config, f, indent=4)

    # 次元数の情報を得る
    feat_dim = 1

    #ダミーの変数
    sos_id = 1e9

    # トークン数(blankを含む)ダミー
    num_tokens = 1e9
    
    # ニューラルネットワークモデルを作成する
    # 入力の次元数は特徴量の次元数，
    # 出力の次元数はトークン数となる
    model = MyE2EModel(dim_in=feat_dim,
                       dim_out=num_tokens,
                       fe_conv_layer=fe_conv_layer,
                       fe_conv_channel=fe_conv_channel,
                       fe_conv_kernel=fe_conv_kernel,
                       fe_conv_stride=fe_conv_stride,
                       fe_conv_dropout_rate=fe_conv_dropout,
                       fe_out_dim=fe_out_dim,
                       PCE_kernel_size=PCE_kernel_size,
                       PCE_groups=PCE_groups,
                       enc_num_layers = enc_num_layers,
                       enc_att_hidden_dim=enc_att_hidden_dim,
                       enc_num_heads = enc_num_heads,
                       enc_input_maxlen = enc_input_maxlen, 
                       enc_att_kernel_size=enc_att_kernel_size,
                       enc_att_filter_size=enc_att_filter_size,
                       enc_dropout_rate = enc_dropout,
                       ds_rate = ds_rate,
                       n_mask = n_mask,
                       n_consec = n_consec,
                       entryV = entryV,
                       num_codebook = num_codebook,
                       tau = tau,
                       temprature_multi = temprature_multi,
                       tau_min = tau_min,
                       dec_num_layers = dec_num_layers,
                       dec_att_hidden_dim=dec_att_hidden_dim,
                       dec_num_heads = dec_num_heads, 
                       dec_target_maxlen = dec_target_maxlen,
                       dec_att_kernel_size = dec_att_kernel_size,
                       dec_att_filter_size = dec_att_filter_size,
                       dec_dropout_rate = dec_dropout,
                       sos_id=sos_id, 
                       )
    print(model)

    # オプティマイザを定義
    optimizer = optim.AdamW(model.parameters(),
                               lr=initial_learning_rate,
                               eps = 1e-6,
                               weight_decay = 0.1
                               )

    # 訓練/開発データのデータセットを作成する
    train_dataset = SequenceDataset(feat_scp_train,
                                    )

    # 開発データのデータセットを作成する
    dev_dataset = SequenceDataset(feat_scp_dev,
                                  )

    # 訓練データのDataLoaderを呼び出す
    # 訓練データはシャッフルして用いる
    #  (num_workerは大きい程処理が速くなりますが，
    #   PCに負担が出ます．PCのスペックに応じて
    #   設定してください)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)
    # 開発データのDataLoaderを呼び出す
    # 開発データはデータはシャッフルしない
    dev_loader = DataLoader(dev_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4)

    del train_dataset
    del dev_dataset
    gc.collect()


    # CUDAが使える場合はモデルパラメータをGPUに，
    # そうでなければCPUに配置する
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    model = model.to(device, non_blocking=True)

    #ダウンサンプリングの割合
    ds_rate = torch.tensor( ds_rate, device=torch.device(device) )
    
    # マスクの数と連続フレーム数
    n_mask = torch.tensor(n_mask, device=torch.device(device))
    n_consec = torch.tensor(n_consec, device=torch.device(device))
    
    # コードブックG のエントリの数V
    entryV = torch.tensor( entryV, device=torch.device(device))
    
    #Gumbel Softmax の温度
    tau = torch.tensor(tau, device=torch.device(device))
    temprature_multi = torch.tensor(temprature_multi, device=torch.device(device))
    tau_min = torch.tensor(tau_min, device=torch.device(device))

    # モデルをトレーニングモードに設定する
    model.train()

    # 訓練データの処理と開発データの処理を
    # for でシンプルに記述するために，辞書データ化しておく
    dataset_loader = {'train': train_loader,
                      'validation': dev_loader}

    # 各エポックにおける損失値と誤り率の履歴
    loss_history = {'train': [],
                    'validation': []}
    error_history = {'train': [],
                     'validation': []}

    # 本プログラムでは，validation時の損失値が
    # 最も低かったモデルを保存する．
    # そのため，最も低い損失値，
    # そのときのモデルとエポック数を記憶しておく
    best_loss = -1
    best_model = None
    best_epoch = 0
    # Early stoppingフラグ．Trueになると学習を打ち切る
    early_stop_flag = False
    # Early stopping判定用(損失値の最低値が
    # 更新されないエポックが何回続いているか)のカウンタ
    counter_for_early_stop = 0

    # ログファイルの準備
    log_file = open(os.path.join(output_dir,
                                 'log.txt'),
                                 mode='w', encoding='utf-8' )
    log_file.write('epoch\ttrain loss\t'\
                   'train err\tvalid loss\tvalid err')

    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor = 0.01, total_iters=100)

    wav2vec2loss = Wav2vec2Loss(
        contrastive_loss_temperature = 0.1,
        num_contrastive_loss_negative_samples = 25,
        num_code_vector_groups = 2,
        num_code_vectors_per_group = 320,
        loss_alpha = 100.0,
    )

    # エポックの数だけループ
    for epoch in range(max_num_epoch):
        # early stopフラグが立っている場合は，
        # 学習を打ち切る
        if early_stop_flag:
            print('    Early stopping.'\
                  ' (early_stop_threshold = %d)' \
                  % (early_stop_threshold))
            log_file.write('\n    Early stopping.'\
                           ' (early_stop_threshold = %d)' \
                           % (early_stop_threshold))
            break

        # エポック数を表示
        print('epoch %d/%d:' % (epoch+1, max_num_epoch))
        log_file.write('\n%d\t' % (epoch+1))

        # trainフェーズとvalidationフェーズを交互に実施する
        for phase in ['train', 'validation']:
            # このエポックにおける累積損失値と発話数
            total_loss = 0
            total_utt = 0
            # このエポックにおける累積認識誤り文字数と総文字数
            total_error = 0
            total_token_length = 0

            # 各フェーズのDataLoaderから1ミニバッチ
            # ずつ取り出して処理する．
            # これを全ミニバッチ処理が終わるまで繰り返す．
            # ミニバッチに含まれるデータは，
            # 音声特徴量，ラベル，フレーム数，
            # ラベル長，発話ID
            n_batch = 0
            total_pos_sim = 0
            total_neg_sim = 0
            total_lm = 0
            total_ld = 0
            for (features, feat_lens, utt_ids) \
                    in dataset_loader[phase]:
                n_batch += 1
                    
                # 現時点でラベルのテンソルサイズは
                # [発話数 x 全データの最大ラベル長]
                # これを[発話数 x バッチ内の最大ラベル長]
                # に切る。(decoder部の冗長な処理を少なくするため。)
                features = features[:,:torch.max(feat_lens)]

                # CUDAが使える場合はデータをGPUに，
                # そうでなければCPUに配置する
                features, feat_lens = features.to(device, non_blocking=True), feat_lens.to(device, non_blocking=True)

                # 勾配をリセット
                #optimizer.zero_grad()

                # モデルの出力を計算(フォワード処理)
                if phase == 'train':
                    outputs, outputs_lens, pgv_bar, mask, quantized_vector = model(features, feat_lens, tau )
                else:
                    with torch.no_grad():
                        outputs, outputs_lens, pgv_bar, mask, quantized_vector = model(features, feat_lens, tau )


                # loss の計算など
                loss, lm, ld, pos_sim, neg_sim = wav2vec2loss( outputs, quantized_vector, pgv_bar, mask )

                total_pos_sim += pos_sim
                total_neg_sim += neg_sim
                total_lm += lm
                total_ld += ld

                # 訓練フェーズの場合は，誤差逆伝搬を実行し，
                # モデルパラメータを更新する
                if phase == 'train':
                    # 勾配を計算する
                    #loss.backward(retain_graph=True)
                    loss.backward()
                    # Cliping Gradient により勾配が
                    # 閾値以下になるよう調整する
                    torch.nn.utils.clip_grad_norm_(\
                                              model.parameters(),
                                              clip_grad_threshold)
                    # オプティマイザにより，パラメータを更新する
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    tau = tau * temprature_multi
                    #with warmup_scheduler.dampening():
                    #    lr_scheduler.step( (epoch + 1 ) + n_batch / iters )

                # 損失値を累積する
                total_loss += loss.item()
                # 処理した発話数をカウントする
                total_utt += outputs.size(0)

                if n_batch % 5 == 0:
                    print( "n_batch:{:>2d},phase:{:>9s},lm:{:>.3e},ld:{:>.3e},loss:{:>.3e},pos_avg:{:>.3e},neg_avg:{:>.3e},lr:{:>.3e}".format( n_batch, phase, \
                        total_lm / n_batch, total_ld /  n_batch, total_loss /  n_batch, total_pos_sim / n_batch, total_neg_sim / n_batch, optimizer.param_groups[0]['lr'] ) )

                #print("2 memory_usage:", get_memory_usage())
            
            torch.cuda.empty_cache()

            #
            # このフェーズにおいて，1エポック終了
            # 損失値，認識エラー率，モデルの保存等を行う
            # 

            # 損失値の累積値を，処理した発話数で割る
            #epoch_loss = total_loss / total_utt
            epoch_loss = total_loss / n_batch
            # 画面とログファイルに出力する
            print("n_batch:{}".format( n_batch ))
            print('    %s loss: %f' \
                  % (phase, epoch_loss))
            log_file.write('%.6f\t' % (epoch_loss))
            # 履歴に加える
            loss_history[phase].append(epoch_loss)
            #
            # validationフェーズ特有の処理を train で行う。
            #
            #if phase == 'validation':
            if phase == 'train':
                if epoch == 0 or best_loss > epoch_loss:
                    # 損失値が最低値を更新した場合は，
                    # その時のモデルを保存する
                    best_loss = epoch_loss
                    torch.save({'model_state_dict': model.state_dict(), 
                               'optimizer_state_dict': optimizer.state_dict(),},
                               output_dir+'/best_model.pt')
                    best_epoch = epoch
                    # Early stopping判定用の
                    # カウンタをリセットする
                    counter_for_early_stop = 0
                else:
                    # 最低値を更新しておらず，
                    if epoch+1 >= lr_decay_start_epoch:
                        # かつlr_decay_start_epoch以上の
                        # エポックに達している場合
                        if counter_for_early_stop+1 \
                               >= early_stop_threshold:
                            # 更新していないエポックが，
                            # 閾値回数以上続いている場合，
                            # Early stopping フラグを立てる
                            early_stop_flag = True
                        else:
                            #Early stopping条件に
                            #達していない場合は
                            #学習率を減衰させて学習続行
                            if lr_decay_factor < 1.0:
                                for i, param_group \
                                      in enumerate(\
                                      optimizer.param_groups):
                                    if i == 0:
                                        lr = param_group['lr']
                                        dlr = lr_decay_factor \
                                            * lr
                                        print('    (Decay '\
                                          'learning rate:'\
                                          ' %f -> %f)' \
                                          % (lr, dlr))
                                        log_file.write(\
                                          '(Decay learning'\
                                          ' rate: %f -> %f)'\
                                           % (lr, dlr))
                                    param_group['lr'] = dlr
                            # Early stopping判定用の
                            # カウンタを増やす
                            counter_for_early_stop += 1

        scheduler.step()

    #
    # 全エポック終了
    # 学習済みモデルの保存とログの書き込みを行う
    #
    print('---------------Summary'\
          '------------------')
    log_file.write('\n---------------Summary'\
                   '------------------\n')
   
    # 最終エポックのモデルを保存する
    torch.save({'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(), },
               os.path.join(output_dir,'final_model.pt'))
    print('Final epoch model -> %s/final_model.pt' \
          % (output_dir))
    log_file.write('Final epoch model ->'\
                   ' %s/final_model.pt\n' \
                   % (output_dir))

    # 最終エポックの情報
    for phase in ['train', 'validation']:
        # 最終エポックの損失値を出力
        print('    %s loss: %f' \
              % (phase, loss_history[phase][-1]))
        log_file.write('    %s loss: %f\n' \
                       % (phase, loss_history[phase][-1]))

    # ベストエポックの情報
    # (validationの損失が最小だったエポック)
    print('Best epoch model (%d-th epoch)'\
          ' -> %s/best_model.pt' \
          % (best_epoch+1, output_dir))
    log_file.write('Best epoch model (%d-th epoch)'\
          ' -> %s/best_model.pt\n' \
          % (best_epoch+1, output_dir))
    for phase in ['train', 'validation']:
        # ベストエポックの損失値を出力
        print('    %s loss: %f' \
              % (phase, loss_history[phase][best_epoch]))
        log_file.write('    %s loss: %f\n' \
              % (phase, loss_history[phase][best_epoch]))

    # 損失値の履歴(Learning Curve)グラフにして保存する
    fig1 = plt.figure()
    for phase in ['train', 'validation']:
        plt.plot(loss_history[phase],
                 label=phase+' loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    fig1.legend()
    fig1.savefig(output_dir+'/loss.png')

    # ログファイルを閉じる
    log_file.close()

#
# メイン関数
#
if __name__ == "__main__":

    main()


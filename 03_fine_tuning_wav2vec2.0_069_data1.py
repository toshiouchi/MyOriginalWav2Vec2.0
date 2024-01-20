# -*- coding: utf-8 -*-

#
# RNN Attention Encoder-Decoderモデルを学習します．
#
#MultiheadAttention の layer_norm と dropout 使わないようにに戻した。

# Pytorchを用いた処理に必要なモジュールをインポート
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import Sequential

# 作成したDatasetクラスをインポート
from my_dataset import SequenceDataset

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# プロット用モジュール(matplotlib)をインポート
import matplotlib.pyplot as plt

# 認識エラー率を計算するモジュールをインポート
import levenshtein

# モデルの定義をインポート
from my_model_ft import MyE2EModel

# json形式の入出力を行うモジュールをインポート
import json

# os, sysモジュールをインポート
import os
import sys

def ctc_simple_decode(int_vector, token_list):
    ''' 以下の手順で，フレーム単位のCTC出力をトークン列に変換する
        1. 同じ文字が連続して出現する場合は削除
        2. blank を削除
    int_vector: フレーム単位のCTC出力(整数値列)
    token_list: トークンリスト
    output:     トークン列
    '''
    # 出力文字列
    output = []
    # 一つ前フレームの文字番号
    prev_token = -1
    # フレーム毎の出力文字系列を前から順番にチェックしていく
    for n in int_vector:
        #print( " n:{}".format( n ))
        #print( " prev_token:{}".format( prev_token ))
        if n != prev_token:
            # 1. 前フレームと同じトークンではない
            if n != 0:
                # 2. かつ，blank(番号=0)ではない
                # --> token_listから対応する文字を抽出し，
                #     出力文字列に加える
                output.append(token_list[n])
                if token_list[n] == '<eos>':
                    break
            # 前フレームのトークンを更新
            prev_token = n
    return output


#
# メイン関数
#
if __name__ == "__main__":
    
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
    train_set_name = os.path.basename(feat_dir_train) 
    exp_dir = './exp_' + os.path.basename(feat_dir_train) 

    # 学習/開発データの特徴量リストファイル
    feat_scp_train = os.path.join(feat_dir_train, 'feats1.scp')
    feat_scp_dev = os.path.join(feat_dir_dev, 'feats1.scp')
    #feat_scp_train = os.path.join(feat_dir_train, 'feats0.scp')
    #feat_scp_dev = os.path.join(feat_dir_dev, 'feats0.scp')

    # 学習/開発データのラベルファイル
    label_train = os.path.join(exp_dir, 'data', unit,
                               'label_'+train_set_name+'1')
    label_dev = os.path.join(exp_dir, 'data', unit,
                             'label_dev1')
    #label_train = os.path.join(exp_dir, 'data', unit,
    #                           'label_'+train_set_name+'0')
    #label_dev = os.path.join(exp_dir, 'data', unit,
    #                         'label_dev0')



    # トークンリスト
    token_list_path = os.path.join(exp_dir, 'data', unit,
                                   'token_list')
    # 学習結果を出力するディレクトリ
    output_dir = os.path.join(exp_dir, unit+'_model_wav2vec2.0_069_2')

    # ミニバッチに含める発話数
    batch_size = 8

    # 最大エポック数
    max_num_epoch = 200
    

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

    # pre-train 時に出力した設定ファイル
    config_file = os.path.join(output_dir, 'config.json')
    
    # 設定ファイルを読み込む
    with open(config_file, mode='r') as f:
        config = json.load(f)

    # 読み込んだ設定を反映する

    fe_conv_layer = config['fe_conv_layer']
    fe_conv_channel = config['fe_conv_channel']
    fe_conv_kernel = config['fe_conv_kernel']
    fe_conv_stride = config['fe_conv_stride']
    fe_conv_dropout_rate = config['fe_conv_dropout_rate']
    fe_out_dim = config['fe_out_dim']
    PCE_kernel_size = config['PCE_kernel_size']
    PCE_groups = config['PCE_groups']
    enc_num_layers = config['enc_num_layers']
    enc_num_heads = config['enc_num_heads']
    enc_input_maxlen = config['enc_input_maxlen']
    enc_att_hidden_dim = config['enc_att_hidden_dim']
    enc_att_kernel_size = config['enc_att_kernel_size']
    enc_att_filter_size = config['enc_att_filter_size']
    ds_rate = config['downsampling_rate']
    n_mask = config['n_mask']
    n_consec = config['n_consec']
    entryV = config['entryV']
    num_codebook = config['num_codebook']
    tau = config['tau']
    temprature_multi = config['temprature_multi']
    tau_min = config['tau_min']
    n_mask = config['n_mask']
    n_consec = config['n_consec']
    entryV = config['entryV']
    enc_dropout = config['enc_dropout_rate']
    dec_num_layers = config['dec_num_layers']
    dec_num_heads = config['dec_num_heads']
    dec_target_maxlen = config['dec_target_maxlen']
    dec_att_hidden_dim = config['dec_att_hidden_dim']
    dec_att_kernel_size = config['dec_att_kernel_size']
    dec_att_filter_size = config['dec_att_filter_size']
    dec_dropout = config['dec_dropout_rate']
    
    
    # 次元数の情報を得る
    feat_dim = 1

    # トークンリストをdictionary型で読み込む
    # このとき，0番目は blank と定義する
    # (ただし，このプログラムではblankは使われない)
    token_list = {0: '<blank>'}
    with open(token_list_path, mode='r', encoding='utf-8' ) as f:
        # 1行ずつ読み込む
        for line in f: 
            # 読み込んだ行をスペースで区切り，
            # リスト型の変数にする
            parts = line.split()
            # 0番目の要素がトークン，1番目の要素がID
            token_list[int(parts[1])] = parts[0]

    # <eos>トークンをユニットリストの末尾に追加
    eos_id = len(token_list)
    token_list[eos_id] = '<eos>'
    # 本プログラムでは、<sos>と<eos>を
    # 同じトークンとして扱う
    #sos_id = eos_id
    sos_id = len(token_list)
    token_list[sos_id] = '<sos>'

    # トークン数(blankを含む)
    num_tokens = len(token_list)
    
    # ニューラルネットワークモデルを作成する
    # 入力の次元数は特徴量の次元数，
    # 出力の次元数はトークン数となる
    model = MyE2EModel(dim_in=feat_dim,
                       dim_out=num_tokens,
                       fe_conv_layer=fe_conv_layer,
                       fe_conv_channel=fe_conv_channel,
                       fe_conv_kernel=fe_conv_kernel,
                       fe_conv_stride=fe_conv_stride,
                       fe_conv_dropout_rate=fe_conv_dropout_rate,
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

    # オプティマイザを定義
    optimizer = optim.AdamW(model.parameters(),
                               lr=initial_learning_rate,
                               )
                               
    # モデルのパラメータを読み込む
    model_file = os.path.join(output_dir, 'best_model.pt')
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #model.load_state_dict(torch.load(model_file))
    #model = MyE2EModel.from_pretrained( model_file )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # optimizerのstateを現在のdeviceに移す。これをしないと、保存前後でdeviceの不整合が起こる可能性がある。
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device, non_blocking=True)
    #epoch = checkpoint['epoch']
    #loss = checkpoint['loss']

    print(model)
    
    # モデルの最終線形層を修正する。
    model.classifier = nn.Linear( dec_att_hidden_dim, num_tokens )

    # 訓練/開発データのデータセットを作成する
    train_dataset = SequenceDataset(feat_scp_train,
                                    label_train,
                                    )

    # 開発データのデータセットを作成する
    dev_dataset = SequenceDataset(feat_scp_dev,
                                  label_dev,
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

    # CTC損失関数を呼び出す．
    # blankは0番目と定義する．
    criterion = nn.CTCLoss(blank=0, reduction='mean')    
    

    # CUDAが使える場合はモデルパラメータをGPUに，
    # そうでなければCPUに配置する
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    print( "device:", device )
    model = model.to(device, non_blocking=True)

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

    #ダミー変数
    tau = 2.0

    optimizer.param_groups[0]['lr'] = initial_learning_rate

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
            for (features, labels, feat_lens,
                 label_lens, utt_ids) \
                    in dataset_loader[phase]:
                n_batch += 1

                # 現時点でラベルのテンソルサイズは
                # [発話数 x 全データの最大ラベル長]
                # これを[発話数 x バッチ内の最大ラベル長]
                # に切る。(decoder部の冗長な処理を少なくするため。)
                features = features[:,:torch.max(feat_lens)]
                labels = labels[:,:torch.max(label_lens)]

                # CUDAが使える場合はデータをGPUに，
                # そうでなければCPUに配置する
                features, labels = \
                    features.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                # 勾配をリセット
                optimizer.zero_grad()

                # モデルの出力を計算(フォワード処理)
                #if phase == 'train':
                #    outputs, outputs_lens = model(features, feat_lens, tau )
                #else:
                #    with torch.no_grad():
                #        outputs, outputs_lens = model(features, feat_lens, tau )
                        
                outputs, outputs_lens = model(features, feat_lens, tau )

                outputs = F.log_softmax( outputs, dim=2 )

                # クロスエントロピー損失関数の入力は
                # [(バッチ）ｘ（クラス）ｘ（時間）」なので、transpose する。
                # target は、hot_vector にしないで、「（バッチ）×（時間）」のままで良い。
                #loss = criterion( outputs.transpose(1,2).to(device, non_blocking=True), dec_target.to(device, non_blocking=True) )
                # 損失値を計算する．このとき，CTCLossへの入力は
                # [フレーム数 x バッチサイズ x クラス数] 
                # である必要があるため，テンソルの0軸と1軸を
                # 転置(transpose(0,1))した上で入力する
                T = outputs.size(1)
                outputs_lens[outputs_lens > T] = T
                out_lens = outputs_lens
                loss = criterion(outputs.transpose(0, 1),labels,out_lens,label_lens)
                #loss *= np.mean(label_lens.numpy())

                #start_backward = time.time()
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
                
                # 認識エラーの算出をTrueにしている場合は，算出する
                if evaluate_error[phase]:
                    # バッチ内の1発話ごとに誤りを計算
                    for n in range(outputs.size(0)):
                        # 各ステップのデコーダ出力を得る
                        _, hyp_per_frame = torch.max(outputs[n], 1)
                        # numpy.array型に変換
                        hyp_per_frame = hyp_per_frame.cpu().numpy()
                        # 認識結果の文字列を取得
                        hypothesis = \
                           ctc_simple_decode(hyp_per_frame,
                                      token_list)
                        # 正解の文字列を取得
                        reference = []
                        for m in labels[n][:label_lens[n]].cpu().numpy().astype( np.int32 ):
                            if token_list[m] != "<sos>" and token_list[m] != "<eos>":
                                reference.append(token_list[m])
                        #print( *reference, sep="" )
                        # 認識誤りを計算
                        (error, substitute, 
                         delete, insert, ref_length) = \
                            levenshtein.calculate_error(hypothesis,
                                                        reference)
                        # 誤り文字数を累積する
                        total_error += error
                        # 文字の総数を累積する
                        total_token_length += ref_length
                        
                        if n < 4 and n_batch == len( dataset_loader[phase] ):
                            print( "%12s, reference :%s" % (phase,''.join(reference) ) )
                            print( "%12s, hypothesis:%s" % (phase,''.join(hypothesis) ) )
                # 損失値を累積する
                total_loss += loss.item()
                # 処理した発話数をカウントする
                total_utt += outputs.size(0)

                if n_batch % 5 == 0:
                    print( "n_batch:{:>2d},phase:{:>9s},loss:{:>.3e},lr:{:>.3e}".format( n_batch, phase, total_loss /  n_batch,  optimizer.param_groups[0]['lr'] ) )

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
            
            # 認識エラー率を計算する
            if evaluate_error[phase]:
                # 総誤りトークン数を，
                # 総トークン数で割ってエラー率に換算
                epoch_error = 100.0 * total_error \
                            / total_token_length
                # 画面とログファイルに出力する
                print('    %s token error rate: %f %%' \
                    % (phase, epoch_error))
                log_file.write('%.6f\t' % (epoch_error))
                # 履歴に加える
                error_history[phase].append(epoch_error)
            else:
                # エラー率を計算していない場合
                log_file.write('     ---     \t')
            
            #
            # validationフェーズ特有の処理
            #
            #if phase == 'validation':
            if phase == 'train':
                if epoch == 0 or best_loss > epoch_loss:
                    # 損失値が最低値を更新した場合は，
                    # その時のモデルを保存する
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), 
                               output_dir+'/best_model_ft.pt')
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
                            # Early stopping条件に
                            # 達していない場合は
                            # 学習率を減衰させて学習続行
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

    #
    # 全エポック終了
    # 学習済みモデルの保存とログの書き込みを行う
    #
    print('---------------Summary'\
          '------------------')
    log_file.write('\n---------------Summary'\
                   '------------------\n')

    # 最終エポックのモデルを保存する
    torch.save(model.state_dict(), 
               os.path.join(output_dir,'final_model_ft.pt'))
    print('Final epoch model -> %s/final_model_ft.pt' \
          % (output_dir))
    log_file.write('Final epoch model ->'\
                   ' %s/final_model_ft.pt\n' \
                   % (output_dir))

    # 最終エポックの情報
    for phase in ['train', 'validation']:
        # 最終エポックの損失値を出力
        print('    %s loss: %f' \
              % (phase, loss_history[phase][-1]))
        log_file.write('    %s loss: %f\n' \
                       % (phase, loss_history[phase][-1]))
        
        # 最終エポックのエラー率を出力    
        if evaluate_error[phase]:
            print('    %s token error rate: %f %%' \
                % (phase, error_history[phase][-1]))
            log_file.write('    %s token error rate: %f %%\n' \
                % (phase, error_history[phase][-1]))
        else:
            print('    %s token error rate: (not evaluated)' \
                % (phase))
            log_file.write('    %s token error rate: '\
                '(not evaluated)\n' % (phase))
        

    # ベストエポックの情報
    # (validationの損失が最小だったエポック)
    print('Best epoch model (%d-th epoch)'\
          ' -> %s/best_model_ft.pt' \
          % (best_epoch+1, output_dir))
    log_file.write('Best epoch model (%d-th epoch)'\
          ' -> %s/best_model_ft.pt\n' \
          % (best_epoch+1, output_dir))
    for phase in ['train', 'validation']:
        # ベストエポックの損失値を出力
        print('    %s loss: %f' \
              % (phase, loss_history[phase][best_epoch]))
        log_file.write('    %s loss: %f\n' \
              % (phase, loss_history[phase][best_epoch]))
        
        # ベストエポックのエラー率を出力
        if evaluate_error[phase]:
            print('    %s token error rate: %f %%' \
                  % (phase, error_history[phase][best_epoch]))
            log_file.write('    %s token error rate: %f %%\n' \
                  % (phase, error_history[phase][best_epoch]))
        else:
            print('    %s token error rate: '\
                  '(not evaluated)' % (phase))
            log_file.write('    %s token error rate: '\
                  '(not evaluated)\n' % (phase))
        

    # 損失値の履歴(Learning Curve)グラフにして保存する
    fig1 = plt.figure()
    for phase in ['train', 'validation']:
        plt.plot(loss_history[phase],
                 label=phase+' loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    fig1.legend()
    fig1.savefig(output_dir+'/ft_loss.png')
    
    # 認識誤り率の履歴グラフにして保存する
    fig2 = plt.figure()
    for phase in ['train', 'validation']:
        if evaluate_error[phase]:
            plt.plot(error_history[phase],
                     label=phase+' error')
    plt.xlabel('Epoch')
    plt.ylabel('Error [%]')
    fig2.legend()
    fig2.savefig(output_dir+'/f_error.png')
    
    # ログファイルを閉じる
    log_file.close()


# -*- coding: utf-8 -*-
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
#gpuの確認
print(torch.cuda.is_available())

import torchtext
from torchtext.data.utils import get_tokenizer

toke = get_tokenizer('spacy')
toke('I am N')

import torchtext
from torchtext.data.utils import get_tokenizer

#テキストに処理を行うFieldを定義
#fix_lengthはtokenの数
TEXT = torchtext.data.Field(sequential=True, use_vocab=True, tokenize=get_tokenizer('spacy'),
                            lower=True, include_lengths=True, batch_first=True, fix_length=37)

LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

#pandasでcsvを保存するときに、labelをintでキャストしておかないとエラーでるから注意
train_ds, val_ds, test_ds = torchtext.data.TabularDataset.splits(
    path='drive/My Drive/dataset/CBET/ekman', train='train.csv', validation='val.csv',
    test='test.csv', format='csv', fields=[('text', TEXT), ('Label', LABEL)])

#ボキャブラリを作成する
TEXT.build_vocab(train_ds)
#TEXT.build_vocab(train_ds, vectors=japanese_word2vec_vectors) #学習ずみの分散表現を使う場合
print(TEXT.vocab.stoi)

#データローダーを作成
train_dl = torchtext.data.Iterator(train_ds, batch_size=64, train=True)
val_dl = torchtext.data.Iterator(val_ds, batch_size=64, train=False, sort=False)
test_dl = torchtext.data.Iterator(test_ds, batch_size=64, train=False, sort=False)

#テスト
batch = next(iter(val_dl))
print(len(batch.text[0][0]))
print(batch.Label)

class Embedder(nn.Module):
  def __init__(self, ntoken, d_model):
    super(Embedder, self).__init__()
    #tokenの数と、分散表現の次元数
    self.embeddings = nn.Embedding(ntoken, d_model)

  def forward(self, x):
    x = self.embeddings(x)
    return x


class LSTM_Layer(nn.Module):
  def __init__(self, d_model, hidden_size):
    super().__init__()
    self.hidden_size = hidden_size
    self.lstm = nn.LSTM(d_model, hidden_size, batch_first=True)

  #入力と(h, c)のタプル
  def forward(self, x, hidden):
    output, hidden = self.lstm(x, hidden)
    return output, hidden



class ClassificationHead(nn.Module):
  def __init__(self, d_model, output_dim):
    super().__init__()
    self.linear = nn.Linear(d_model, output_dim)
    nn.init.normal_(self.linear.weight, std=0.02)
    nn.init.normal_(self.linear.bias, 0)

  def forward(self, x):
    output = self.linear(x)
    return output

class LSTM_Classification(nn.Module):
  def __init__(self, batch_size, ntoken, d_model, hidden_size, output_dim):
    super().__init__()
    self.batch_size = batch_size
    self.hidden_size = hidden_size
    self.net1 = Embedder(ntoken, d_model)
    self.net2 = LSTM_Layer(d_model, hidden_size)
    self.net3 = ClassificationHead(hidden_size, output_dim)

  def forward(self, x, hidden):
    x1 = self.net1(x) # [batch_size, ntoken, d_model]
    x2, (ht, ct) = self.net2(x1, hidden) # [batch_size, ntoken, hidden_size], ([1, batch_size, hidden_size], [1, batch_size, hidden_size])
    #隠れ状態の最後を使う
    out = self.net3(ht[-1])
    return out, hidden

  def init_hidden(self, device):
    return (torch.zeros(1*1, self.batch_size, self.hidden_size, device=device),
            torch.zeros(1*1, self.batch_size, self.hidden_size, device=device))

#テスト
batch = next(iter(train_dl))
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

# モデル構築
net = LSTM_Classification(64, len(TEXT.vocab.stoi), 300, 256, 5)
hidden = (torch.zeros(1*1, 64, 256),
            torch.zeros(1*1, 64, 256))

#hidden = net.init_hidden(device)

# 入出力
x = batch.text[0]
x1, hidden = net(x, hidden)

print("入力のテンソルサイズ：", x.shape)
print("出力のテンソルサイズ：", x1.shape)
print(hidden[0].shape)

# -*- coding: utf-8 -*-
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#gpuの確認
print(torch.cuda.is_available())

#学習済みの分散表現をロードする
from torchtext.vocab import Vectors

english_fasttext_vectors = Vectors(name='drive/My Drive/wiki-news-300d-1M.vec')

print(english_fasttext_vectors.dim)
print(len(english_fasttext_vectors.itos))

import string
import re

# 以下の記号はスペースに置き換えます（カンマ、ピリオドを除く）。
# punctuationとは日本語で句点という意味です
print("区切り文字：", string.punctuation)
# !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

# 前処理


def preprocessing_text(text):
    # 改行コードを消去
    text = re.sub('<br />', '', text)

    # カンマ、ピリオド以外の記号をスペースに置換
    for p in string.punctuation:
        if (p == ".") or (p == ","):
            continue
        else:
            text = text.replace(p, " ")

    # ピリオドなどの前後にはスペースを入れておく
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    return text

# 分かち書き（今回はデータが英語で、簡易的にスペースで区切る）


def tokenizer_punctuation(text):
    return text.strip().split()


# 前処理と分かち書きをまとめた関数を定義
def tokenizer_with_preprocessing(text):
    text = preprocessing_text(text)
    ret = tokenizer_punctuation(text)
    return ret


# 動作を確認します
print(tokenizer_with_preprocessing('I like cats+'))

import torchtext
from torchtext.data.utils import get_tokenizer

#テキストに処理を行うFieldを定義
#fix_lengthはtokenの数
TEXT = torchtext.data.Field(sequential=True, use_vocab=True, tokenize=tokenizer_with_preprocessing,
                            lower=True, include_lengths=True, batch_first=True, fix_length=37)

LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

#pandasでcsvを保存するときに、labelをintでキャストしておかないとエラーでるから注意
train_ds, val_ds, test_ds = torchtext.data.TabularDataset.splits(
    path='drive/My Drive/dataset/CBET/ekman', train='train.csv', validation='val.csv',
    test='test.csv', format='csv', fields=[('Text', TEXT), ('Label', LABEL)])

#ボキャブラリを作成する
TEXT.build_vocab(train_ds, vectors=english_fasttext_vectors)

print(len(TEXT.vocab.stoi))

batch_size = 64
d_model = 300
num_filters = [100, 100, 100]
filter_sizes = [3, 4, 5]
num_unit = 100
output_dim = 5
dropout_rate = 0.5

#データローダを作成
train_dl = torchtext.data.Iterator(train_ds, batch_size=batch_size, train=True)
val_dl = torchtext.data.Iterator(val_ds, batch_size=batch_size, train=False, sort=False)
test_dl = torchtext.data.Iterator(test_ds, batch_size=batch_size, train=False, sort=False)

#テスト
batch = next(iter(val_dl))
print(len(batch.Text[0][0]))
print(batch.Label)

class Embedder(nn.Module):
  def __init__(self, text_embedding_vectors, dropout_rate):
    super(Embedder, self).__init__()
    #tokenの数と、分散表現の次元数
    self.embeddings = nn.Embedding.from_pretrained(
        embeddings=text_embedding_vectors, freeze=True)
    self.dropout = nn.Dropout(dropout_rate)

  def forward(self, x):
    x = self.embeddings(x).permute(0, 2, 1) #[batch, d_model, length]
    x = self.dropout(x)
    return x


class CNN_Layer(nn.Module):
  def __init__(self, d_model, num_filters, filter_sizes, dropout_rate):
    super().__init__()
    self.convs = nn.ModuleList([nn.Conv1d(d_model, nf, fs) for nf, fs in zip(num_filters, filter_sizes)])
    self.dropout = nn.Dropout(dropout_rate)

  def forward(self, x):
    x = [F.relu(conv(x).permute(0, 2, 1).max(1)[0]) for conv in self.convs]
    return x


class ClassificationHead(nn.Module):
  def __init__(self, num_unit, output_dim, dropout_rate):
    super().__init__()
    self.linear = nn.Linear(num_unit*3, output_dim)
    self.dropout = nn.Dropout(dropout_rate)
    nn.init.normal_(self.linear.weight, std=0.02)
    nn.init.normal_(self.linear.bias, 0)

  def forward(self, x):
    print(x[0].shape)
    print(x[1].shape)
    print(x[2].shape)
    print(torch.cat(x, 1).shape)
    # torch.cat(x, 1).shape -> [batch, sum(filter_sizes)]
    x = self.linear(torch.cat(x, 1)) # [batch, output_dim]
    output = self.dropout(x)
    return output

class CNN_Classification(nn.Module):
  def __init__(self, text_embedding_vectors, d_model, num_filters, filter_sizes, num_unit, droutput_dim, dropout_rate):
    super().__init__()
    self.net1 = Embedder(text_embedding_vectors, dropout_rate)
    self.net2 = CNN_Layer(d_model, num_filters, filter_sizes, dropout_rate)
    self.net3 = ClassificationHead(num_unit, output_dim, dropout_rate)

  def forward(self, x):
    x1 = self.net1(x) # [batch_size, ntoken, d_model]
    x2 = self.net2(x1)
    out = self.net3(x2)
    return out

#テスト
batch = next(iter(train_dl))
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

# モデル構築
net = CNN_Classification(TEXT.vocab.vectors, d_model, num_filters, filter_sizes,
                         num_unit, output_dim, dropout_rate)
#hidden = net.init_hidden(device)

# 入出力
x = batch.Text[0]
x1 = net(x)

print("入力のテンソルサイズ：", x.shape)
print("出力のテンソルサイズ：", x1.shape)

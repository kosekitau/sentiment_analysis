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
    test='test.csv', format='csv', fields=[('Text', TEXT), ('Label', LABEL)])

from torchtext.vocab import Vectors

english_fasttext_vectors = Vectors(name='drive/My Drive/wiki-news-300d-1M.vec')

print(english_fasttext_vectors.dim)
print(len(english_fasttext_vectors.itos))

#ボキャブラリを作成する
TEXT.build_vocab(train_ds, vectors=english_fasttext_vectors)

print(TEXT.vocab.stoi)

batch_size = 64
d_model = 300
hidden_size = 512
output_dim = 5
dropout_rate = 0.1

#データローダーを作成
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
    x = self.embeddings(x)
    x = self.dropout(x)
    return x


class LSTM_Layer(nn.Module):
  def __init__(self, d_model, hidden_size):
    super().__init__()
    self.hidden_size = hidden_size
    self.lstm = nn.LSTM(d_model, hidden_size, batch_first=True)

  #入力と(h, c)のタプル
  def forward(self, x):
    output, hidden = self.lstm(x)
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
  def __init__(self, text_embedding_vectors, d_model, hidden_size, output_dim, dropout_rate):
    super().__init__()
    self.hidden_size = hidden_size
    self.net1 = Embedder(text_embedding_vectors, dropout_rate)
    self.net2 = LSTM_Layer(d_model, hidden_size)
    self.net3 = ClassificationHead(hidden_size, output_dim)

  def forward(self, x):
    x1 = self.net1(x) # [batch_size, ntoken, d_model]
    x2, (ht, ct) = self.net2(x1) # [batch_size, ntoken, hidden_size], ([1, batch_size, hidden_size], [1, batch_size, hidden_size])
    #隠れ状態の最後を使う
    out = self.net3(ht[-1])
    return out

"""
#テスト


batch = next(iter(train_dl))
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

# モデル構築
net = LSTM_Classification(64, len(TEXT.vocab.stoi), 300, 256, 5)
hidden = (torch.zeros(1*1, 64, 256),
            torch.zeros(1*1, 64, 256))

#hidden = net.init_hidden(device)

# 入出力
x = batch.Text[0]
x1 = net(x, hidden)

print("入力のテンソルサイズ：", x.shape)
print("出力のテンソルサイズ：", x1.shape)
print(hidden[0].shape)
"""

dataloaders_dict = {'train': train_dl, 'val': val_dl}
criterion = nn.CrossEntropyLoss()
net = LSTM_Classification(TEXT.vocab.vectors, d_model, hidden_size, output_dim, dropout_rate)
net.train()

learning_rate = 10e-5
#optimizer = optim.SGD(net.parameters(), lr=learning_rate)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  net.to(device)

  torch.backends.cudnn.benchmark = True

  #各epoch
  for epoch in range(num_epochs):
    #訓練と評価
    for phase in ['train', 'val']:
      if phase == 'train':
        net.train()
      else:
        net.eval()

      epoch_loss = 0.0 #各epochの損失の和
      epoch_corrects = 0 #各epochの正解数

      for batch in (dataloaders_dict[phase]):
        inputs = batch.Text[0].to(device)
        labels = batch.Label.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase=='train'):
          #hidden = net.init_hidden(device) #LSTM隠れ状態の初期化
          outputs = net(inputs) #[batch_size, output_dim]

          loss = criterion(outputs, labels) #softmaxは中に入ってる
          _, preds = torch.max(outputs, 1)

          if phase == 'train':
            loss.backward() #勾配を計算
            optimizer.step() #パラメータを更新

          epoch_loss += loss.item()*inputs.size(0) #バッチ数をかけてあとでデータ量で割る
          epoch_corrects += torch.sum(preds == labels.data)

      #各epochのloss、正解数をだす
      epoch_loss = epoch_loss/len(dataloaders_dict[phase].dataset)
      epoch_acc = epoch_acc = epoch_corrects.double()/len(dataloaders_dict[phase].dataset)
      print('Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}'.format(epoch+1,
                                                                     num_epochs, phase, epoch_loss, epoch_acc))
  return net

num_epochs = 10
net_trained = train_model(net, dataloaders_dict,
                          criterion, optimizer, num_epochs=num_epochs)

net_trained = train_model(net_trained, dataloaders_dict,
                          criterion, optimizer, num_epochs=num_epochs)

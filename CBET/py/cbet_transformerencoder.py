# -*- coding: utf-8 -*-
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
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
                            lower=True, include_lengths=True, batch_first=True, fix_length=40,
                            init_token='<cls>', eos_token='<eos>')

LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

#pandasでcsvを保存するときに、labelをintでキャストしておかないとエラーでるから注意
train_ds, val_ds, test_ds = torchtext.data.TabularDataset.splits(
    path='drive/My Drive/dataset/CBET/ekman', train='train.csv', validation='val.csv',
    test='test.csv', format='csv', fields=[('Text', TEXT), ('Label', LABEL)])

#ボキャブラリを作成する
TEXT.build_vocab(train_ds, vectors=english_fasttext_vectors)

print(len(TEXT.vocab.stoi))
print(TEXT.vocab.stoi)

batch_size = 64
d_model = 300
output_dim = 5
N = 3
heads = 5
dropout_rate = 0.1

#データローダを作成
train_dl = torchtext.data.Iterator(train_ds, batch_size=batch_size, train=True)
val_dl = torchtext.data.Iterator(val_ds, batch_size=batch_size, train=False, sort=False)
test_dl = torchtext.data.Iterator(test_ds, batch_size=batch_size, train=False, sort=False)

#テスト
batch = next(iter(val_dl))
print(len(batch.Text[0][0]))
print(batch.Label)

import math
from torch.autograd import Variable
import torch.nn.functional as F

#モデルの定義
class Embedder(nn.Module):
  def __init__(self, text_embedding_vecotrs):
    super(Embedder, self).__init__()
    self.embeddings = nn.Embedding.from_pretrained(
        embeddings=text_embedding_vecotrs, freeze=True)

  def forward(self, x):
    x = self.embeddings(x)
    return x

class PositionalEncoder(nn.Module):
  def __init__(self, d_model, max_seq_len=200, dropout_rate=0.1):
    super().__init__()
    self.d_model = d_model
    self.dropout = nn.Dropout(dropout_rate)
    # create constant 'pe' matrix with values dependant on
    # pos and i
    pe = torch.zeros(max_seq_len, d_model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pe = pe.to(device)

    for pos in range(max_seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = \
            math.sin(pos / (10000 ** ((2 * i)/d_model)))
            pe[pos, i + 1] = \
            math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)


  def forward(self, x):
    # make embeddings relatively larger
    x = x * math.sqrt(self.d_model)
    #add constant to embedding
    seq_len = x.size(1)
    pe = Variable(self.pe[:,:seq_len], requires_grad=False)

    x = x + pe
    return self.dropout(x)

def attention(q, k, v, d_k, mask=None, dropout=None):

  #queryとkeyの関連度をだす
  scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k) #[batch, heads, length, length]
  #maskをかける
  if mask is not None:
    mask = mask.unsqueeze(1)
    scores = scores.masked_fill(mask==0, -1e9)

  #AttentionWeight
  scores = F.softmax(scores, dim=-1)
  attention_weight = scores

  if dropout is not None:
    scores = dropout(scores)
  #valueを取り出す
  output = torch.matmul(scores, v) #[batch, heads, length, d_model]
  return output, attention_weight

class MultiHeadAttention(nn.Module):
  def __init__(self, heads, d_model, dropout_rate=0.1):
    super().__init__()
    self.d_model = d_model
    self.d_k = d_model // heads
    self.h = heads
    self.q_linear = nn.Linear(d_model, d_model)
    self.k_linear = nn.Linear(d_model, d_model)
    self.v_linear = nn.Linear(d_model, d_model)

    self.dropout = nn.Dropout(dropout_rate)
    self.out = nn.Linear(d_model, d_model)

  def forward(self, q, k, v, mask=None):
    bs = q.size(0)

    q = self.q_linear(q).view(bs, -1, self.h, self.d_k) #[batch_size, length, heads, d_k]
    k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
    v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    #
    scores, attention_weight = attention(q, k, v, self.d_k, mask, self.dropout)
    concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
    output = self.out(concat)
    return output, attention_weight

class FeedForward(nn.Module):
  def __init__(self, d_model, d_ff=2048, dropout_rate=0.1):
    super().__init__()
    self.linear_1 = nn.Linear(d_model, d_ff)
    self.dropout = nn.Dropout(dropout_rate)
    self.linear_2 = nn.Linear(d_ff, d_model)

  def forward(self, x):
    x = self.dropout(F.relu(self.linear_1(x)))
    x = self.linear_2(x)
    return x

class EncoderLayer(nn.Module):
  def __init__(self, d_model, heads, dropout_rate=0.1):
    super().__init__()
    #LayerNormalizetion
    self.norm_1 = nn.LayerNorm(d_model)
    self.norm_2 = nn.LayerNorm(d_model)

    self.attn = MultiHeadAttention(heads, d_model, dropout_rate=dropout_rate)
    self.ff = FeedForward(d_model, dropout_rate=dropout_rate)

    self.dropout_1 = nn.Dropout(dropout_rate)
    self.dropout_2 = nn.Dropout(dropout_rate)

  def forward(self, x, mask):
    x2 = self.norm_1(x)
    output, attention_weight = self.attn(x2, x2, x2, mask)
    x = x+self.dropout_1(output)
    x2 = self.norm_2(x)
    x = x+self.dropout_2(self.ff(x2))

    return x, attention_weight

import copy

def get_clones(module, N):
  return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
  def __init__(self, text_embedding_vecotrs, N, heads, dropout):
    super().__init__()
    self.N = N
    self.embed = Embedder(text_embedding_vecotrs)
    self.pe = PositionalEncoder(d_model, dropout_rate=dropout)
    self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
    self.norm = nn.LayerNorm(d_model)
  def forward(self, src, mask):
    x = self.embed(src)
    x = self.pe(x)
    for i in range(self.N):
      x, attention_weight = self.layers[i](x, mask)
    return self.norm(x), attention_weight

class TranformerEncoderClassification(nn.Module):
  def __init__(self, text_embedding_vecotrs, d_model, N, heads, output_dim=5, dropout_rate=0.1):
    super().__init__()
    self.encoder = Encoder(text_embedding_vecotrs, N, heads, dropout_rate)
    self.out = nn.Linear(d_model, output_dim)
    # 重み初期化処理
    nn.init.normal_(self.out.weight, std=0.02)
    nn.init.normal_(self.out.bias, 0)

  def forward(self, src, mask):
    x, attention_weight = self.encoder(src, mask)
    output = self.out(x[:, 0, :])
    return output, attention_weight

#テスト
batch = next(iter(train_dl))
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

# モデル構築
net = TranformerEncoderClassification(TEXT.vocab.vectors, d_model, 3, 5)
net.to(device)

# 入出力
x = batch.Text[0].to(device)
src_mask = (x != TEXT.vocab.stoi['<pad>']).unsqueeze(-2)
print(src_mask.shape)
x1, attention_weight = net(x, src_mask)

print("入力のテンソルサイズ：", x.shape)
print("出力のテンソルサイズ：", x1.shape)
print(attention_weight[0][0][0])

net = TranformerEncoderClassification(TEXT.vocab.vectors, d_model, N, heads, output_dim, dropout_rate)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # Liner層の初期化
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


# TransformerBlockモジュールを初期化実行
net.encoder.apply(weights_init)

print('モデルの初期化')

# 辞書オブジェクトにまとめる
dataloaders_dict = {"train": train_dl, "val": val_dl}

# 損失関数の設定
criterion = nn.CrossEntropyLoss()

# 最適化手法の設定
learning_rate = 2e-4
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# モデルを学習させる関数を作成


def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
  # GPUが使えるかを確認
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("使用デバイス：", device)
  print('-----start-------')
  # ネットワークをGPUへ
  net.to(device)

  # ネットワークがある程度固定であれば、高速化させる
  torch.backends.cudnn.benchmark = True

  # epochのループ
  for epoch in range(num_epochs):
    # epochごとの訓練と検証のループ
    for phase in ['train', 'val']:
      if phase == 'train':
        net.train()  # モデルを訓練モードに
      else:
        net.eval()   # モデルを検証モードに

      epoch_loss = 0.0  # epochの損失和
      epoch_corrects = 0  # epochの正解数

      # データローダーからミニバッチを取り出すループ
      for batch in (dataloaders_dict[phase]):
        # batchはTextとLableの辞書オブジェクト

        # GPUが使えるならGPUにデータを送る
        inputs = batch.Text[0].to(device)  # 文章
        labels = batch.Label.to(device)  # ラベル

        # optimizerを初期化
        optimizer.zero_grad()

        # 順伝搬（forward）計算
        with torch.set_grad_enabled(phase == 'train'):
          input_mask = (inputs != TEXT.vocab.stoi['<pad>']).unsqueeze(-2)

          # Transformerに入力
          outputs, _ = net(inputs, input_mask)
          loss = criterion(outputs, labels)  # 損失を計算

          _, preds = torch.max(outputs, 1)  # ラベルを予測

          # 訓練時はバックプロパゲーション
          if phase == 'train':
            loss.backward()
            optimizer.step()

          # 結果の計算
          epoch_loss += loss.item() * inputs.size(0)  # lossの合計を更新
          # 正解数の合計を更新
          epoch_corrects += torch.sum(preds == labels.data)

      # epochごとのlossと正解率
      epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
      epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
      print('Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, num_epochs,
                                                                           phase, epoch_loss, epoch_acc))

  return net

num_epochs = 10
net_trained = train_model(net, dataloaders_dict,
                          criterion, optimizer, num_epochs=num_epochs)
print({'次元数': d_model, 'Nx': N, 'ヘッド数': heads, 'クラス数': output_dim, 'ドロップアウト': dropout_rate})

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net_trained.eval()
net_trained.to(device)

y_true = np.array([])
y_pred = np.array([])

epoch_corrects = 0

for batch in (test_dl):
  inputs = batch.Text[0].to(device)
  labels = batch.Label.to(device)

  with torch.set_grad_enabled(False):
    input_mask = (inputs != TEXT.vocab.stoi['<pad>']).unsqueeze(-2)
    outputs, _ = net_trained(inputs, input_mask)
    _, preds = torch.max(outputs, 1)

    y_true = np.concatenate([y_true, labels.to("cpu", torch.double).numpy()])
    y_pred = np.concatenate([y_pred, preds.to("cpu", torch.double).numpy()])

    epoch_corrects += torch.sum(preds == labels.data)

# 正解率
epoch_acc = epoch_corrects.double() / len(test_dl.dataset)

print('テストデータ{}個での正解率：{:.4f}'.format(len(test_dl.dataset),epoch_acc))

from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

confmat = confusion_matrix(y_true=y_true, y_pred=y_pred)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('Predicted label')
plt.ylabel('True label')

plt.tight_layout()
#plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


#tokenの長さに制限をかける
def filter(sentence):
    return MAXLENGTH > len(sentence.split()) > MINLENGTH
def filterLength(df):
    return [filter(sentence) for sentence in df[0].to_list()]


df = pd.read_csv('CBET.csv')
#emotion = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise', 'thankfulness', 'disgust', 'guilt']
emotion = ['anger', 'fear', 'joy', 'sadness', 'disgust'] #ekman
ids = []
review = []
sentiment = []
for i, e in enumerate(emotion):
    ids.extend(df[ df[e]==1 ]['id'].to_list())
    review.extend(df[ df[e]==1 ]['text'].to_list())
    sentiment.extend([i]*len(df[ df[e]==1 ]))

df = pd.DataFrame({0:review, 1:sentiment, 2:ids})
#idから複数感情のデータを削除する
df = df.drop_duplicates(keep=False, subset=2)

"""
#tokenの長さに制限をかける
length = sorted([len(sentence.split()) for sentence in df[0].to_list()])
MAXLENGTH = length[int(len(length)*0.9)]
MINLENGTH = length[int(len(length)*0.1)]
a = filterLength(df)
df = df[a]
"""

print(len(df))
print(np.bincount(df[1]))

X_train, X_test, y_train, y_test = train_test_split(df[0], df[1], test_size=0.1, random_state=12, stratify=df[1])
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=12, stratify=y_train)

print(len(X_train), len(X_val), len(X_test))
pd.DataFrame({0:X_train, 1:y_train}).to_csv('train.csv', index=None, header=None)
pd.DataFrame({0:X_val, 1:y_val}).to_csv('val.csv', index=None, header=None)
pd.DataFrame({0:X_test, 1:y_test}).to_csv('test.csv', index=None, header=None)

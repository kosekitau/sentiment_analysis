import pandas as pd
import numpy as np


def make(path):

    with open(path + '/dialogues_' + path + '.txt') as f:
        data = f.readlines()

        review = []
        for i in range(len(data)):
            review.extend(data[i].split('__eou__')[:-1])

    #{ 0: no emotion, 1: anger, 2: disgust, 3: fear, 4: happiness, 5: sadness, 6: surprise}
    with open(path + '/dialogues_emotion_' + path + '.txt') as f:
        data = f.readlines()
        sentiment = []
        for i in range(len(data)):
            sentiment.extend(data[i].split())

    df = pd.DataFrame({0:review, 1:sentiment})
    df[1] = df[1].apply(int)
    #5感情のみ
    df = df[df[1] != 6]
    df.to_csv('sentiment_analysis/sentiment_' + path + '.csv', header=None, index=None)
    print(path, np.bincount(df[1]))

paths = ['train', 'validation', 'test']
for path in paths:
    make(path)

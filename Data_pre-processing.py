import pandas as pd
import numpy as np
from tqdm import tqdm
from kss import split_sentences

# KBO

df_KBO = pd.read_csv('/opt/ml/code/KBOBERT/KBO_NEWS.csv')

for num in tqdm(range(len(df_KBO))):
    if len(df_KBO.iloc[num, 1]) <= 30:
        df_KBO.iloc[num, 0] = np.nan
        df_KBO.iloc[num, 1] = np.nan

df_KBO = df_KBO.dropna()

KBO_news = []

for num in tqdm(range(len(df_KBO))):
    KBO_news.append('\n'.join([df_KBO.iloc[num, 0]] + split_sentences(df_KBO.iloc[num, 1].strip())))

KBO_news = '\n\n'.join(KBO_news)

# NAVER

df_NAVER = pd.read_csv('/opt/ml/code/KBOBERT/NAVER_NEWS.csv')

for num in tqdm(range(len(df_NAVER))):
    if len(df_NAVER.iloc[num, 1]) <= 30:
        df_NAVER.iloc[num, 0] = np.nan
        df_NAVER.iloc[num, 1] = np.nan

df_NAVER = df_NAVER.dropna()

NAVER_news = []

for num in tqdm(range(len(df_NAVER))):
    NAVER_news.append('\n'.join([df_NAVER.iloc[num, 0]] + split_sentences(df_NAVER.iloc[num, 1].strip())))

NAVER_news = '\n\n'.join(NAVER_news)

# total

total = KBO_news + '\n\n' + NAVER_news

file = open('/opt/ml/code/KBOBERT/KBOBERT_Data.txt', 'w')
file.write(total)
file.close()
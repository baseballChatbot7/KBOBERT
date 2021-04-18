import pandas as pd
import numpy as np
from tqdm import tqdm
from kss import split_sentences

df = pd.read_csv('/opt/ml/code/KBOBERT/NEWS.csv')

for num in tqdm(range(len(df))):
    if len(df.iloc[num,1]) <= 30:
        df.iloc[num,0] = np.nan
        df.iloc[num,1] = np.nan

df = df.dropna()

news = []

for num in tqdm(range(len(df))):
    news.append('\n'.join([df.iloc[num,0]]+split_sentences(df.iloc[num,1].strip())))

news = '\n\n'.join(news)

file = open('/opt/ml/code/KBOBERT/NEWS.txt', 'w')
file.write(news)
file.close()
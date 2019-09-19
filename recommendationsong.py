import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('song.csv')
df = df[['THEME', 'TITLE','ARTIST']]

def kombinasi(i):
    return str(i['THEME']) + '$' + str(i['ARTIST'])

df['x'] = df.apply(kombinasi,axis=1)
model = CountVectorizer(
    tokenizer = lambda i: i.split('$')
)
kategori = model.fit_transform(df['x'])

cosScore = cosine_similarity(kategori)

fav = 'The Look of Love'
indexSuka = (df[df['TITLE'] == fav].index.values[0])

song = list(enumerate(cosScore[indexSuka]))
sortSong = sorted(song, key=lambda i: i[1],reverse=True)


for i in sortSong[:10]:
    print(df.iloc[i[0]]['TITLE'], df.iloc[i[0]]['ARTIST'], round(i[1]*100),'%' )
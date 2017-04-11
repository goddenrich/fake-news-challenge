'''
Run the file this way:
python data_crossing.py train_bodies.csv train_stances.csv

'''

import pandas as pd
import collections as coll
import sys

bodies_file = sys.argv[1]
stances_file = sys.argv[2]


def data_crossing(df_B,df_S):
    df_H = coll.Counter(df_S['Headline'])
    df_H = pd.DataFrame.from_dict(df_H, orient='index').reset_index()
    df_H['Stance ID'] = df_H.index
    df_H = df_H.rename(columns={'index': 'Headline'})
    df_A = coll.Counter(df_B['Body ID'])
    df_A = pd.DataFrame.from_dict(df_A, orient='index').reset_index()
    df_A['New Body ID'] = df_A.index
    df_A = df_A.rename(columns={'index': 'Body ID'})
    df_S = pd.merge(df_H, df_S, on='Headline')
    df_S = df_S.drop(0, axis=1)
    df_B = pd.merge(df_B, df_A, on='Body ID')
    df_B = df_B.drop(0, axis=1)
    df_ALL = pd.merge(df_S, df_B, on='Body ID')
    return df_ALL


df_B = pd.read_csv(bodies_file)
df_S = pd.read_csv(stances_file)
df_ALL = data_crossing(df_B,df_S)
df_ALL.to_csv('training-all.csv')


'''
Some Notes:
- There are duplicate article bodies in the training set that have different
BODY IDs, there are a bit under 20 of them. When counting by text there are about
1669 items, when counting by BODY ID there are 1683
- BODY IDs are not listed in ordely fashion, I've created 'New BODY ID' which
does in fact run from 0 to 1982, this will be easier if we want to randomly sample
from an index range
- There was no label for STANCE ID, though the data set does have 20-150 copies
of each stance in train_stances that are associated to bodies
- Lastly, created a cross-joined dataframe which contains: 'Headline	Stance ID
Body ID	Stance	articleBody	New Body ID'
- I maintained old BODY ID in case we need to recover the old IDs
- We can approach this data from two ways: take a subset of the stance ID and
recover associated bodies and treat the whole thing as training or viceversa, ie.
take a subset of bodies and recover associated headlines. We could also work with
a subset of associated headlines ie. 2-3 per body.
- This is tricky because the panel of data isn't Bodies X Headlines - we can treat
each pairing as independent from one another, but I dont know what implications
this assumption may have
- Another approach is to take ALL bodies and train on 70 percent of their associated
headlines and test on the rest of associated headlines. This will not be consistent
across samples because some bodies are associated with 10 headlines and some with 50
'''

import pandas as pd
import collections as coll



df_B = pd.read_csv('train_bodies.csv')
df_S = pd.read_csv('train_stances.csv')



df_H = coll.Counter(df_S['Headline'])
df_H = pd.DataFrame.from_dict(df_H, orient='index').reset_index()
df_H['Stance ID'] = df_H.index
df_H = df_H.rename(columns={'index': 'Headline'})


#
df_A = coll.Counter(df_B['Body ID'])
df_A = pd.DataFrame.from_dict(df_A, orient='index').reset_index()
df_A['New Body ID'] = df_A.index
df_A = df_A.rename(columns={'index': 'Body ID'})



df_S = pd.merge(df_H, df_S, on='Headline')
df_S = df_S.drop(0, axis=1)
df_B = pd.merge(df_B, df_A, on='Body ID')
df_B = df_B.drop(0, axis=1)
df_ALL = pd.merge(df_S, df_B, on='Body ID')

df_ALL.to_csv('training-all.csv')

print 'Total stances: ', df_ALL['Stance ID'].max()
print 'Total bodies: ', df_ALL['New Body ID'].max()


df_ALL.sort_values(by='Body ID')

'''
Run the file this way:
python data_splitting.py training-all.csv 0.5 0.03

'''
from __future__ import division
import pandas as pd
import collections as coll
import sys

all_file = sys.argv[1]
per = float(sys.argv[2])
diff = float(sys.argv[3])

def data_splitting(per,df_ALL):

    headlines = pd.DataFrame.from_dict(coll.Counter(df_ALL['Stance ID']), orient='index').reset_index()
    bodies = pd.DataFrame.from_dict(coll.Counter(df_ALL['New Body ID']), orient='index').reset_index()

    # if you want to sample bodies
    train_H = headlines['index'].sample(int(len(headlines)*per))
    train_B = bodies['index'].sample(int(len(bodies)*per))

    #training set
    train = df_ALL[df_ALL['New Body ID'].isin(train_B)]
    train = train[train['Stance ID'].isin(train_H)]

    #testing set
    test = df_ALL[~df_ALL['New Body ID'].isin(train_B)]
    #remove the headlines trained on
    test = test[~test['Stance ID'].isin(train_H)]

    return train, test

def getper(data):
    percentages ={}
    for key in coll.Counter(data['Stance']):
        percentages[key]=coll.Counter(data['Stance'])[key]/sum(coll.Counter(data['Stance']).values())
    return percentages

def check_splitting(diff,df_ALL,train,test):
    #at the moment only verifies that all stances are present
    for i in getper(df_ALL).keys():
        if abs(getper(train)[i]-getper(df_ALL)[i])>diff:
            print 'training split for', i, 'too low'
            return False
        if abs(getper(test)[i]-getper(df_ALL)[i])>diff:
            print 'testing split for', i, 'too low'
            return False
    return True




df_ALL = pd.read_csv(all_file)
#train, test = data_splitting(per,df_ALL)
even_split = False
tries= 0
#diff = 0.03

while even_split!=True and tries<10:
    train, test = data_splitting(per,df_ALL)
    even_split = check_splitting(diff,df_ALL,train,test)
    tries+=1
if even_split == False:
    print 'Warning: your split percentage does not have all stances in test and training, try different per'
    print 'Testing:', getper(test),'\n Training:', getper(train)
if even_split == True:
    print 'Whole set:', getper(df_ALL),'\n', 'Testing:', len(train), getper(train),'\n', 'Training:', len(test),getper(test)

train.to_csv('train_split.csv')
test.to_csv('test_split.csv')
#print 'Training set:',len(train), 'Testing set:',len(test)

'''
Run the file this way:
python data_crossing.py train_bodies.csv train_stances.csv

'''
from __future__ import division
import pandas as pd
import collections as coll
import sys
import argparse


def even_split(df_A, per=0.5, max_tries = 10, diff=0.03, train_file= None, test_file = None):
    even=False
    tries=0

    while not even and tries < max_tries:
        df_train, df_test = data_splitting(per, df_A)
        even = check_splitting(diff, df_A, df_train, df_test)
        tries += 1

    if even:
        print 'Whole set:', getper(df_ALL)
        print 'Testing:', len(df_train), getper(df_train)
        print 'Training:', len(df_test),getper(df_test)
    else:
        print 'Warning: your split percentage does not have all stances in test and training'
        print 'try different per'
        print 'Testing:', getper(df_test)
        print 'Training:', getper(df_train)
    
    if train_file is not None:
        df_train.to_csv(train_file)

    if test_file is not None:
        df_test.to_csv(test_file)

    return df_train, df_test


def import_data(B,S,all_save):
    df_B = pd.read_csv(B)
    df_S = pd.read_csv(S)
    df_ALL = data_crossing(df_B,df_S,all_save)
    return df_ALL
    
def data_crossing(df_B,df_S,df_all_save='training-all.csv'):
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
    
    if df_all_save is not None:
        df_ALL.to_csv(df_all_save)

    return df_ALL


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

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bodies', required=True, type=str)
    parser.add_argument('--stances', required=True, type=str)
    parser.add_argument('--split_per', default=0.2, type=float)
    parser.add_argument('--diff', default=0.03, type=float)
    parser.add_argument('--save_all', type=str)
    parser.add_argument('--save_train', type=str)
    parser.add_argument('--save_test', type=str)
    parser.add_argument('--max_tries', default=10, type=int)
    
    args = parser.parse_args(sys.argv[1:])

    df_ALL = import_data(args.bodies, args.stances, args.save_all)
    even_split(df_ALL, args.split_per, args.max_tries, args.diff, args.save_train, args.save_test)


from __future__ import division
import pandas as pd
import collections as coll
import sys
import argparse
import augment_synonym
import utils
import os


def even_split(df_A, per=0.8, valper=0.2, max_tries = 10, diff=0.03, train_file= None, test_file = None, val_file = None, sfo=False):
    even=False
    tries=0

    while not even and tries < max_tries:
        df_train, df_test = data_splitting(per, df_A)
        even = check_splitting(diff, df_A, df_train, df_test)
        tries += 1

    if even:
        print 'Whole set:', getper(df_A)
        print 'Testing:', len(df_test), getper(df_test)
        print 'Training:', len(df_train),getper(df_train)
    else:
        print 'Warning: your split percentage does not have all stances in test and training'
        print 'try different per'
        print 'Testing:', getper(df_test)
        print 'Training:', getper(df_train)

    '''
    even=False
    tries=0
    #split again to obtain cross validation file
    while not even and tries < max_tries:
        df_train, df_val = data_splitting(1-valper, df_train)
        even = check_splitting(diff, df_A, df_train, df_val)
        tries += 1

    if even:
        print 'Whole set:', getper(df_A)
        print 'validation:', len(df_val),getper(df_val)
        print 'Training:', len(df_train), getper(df_train)
    else:
        print 'Warning: your split percentage does not have all stances in validation and training'
        print 'try different per'
        print 'validation:', getper(df_val)
        print 'Training:', getper(df_train)


    if train_file is not None:
        save_dataframe(df_train, train_file, sfo)

    if test_file is not None:
        save_dataframe(df_test, test_file, sfo)

    if val_file is not None:
        save_dataframe(df_val, val_file, sfo)

    '''

    return df_train, df_test#, df_val

def save_dataframe(df, save_file, sfo=False):
    if not sfo:
        df.to_csv(save_file)
    else:
        df.to_csv(save_file, columns=['Headline', 'Body ID', 'Stance'], index=False)


def oneoff_cleanup(df_B, df_S,cleanup=False,excl_stances=[],excl_bodies=[624,1686,2515]):
    '''
    Call: has to run after import before crossing
    inputs: cleanup flag, runs if True, bodies and stances, dict of IDs to remove before crossing
    Stances not being excluded at this moment
    ouput: pandas df to send to crossing
    '''
    if cleanup==False:
        return df_B, df_S
    else:
        if excl_bodies:
            mask = df_B['Body ID'].isin(excl_bodies)
            df_B = df_B[~mask]
            mask = df_S['Body ID'].isin(excl_bodies)
            df_S = df_S[~mask]
    df_B['articleBody']=df_B['articleBody'].map(lambda x: x.replace('|',' ').replace(',','').replace('\n',' '))
    df_S['Headline']=df_S['Headline'].map(lambda x: x.replace('|',' ').replace(',','').replace('\n',' '))

    return df_B, df_S

def window_split(s, w2v,size=150,window=75):
    '''
    Has to be run before data crossing - will explode rows later on after crossed
    input: sentence
    call: run tokenization and split into chunks, join again
    output: sliding window of chunks which are stings, default size=150, default window=75
    MISSING: integrate the tokenization from w2v here
    '''
    s = augment_synonym.pairwise_tokenize(s,w2v,remove_stopwords=True)
    #s = str.split(s) #call the tokenize function here
    if len(s)<size:
        return [" ".join(s)]
    chunks = [[" ".join(s   [i:i+size])] for i in xrange(0,len(s)-window,window)]
    return chunks

def import_data(B, S, w2v, all_save = None, window=True):
    df_B = pd.read_csv(B)
    df_S = pd.read_csv(S)
    df_B, df_S = oneoff_cleanup(df_B, df_S, True)
    if window:
        df_B['articleBody'] = df_B['articleBody'].apply(window_split, args=(w2v,150,75))
    df_ALL = data_crossing(df_B, df_S, all_save)
    return df_ALL

def explode_data(df):
    df = df.sort('Stance ID')
    all_cols = list(df.columns.values)
    all_cols.remove('articleBody')
    level= 'level_' + str(len(all_cols))
    print len(df)
    df_expl =  df.groupby(all_cols).articleBody.apply(lambda x: pd.DataFrame(x.values[0])).reset_index().drop(level, axis = 1)
    df_expl['articleBody'] = df_expl[0]
    df_expl = df_expl.sort('Stance ID')
    print len(df_expl)
    return df_expl

def data_crossing(df_B, df_S, df_all_save= None):
    df_H = coll.Counter(df_S['Headline'])
    df_H = pd.DataFrame.from_dict(df_H, orient='index').reset_index()
    df_H['Stance ID'] = df_H.index
    df_H = df_H.rename(columns={'index': 'Headline'})
    #df_A = coll.Counter(df_B['Body ID'])
    #df_A = pd.DataFrame.from_dict(df_A, orient='index').reset_index()
    #df_A['New Body ID'] = df_A.index
    #df_A = df_A.rename(columns={'index': 'Body ID'})
    df_S = pd.merge(df_H, df_S, on='Headline')
    df_S = df_S.drop(0, axis=1)
    #df_B = pd.merge(df_B, df_A, on='Body ID')
    #df_B = df_B.drop(0, axis=1)
    df_B['New Body ID'] = df_B.index
    df_ALL = pd.merge(df_S, df_B, on='Body ID')
    print df_ALL
    #df_ALL.apply(lambda x:'%s_%s' % (x['Stance ID'],x['Body Id']),axis=1)
    df_ALL['Stance ID'] = df_ALL['Stance ID'].map(str) + '_'+ df_ALL['Body ID'].map(str)
    print df_ALL


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
    '''
    now checks for relative per difference rather than absolute difference
    '''
    for i in getper(df_ALL).keys():
        if abs((getper(train)[i]-getper(df_ALL)[i])/getper(df_ALL)[i])>diff:
            print 'training split for', i, 'too low'
            return False
        if abs((getper(test)[i]-getper(df_ALL)[i])/getper(df_ALL)[i])>diff:
            print 'testing/validation split for', i, 'too low'
            return False
    return True

def preprocess_all(bodies, stances, save_dir='./', augment_syn=True, window=True, split_per=0.6, val_per=0.2, diff=0.03, max_tries=10):
    w2v_file = 'GoogleNews-vectors-negative300.bin.gz'
    w2v_url = 'https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM'
    w2v = utils.load_w2v(w2v_file,w2v_url)
    df_all = import_data(bodies, stances, w2v, window=window)
    #df_tr, df_te, df_val = even_split(df_all, split_per, val_per, max_tries, diff)
    df_tr, df_te = even_split(df_all, split_per, val_per, max_tries, diff)
    if window:
        print 'test'
        df_te = explode_data(df_te)
        #print 'val'
        #df_val = explode_data(df_val)
        print 'train'
        df_tr = explode_data(df_tr)
    print 'len train', len(df_tr)
    print 'len test', len(df_te)
    #print 'len val', len(df_val)
    augment_synonym.augment_headlines(df_tr, w2v, os.path.join(save_dir,'train.csv'), nrows=None, augment=augment_syn)
    print 'done train'
    #augment_synonym.augment_headlines(df_val, w2v, os.path.join(save_dir,'val.csv'), nrows=None, augment=False)
    augment_synonym.augment_headlines(df_te, w2v, os.path.join(save_dir,'test.csv'), nrows=None, augment=False)


if __name__=='__main__':
    '''
    python data_splitting.py --bodies '../dataset/train_bodies.csv' --stances '../dataset/train_stances.csv' --save_all 'all.csv' --save_train 'train.csv' --save_test 'test.csv' --save_val 'validation.csv'  --save_format_original
    '''
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--bodies', required=True, type=str, help='specify the location of the training bodies file')
    parser.add_argument('--stances', required=True, type=str, help='specify the location of the training stances file')
    parser.add_argument('--train_per', default=0.8, type=float, help='specify the percentage for the train set')
    parser.add_argument('--val_per', default=0.2, type=float, help='specify the percentage for the validation set')
    parser.add_argument('--diff', default=0.03, type=float, help='max ratio difference between the original stratification... eg 0.03 and 75%% original split -> 78%% or 72%%')
    parser.add_argument('--save_all', type=str, help='specify the file to save the joined bodies and stances table')
    parser.add_argument('--save_train', type=str, help='specify the file to save the training set')
    parser.add_argument('--save_test', type=str, help='specify the file to save the test set')
    parser.add_argument('--save_val', type=str, help='specify the file to save the validation set')
    parser.add_argument('--max_tries', default=10, type=int, help='how many times to try the split to get the tolerated ratio difference')
    parser.add_argument('--save_format_original', action='store_true', help='use flag to save the train and test set in original format')

    w2v_file = 'GoogleNews-vectors-negative300.bin.gz'
    w2v_url = 'https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM'

    w2v = utils.load_w2v(w2v_file,w2v_url)



    if len(sys.argv) == 1:
        parser.print_help()
    args = parser.parse_args(sys.argv[1:])

    df_ALL = import_data(args.bodies, args.stances, w2v, args.save_all)
    even_split(df_ALL, args.split_per, args.val_per, args.max_tries, args.diff, args.save_train, args.save_test, args.save_val, args.save_format_original)
    '''
    preprocess_all('../dataset/train_bodies.csv', '../dataset/train_stances.csv', window=True, augment_syn=False)

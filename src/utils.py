import logging
logger = logging.getLogger(__name__)

import os
import sys
import pandas
import numpy as np
import tarfile
from gensim.models.keyedvectors import KeyedVectors
import collections as coll

def load_w2v(w2v_file,w2v_url):
    '''
    returns a gensim.KeyedVector object with the specified word2vec embeddings
    or prompts the user if the embedding file doesn't exist
    '''

    # download pretrained w2v embeddings
    w2v_data=check_dataset(w2v_file,w2v_url,unzip=True,gd=True)

    
    if w2v_data==False:
        print 'could not load w2v file'
        return w2v_data

    # unzip if necessary
    if (not file_exists(w2v_data[:-3])):
        w2v_data = unzip(w2v_data)
    else:
        w2v_data = w2v_data[:-3]

    # load the vectors
    w2v = KeyedVectors.load_word2vec_format(w2v_data, binary=True)
    return w2v


def download_file(path,url):
    ''' 
    downlad file from the url path to the specified path
    '''
    from six.moves import urllib
    urllib.request.urlretrieve(url,path)

def unzip(path):
    '''
    unzips a file and returns the filenames of the unzipped items
    '''
    print path
    tar=tarfile.open(path)
    file_names = tar.getnames()
    for file_name in file_names:
        tar.extract(file_name,f_name)
    tar.close()
    return file_names
    
def file_exists(path):
    '''
    checks if a file exists
    '''
    if (os.path.isfile(path)):
        return True
    else:
        return False

def directory_exists(path):
    '''
    checks if a directory exists
    '''
    if (os.path.isdir(path)):
        return False
    else:
        return True

def check_dataset(dataset,url,unzip=False,gd=False):
    '''
    checks if dataset has been downloaded and if not downloades it from the specified url
    unzips the file if unzip is true
    returns the full filepath or list of filepaths if unzipped
    '''
    new_path = os.path.join(
            os.getcwd(),
            'data'
            )
    data = os.path.join(
            new_path,
            dataset
            )
    if (not os.path.isdir(new_path)):
        os.makedirs(new_path)
    if (not os.path.isfile(data)):
        if gd==False:
            logger.info('Downloading data from %s' % url)
            download_file(data,url)
            if unzip==True:
                data = unzip(data)
        else:
            print 'please go to %s and download the dataset' % url
            print 'then save it within %s as %s' % (new_path,dataset)
            if unzip==True:
                print 'then unzip the file'
            data=False
    return data

def load_dataset(data_file,url):
    '''
    encodes sem-eval-16 tweets as one hot character matricies and labels as one hot labels
    returns a pandas dataframe
    '''
    # ensure dataset exists
    data_file=check_dataset(data_file,url)

    if data_file==False:
        print 'there was an error loading the file'
        return data_file

    # load dataset into a panda dataframe
    data = pandas.read_csv(data_file, sep='\t')

    # encode tweets
    tweet_encoding=encode_tweets(data)
    
    # encode labels
    onehot_labels=one_hot_labels(data)

    # append to panda dataframe
    data['one_hot']=pandas.Series(tweet_encoding)
    data['onehot_label']=pandas.Series(onehot_labels)

    return data

def encode_tweets(data):
    '''
    encodes all tweets in pandas dataframe as one hot character matrix
    '''
    tweets = data['Tweet']
    col = []
    for tweet in tweets:
        tweet_mat = mat_vers(tweet)
        col.append(tweet_mat)

    return col


def mat_vers(tweet):
    '''
    encodes a single tweet as one hot charater matrix
    '''
    numtweet=[]
    for char in tweet:
        numtweet.append(ord(char))
    n_val = 256
    mat_tweet=np.eye(n_val)[numtweet]
    return mat_tweet

def one_hot_labels(data):
    '''
    encodes labels as one hot (against,favor,none)
    '''
    y = data['Stance'].values
    label={'AGAINST':0,'FAVOR':1,'NONE':2}
    col=[]
    for yi in y:
        y_num =label[yi]
        y_one_hot = np.eye(3)[y_num]
        col.append(y_one_hot)
    return col

def data_crossing(df_B,df_S):
    df_H = coll.Counter(df_S['Headline'])
    df_H = pandas.DataFrame.from_dict(df_H, orient='index').reset_index()
    df_H['Stance ID'] = df_H.index
    df_H = df_H.rename(columns={'index': 'Headline'})
    df_A = coll.Counter(df_B['Body ID'])
    df_A = pandas.DataFrame.from_dict(df_A, orient='index').reset_index()
    df_A['New Body ID'] = df_A.index
    df_A = df_A.rename(columns={'index': 'Body ID'})
    df_S = pandas.merge(df_H, df_S, on='Headline')
    df_S = df_S.drop(0, axis=1)
    df_B = pandas.merge(df_B, df_A, on='Body ID')
    df_B = df_B.drop(0, axis=1)
    df_ALL = pandas.merge(df_S, df_B, on='Body ID')
    return df_ALL

def data_splitting(df_ALL, per = 0.5):


    headlines = pandas.DataFrame.from_dict(coll.Counter(df_ALL['Stance ID']), orient='index').reset_index()
    bodies = pandas.DataFrame.from_dict(coll.Counter(df_ALL['New Body ID']), orient='index').reset_index()

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

def to_df(bodies='../data/train_bodies.csv', stances='../data/train_stances.csv'):
    df_B = pandas.read_csv(bodies)
    df_S = pandas.read_csv(stances)
    return df_B, df_S



import logging
logger = logging.getLogger(__name__)

import os
import sys
import pandas
import numpy as np
import tarfile
from gensim.models.keyedvectors import KeyedVectors

def load_w2v(w2v_file,w2v_url):
    '''
    returns a gensim.KeyedVector object with the specified word2vec embeddings
    or prompts the user if the embedding file doesn't exist
    '''

    # download pretrained w2v embeddings
    w2v_data=check_dataset(w2v_file,w2v_url,unzip=True,gd=True)
    
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
    tar=tarfile.open(data)
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
        
    return data

def load_dataset(data_file,url):
    '''
    encodes sem-eval-16 tweets as one hot character matricies and labels as one hot labels
    returns a pandas dataframe
    '''
    # ensure dataset exists
    data_file=check_dataset(data_file,url)

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



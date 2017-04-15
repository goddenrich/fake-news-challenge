import utils
import augment_synonym
import numpy as np
import lstm
import tensorflow as tf

def sentence_to_mat(sentence,w2vmodel):
    temp = []
    words = augment_synonym.pairwise_tokenize(sentence,w2vmodel,remove_stopwords=True)
    for j in words:
        try:
            temp.append(w2vmodel[j].reshape((300,1)))
        except KeyError:
            continue
    temp = np.concatenate(temp, axis=1)
    return temp

def db(msg, deb):
    if deb:
        print msg

def augment(data, augw2v=False, augSplit=False):
    return data


def train(params, bf='../data/train_bodies.csv', sf='../data/train_stances.csv', model='neural_attention', train_per=0.8, n_folds=10, deb=True):

    model_type='a'
    weights = 'b'

    # import data and test train split
    bodies, stances = utils.to_df(bf, sf)
    data = utils.data_crossing(bodies,stances)
    train, test = utils.data_splitting(data,train_per)
    db('\nn train: %d, n test: %d' % (len(train),len(test)) , deb)

    augw2v=False
    augSplit=False
    train_aug = augment(train,augw2v,augSplit)
    
    scores=[]
    for params in hyperparams:
        score = 0
        #perform cross validation
        for cvn in range(n_folds):
            # train validation split
            db('\nsplit n: %d' % (cvn+1), deb)
            train_cv, val_cv = utils.data_splitting(train_aug,(n_folds-1)/float(n_folds))
            db('n cv train: %d, n val: %d' % (len(train_cv),len(val_cv)) , deb)
            db('params:',deb)
            db(params,deb)
            model = lstm.lstm(params)
            if weights:
                model.load(weights)

            model.train(train_cv)

            score += model.score(train_cv)/float(n_folds)

        scores.append(score)
    
    best_params = hyperparams[np.argmin(scores)]
    db('\nchosen params:',deb)
    db(best_params,deb)
    model = lstm.lstm(best_params)
    model.train(train)
    print 'chosen model score: %d' %model.score(test)
    model.save('/trained/model')

def test_train():
    params={'batch_size':40,
            'learning_rate': .01,
            'l2': 1e-4,
            'data_dim': 300}

    train_filename = "temp.txt" #temp is the output of running augment_synonyms.py


    with tf.Session() as sess:
        model = lstm.lstm(params,sess)
        init = tf.global_variables_initializer()
        sess.run(init)
        model.train(train_filename,None,None,1,sess,False)

    return 

def test_pred():
    params={'batch_size':40,
        'learning_rate': .01,
        'l2': 1e-4,
        'data_dim': 300}
    data_filename = "temp.txt" #temp is the output of running augment_synonyms.py
    model_filename = "nn_snapshots/lstm-10"
    with tf.Session() as sess:
            model = lstm.lstm(params,sess)
            init = tf.global_variables_initializer()
            sess.run(init)
            model.pred(sess,data_filename,model_filename)

def classify(data='', model='', params='', weights=''):
    model = lstm.lstm(params)
    model.load(weights)
    model.conf_mat(data)

if __name__ == "__main__":
    # hyperparams = [{'learning_rate':1,'dropout':0.2,'l2Reg':0.1}]
    # train(params=hyperparams)
    # classify(params=hyperparams[0])

    test_train()
    # test_pred()

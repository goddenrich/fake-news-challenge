import utils
import augment_synonym
import numpy as np

def db(msg,deb):
    if deb:
        print msg

class model_select:
    def __init__(self,model_type, weights, params):
        print 'init'
        self.model_type = model_type
        self.weights = weights
        self.params = params

    def train(self, data):
        print 'train'

    def score(self, data):
        print 'score'
        return 1

    def save(self, path):
        print 'save'

def augment(data, augw2v=False, augSplit=False):
    return data


def train(bf='../data/train_bodies.csv', sf='../data/train_stances.csv', model='neural_attention', train_per=0.8, n_folds=10, deb=True):

    model_type='a'
    weights = 'b'
    hyperparams = [{'a':1}]

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

            model = model_select(model_type, weights, params)
            model.train(train_cv)

            score += model.score(train_cv)/float(n_folds)

        scores.append(score)
    
    best_params = hyperparams[np.argmin(scores)]
    model = model_select(model_type,weights,best_params)
    model.train(train)
    print 'chosen model score: %d' %model.score(test)
    model.save('/trained/model')



if __name__ == "__main__":
    train()

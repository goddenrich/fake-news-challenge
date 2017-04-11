import utils
import augment_synonym

def db(msg,deb):
    if deb:
        print msg

def train(bf='../data/train_bodies.csv', sf='../data/train_stances.csv', model='neural_attention', train_per=0.8, n_folds=10, deb=True):
    # import data and test train split
    bodies, stances = utils.to_df(bf, sf)
    data = utils.data_crossing(bodies,stances)
    train, test = utils.data_splitting(data,train_per)
    db('\nn train: %d, n test: %d' % (len(train),len(test)) , deb)

    #perform cross validation
    for cvn in range(n_folds):
        # train validation split
        db('\nsplit n: %d' % (cvn+1), deb)
        train_cv, val_cv = utils.data_splitting(train,(n_folds-1)/float(n_folds))
        db('n cv train: %d, n val: %d' % (len(train_cv),len(val_cv)) , deb)




if __name__ == "__main__":
    train()

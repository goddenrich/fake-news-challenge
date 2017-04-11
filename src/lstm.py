import tensorflow as tf

class lstm(object):
    def __init__(self, params, data_shapes):
        print 'init'
        self.params = params
        learningRate = params['learning_rate']
        dropout = params['dropout']
        l2Reg = params['l2Reg']
        self.k= params['n_lstm']
        self.batchsize = params['batchsize']
        optimizer = 'Adam'
        momentum1 = 0.9
        momentum2 = 0.999
        self.data_shapes = data_shapes

    def inference(self):
        hb, y = placeholder_inputs()



        return hb, y,

    def placeholder_inputs(self):
        hb = tf.placeholder(tf.float32, shape=(batchsize,
                                       hb_len,
                                       w2v_dim))
        y = tf.placeholder(tf.float32, shape=(batchsize,
                                        num_class))
        return hb, y

    def train(self, data):
        print 'train'

    def pred(self,data):
        print 'pred'
    
    def score(self, data):
        print 'score'
        return 1

    def save(self, path):
        print 'save'

    def load(self, weights):
        print 'weights'

    def conf_mat(self, data):
        print 'conf mat'


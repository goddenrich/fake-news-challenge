import tensorflow as tf
import nn_input_data
import utils
import numpy as np
import time
import os

class lstm(object):
    def __init__(self, params,sess):
        print 'init LSTM'

        self.params = params
        
        #Model Params
        self.batch_size = params['batch_size']
        self.learning_rate = params['learning_rate']
        self.l2 = params['l2']
        self.lstm_units = 100
        self.hidden_units = 512

        #DataPrams
        self.body_truncate_len = 150
        self.headline_truncate_len = 40
        self.input_truncate_len = self.body_truncate_len + self.headline_truncate_len
        self.data_dim = params['data_dim']

        #W2V
        w2v_file = 'GoogleNews-vectors-negative300.bin.gz'
        w2v_url = 'https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM'        
        self.w2v = utils.load_w2v(w2v_file,w2v_url)

        #Build the model
        self.output = self.add_model()
        self.loss = self.calculate_loss(self.output)
        self.train_step = self.add_training()
        self.eval_correct = self.evaluate(self.output)
        self.iterations = 0 #counts the iterations we've run

        #Summary writers
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('visual_logs_lstm' + '/train',
                                              sess.graph)
        self.test_writer = tf.summary.FileWriter('visual_logs_lstm'+ '/test')

        #Saver
        self.saver = tf.train.Saver()

    def add_model(self):
        print "building model"
        with tf.device('/gpu:0'):
            self.input_placeholder = tf.placeholder(tf.float32, shape=(None, self.input_truncate_len, self.data_dim))
            self.label_placeholder = tf.placeholder(tf.int64, shape=(self.batch_size,))
            self.length_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size,))

            inputs = tf.transpose(self.input_placeholder,[1,0,2])
            inputs = tf.reshape(self.input_placeholder,[-1,self.data_dim])
            # inputs = tf.split(axis=0,num_or_size_splits=self.input_truncate_len,value=inputs)
            inputs = tf.split(inputs, self.input_truncate_len, 0)

            # inputs = tf.unstack(self.input_placeholder,self.input_truncate_len,1)

            #layer 1
            with tf.variable_scope('BIRNN',initializer = tf.contrib.layers.xavier_initializer()):
                fwd_cell_1 = tf.contrib.rnn.LSTMCell(self.lstm_units,forget_bias = 1.0)
                bkwd_cell_1 = tf.contrib.rnn.LSTMCell(self.lstm_units,forget_bias = 1.0)

                bidi_output,_,_ = tf.contrib.rnn.static_bidirectional_rnn(fwd_cell_1,bkwd_cell_1, inputs, 
                       initial_state_fw=None, initial_state_bw=None, dtype=tf.float32,sequence_length=self.length_placeholder)

                outputs = tf.stack(bidi_output)
                outputs = tf.transpose(outputs, [1, 0, 2])

                # Hack to build the indexing and retrieve the right output.
                batch_size = tf.shape(outputs)[0]
                # Start indices for each sample
                index = tf.range(0, batch_size) * self.input_truncate_len + (self.length_placeholder - 1)
                # Indexing
                outputs = tf.gather(tf.reshape(outputs, [-1, self.lstm_units*2]), index)

            W3 = tf.get_variable("W3", shape=[self.lstm_units*2,4],initializer=tf.truncated_normal_initializer(stddev=0.01))
            b3 = tf.Variable(tf.zeros([4]))
            y3 = tf.matmul(outputs,W3) + b3
        return y3

    def calculate_loss(self,logits):
        regularization = 0
        for i in tf.trainable_variables():
            regularization += tf.nn.l2_loss(i)
        data_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.label_placeholder))
        loss = data_loss

        tf.summary.scalar('data_loss', data_loss)

        return loss
        

    def add_training(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads = optimizer.compute_gradients(self.loss)

        clipped_grads = [(tf.clip_by_value(grad,-1,1),var) for grad,var in grads]
        train_step = optimizer.apply_gradients(clipped_grads)
        return train_step

    def evaluate(self,output):
        # eval_correct = tf.nn.in_top_k(output, self.label_placeholder, 1)

        correct_pred = tf.equal(tf.argmax(output, 1), self.label_placeholder)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        return tf.reduce_sum(tf.cast(correct_pred,tf.int32))

    def train(self,train_filename,valid_filename,test_filename,num_epochs,sess,run_test=False):
        for epoch in xrange(num_epochs):
            train_loss,acc = self.run_epoch(train_filename,True,sess,verbose=True)
            if run_test:
                valid_loss,valid_acc = self.pred(sess,valid_filename)
                print "=========== epoch = " + str(epoch) + "=============="
                print "train_loss: " + str(train_loss) + "\ttrain_acc: " + str(acc) + \
                    " valid_loss: " + str(valid_loss) + "\tvalid_acc: " + str(valid_acc)
            else:
                print "=========== epoch = " + str(epoch) + "=============="
                print "train_loss: " + str(train_loss) + "\ttrain_acc: " + str(acc)


    def run_epoch(self,data_filename,isTraining,session,verbose =False):
        data_len = len(list(open(data_filename,'rb')))
        num_batches = data_len/self.batch_size
        losses = []
        total_count = 0
        acc = 0.0
        indices = []
        for index in xrange(int(num_batches)):
            start_time = time.time()
            loss,true_count,indices,summaries,output = self.run_batch(data_filename,indices,isTraining,session)
            dur = time.time() - start_time

            self.iterations += 1
            if isTraining:
                self.train_writer.add_summary(summaries,self.iterations)
            else:
                self.test_writer.add_summary(summaries,self.iterations)
            if self.iterations % 100 == 0:
                print output
                self.saver.save(session, os.path.join('nn_snapshots/', 'lstm-v5'), global_step=self.iterations)

            print true_count
            total_count += np.sum(true_count)
            acc = total_count/float((index+1) * self.batch_size)
            #print acc
            if verbose:
                print "batch " + str(index+1) + "/" + str(num_batches) + "\tbatch loss: " + str(loss) + "\t batch acc: " + str(float(np.sum(true_count))/self.batch_size) + "\t dur:" + str(dur)
            losses.append(loss)

        return np.mean(losses),acc

    def run_batch(self,data_filename,indices,isTraining,session):
        batch,indices = nn_input_data.get_batch(data_filename,indices,self.batch_size,self.headline_truncate_len,self.body_truncate_len,self.data_dim,self.w2v,isTraining)
        input_vectors = nn_input_data.simple_combine(batch['batch_headlines'],batch['batch_bodies'])

        feed = {self.input_placeholder: input_vectors,self.label_placeholder: batch['batch_labels'], self.length_placeholder: batch['batch_input_len']}

        if isTraining:
            loss, true_count, _, output,summaries = session.run([self.loss, self.eval_correct, self.train_step, self.output,self.merged], feed_dict=feed)
        else:
            loss, true_count, output,summaries = session.run([self.loss, self.eval_correct, self.output,self.merged], feed_dict=feed)

        return loss,true_count,indices,summaries,output

    def pred(self,session,data_filename,model_filename):
        """Gets the loss, acc of a loaded model, where model_filename is the relative path
        e.g. nn_snapshots/lstm-1000"""
        self.load(session,model_filename)
        loss,acc = self.run_epoch(data_filename,False,session,verbose =True)

        return loss,acc
    
    def score(self, data):
        print 'score'
        return 1

    def load(self, session,filename):
        """loads a previously saved LSTM, uses relative path.
        e.g. nn_snapshots/lstm-1000"""

        self.saver.restore(session, filename)

    def conf_mat(self, data):
        print 'conf mat'


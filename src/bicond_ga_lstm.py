import tensorflow as tf
import nn_input_data
import utils
import numpy as np
import time
import os

class bicond_ga_lstm(object):
    def __init__(self, params,sess):
        print 'init BiCond_Global_Attention_LSTM'

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
        self.train_writer = tf.summary.FileWriter('visual_logs' + '/train',
                                              sess.graph)
        self.test_writer = tf.summary.FileWriter('visual_logs'+ '/test')

        #Saver
        self.saver = tf.train.Saver()

    def add_model(self):
        print "building model"
        with tf.device('/gpu:0'):
            self.headline_placeholder = tf.placeholder(tf.float32, shape=(None, self.headline_truncate_len, self.data_dim))
            self.head_len_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size,))

            self.body_placeholder = tf.placeholder(tf.float32, shape=(None, self.body_truncate_len, self.data_dim))
            self.body_len_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size,))

            self.label_placeholder = tf.placeholder(tf.int64, shape=(self.batch_size,))

            h_inputs = tf.unstack(self.headline_placeholder,self.headline_truncate_len,1)
            b_inputs = tf.unstack(self.body_placeholder,self.body_truncate_len,1)
            #layer 1
            with tf.variable_scope('Headline_BIRNN',initializer = tf.contrib.layers.xavier_initializer()):
                h_fwd_cell_1 = tf.contrib.rnn.LSTMCell(self.lstm_units,forget_bias = 1.0)
                h_bkwd_cell_1 = tf.contrib.rnn.LSTMCell(self.lstm_units,forget_bias = 1.0)

                h_bidi_outputs,h_final_hidden_state,_ = tf.contrib.rnn.static_bidirectional_rnn(h_fwd_cell_1,h_bkwd_cell_1, h_inputs, 
                       initial_state_fw=None, initial_state_bw=None, dtype=tf.float32)

            with tf.variable_scope('Body_BIRNN',initializer = tf.contrib.layers.xavier_initializer()):

                b_fwd_cell_1 = tf.contrib.rnn.LSTMCell(self.lstm_units,forget_bias = 1.0)
                b_bkwd_cell_1 = tf.contrib.rnn.LSTMCell(self.lstm_units,forget_bias = 1.0)

                b_bidi_outputs,_,_ = tf.contrib.rnn.static_bidirectional_rnn(b_fwd_cell_1,b_bkwd_cell_1, b_inputs, 
                       initial_state_fw=h_final_hidden_state, initial_state_bw=None, dtype=tf.float32)

                #Global Attention
                Wy = tf.get_variable('Wy',[self.lstm_units*2,self.lstm_units*2],initializer=tf.truncated_normal_initializer())
                Wh = tf.get_variable('Wh',[self.lstm_units*2,self.lstm_units*2],initializer=tf.truncated_normal_initializer())
                Wx = tf.get_variable('Wx',[self.lstm_units*2,self.lstm_units*2],initializer=tf.truncated_normal_initializer())
                Wp = tf.get_variable('Wp',[self.lstm_units*2,self.lstm_units*2],initializer=tf.truncated_normal_initializer())
                Wa = tf.get_variable('Wa',[self.lstm_units*2],initializer=tf.truncated_normal_initializer())

                Wfinal = tf.get_variable('Wfinal',[self.lstm_units*2,4],initializer=tf.truncated_normal_initializer())
                bfinal = tf.get_variable('bfinal',[4],initializer=tf.constant_initializer(0))

                block_Wy = tf.multiply(Wy,tf.constant(1.0,shape=[self.batch_size,self.lstm_units*2,self.lstm_units*2]))
                block_Wh = tf.multiply(Wh,tf.constant(1.0,shape=[self.batch_size,self.lstm_units*2,self.lstm_units*2]))
                block_Wp = tf.multiply(Wp,tf.constant(1.0,shape=[self.batch_size,self.lstm_units*2,self.lstm_units*2]))
                block_Wx = tf.multiply(Wx,tf.constant(1.0,shape=[self.batch_size,self.lstm_units*2,self.lstm_units*2]))
                block_Wa = tf.multiply(Wa,tf.constant(1.0,shape=[self.batch_size,1,self.lstm_units*2]))
                #repeat-stack column-wise HN
                e = tf.constant(1.0,shape=[self.headline_truncate_len,self.batch_size,self.lstm_units*2])
                block_HN = tf.transpose(tf.multiply(b_bidi_outputs[-1],e),[1,2,0]) #(batchsize,lstm*2,length)

                block_Y = tf.transpose(h_bidi_outputs,[1,2,0])
                M_p1 = tf.matmul(block_Wy,block_Y)
                M_p2 = tf.matmul(block_Wh,block_HN)
                M = tf.tanh(M_p1+ M_p2) # (batch_size,lstm*2,length)
                alpha = tf.nn.softmax(tf.matmul(block_Wa,M)) # (batch_size,1,lstm*2) x (batch_size,lstm*2,length) = batch_size,1,length)

                r = tf.matmul(block_Y,tf.transpose(alpha,[0,2,1])) #(batch_size,lstm*2,1)
                attention_output = tf.tanh(tf.matmul(block_Wp,r)+ tf.matmul(block_Wx,tf.reshape(b_bidi_outputs[-1],[-1,self.lstm_units*2,1])))
                attention_output = tf.reshape(attention_output,[self.batch_size,self.lstm_units*2])

                logits = tf.matmul(attention_output,Wfinal) + bfinal

        return logits

    def calculate_loss(self,logits):
        regularization = 0
        for i in tf.trainable_variables():
            regularization += tf.nn.l2_loss(i)
        data_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.label_placeholder))
        loss = data_loss + self.l2 * regularization

        tf.summary.scalar('data_loss', data_loss)

        return loss
        

    def add_training(self):
        return tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def evaluate(self,output):
        eval_correct = tf.nn.in_top_k(output, self.label_placeholder, 1)

        correct_pred = tf.equal(tf.argmax(output, 1), self.label_placeholder)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        return eval_correct

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
            loss,true_count,indices,summaries = self.run_batch(data_filename,indices,isTraining,session)
            dur = time.time() - start_time

            self.iterations += 1
            if isTraining:
                self.train_writer.add_summary(summaries,self.iterations)
            else:
                self.test_writer.add_summary(summaries,self.iterations)
            if self.iterations % 100 == 0:
                self.saver.save(session, os.path.join('nn_snapshots/', 'lstm'), global_step=self.iterations)

            total_count += true_count.sum()
            acc = total_count/float((index+1) * self.batch_size)
            #print acc
            if verbose:
                print "batch " + str(index+1) + "/" + str(num_batches) + "\tbatch loss: " + str(loss) + "\t batch acc: " + str(float(true_count.sum())/self.batch_size) + "\t dur:" + str(dur)
            losses.append(loss)

        return np.mean(losses),acc

    def run_batch(self,data_filename,indices,isTraining,session):
        batch,indices = nn_input_data.get_batch(data_filename,indices,self.batch_size,self.headline_truncate_len,self.body_truncate_len,self.data_dim,self.w2v,isTraining)

        feed = {self.headline_placeholder: batch['batch_headlines'],self.body_placeholder:batch['batch_bodies'], self.label_placeholder: batch['batch_labels'], self.head_len_placeholder : batch['batch_headline_len'],self.body_len_placeholder: batch['batch_body_len']}

        if isTraining:
            loss, true_count, _, output,summaries = session.run([self.loss, self.eval_correct, self.train_step, self.output,self.merged], feed_dict=feed)
        else:
            loss, true_count, output,summaries = session.run([self.loss, self.eval_correct, self.output,self.merged], feed_dict=feed)

        return loss,true_count,indices,summaries

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


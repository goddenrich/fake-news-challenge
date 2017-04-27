import tensorflow as tf
import nn_input_data
import utils
import numpy as np
import time
import os
import math

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
            self.head_len_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size,))

            self.body_placeholder = tf.placeholder(tf.float32, shape=(None, self.body_truncate_len, self.data_dim))
            self.body_len_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size,))

            self.label_placeholder = tf.placeholder(tf.int64, shape=(self.batch_size,))

            h_inputs = tf.unstack(self.headline_placeholder,self.headline_truncate_len,1)
            b_inputs = tf.unstack(self.body_placeholder,self.body_truncate_len,1)
            #layer 1
            with tf.variable_scope('Headline_BIRNN',initializer = tf.contrib.layers.xavier_initializer()):
                h_fwd_cell_1 = tf.contrib.rnn.LSTMCell(self.lstm_units,forget_bias = 1.0)
                h_bkwd_cell_1 = tf.contrib.rnn.LSTMCell(self.lstm_units,forget_bias = 1.0)

                h_bidi_outputs,h_final_hidden_state,_ = tf.contrib.rnn.static_bidirectional_rnn(h_fwd_cell_1,h_bkwd_cell_1, h_inputs, 
                       initial_state_fw=None, initial_state_bw=None, dtype=tf.float32,sequence_length=self.head_len_placeholder)

            with tf.variable_scope('Body_BIRNN',initializer = tf.contrib.layers.xavier_initializer()):

                b_fwd_cell_1 = tf.contrib.rnn.LSTMCell(self.lstm_units,forget_bias = 1.0)
                b_bkwd_cell_1 = tf.contrib.rnn.LSTMCell(self.lstm_units,forget_bias = 1.0)

                b_bidi_outputs,_,_ = tf.contrib.rnn.static_bidirectional_rnn(b_fwd_cell_1,b_bkwd_cell_1, b_inputs, 
                       initial_state_fw=h_final_hidden_state, initial_state_bw=None, dtype=tf.float32,sequence_length=self.body_len_placeholder)

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

                #### RNN indexing code adapted from: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dynamic_rnn.py
                body_outputs = tf.stack(b_bidi_outputs)
                body_outputs = tf.transpose(body_outputs, [1, 0, 2])

                # Hack to build the indexing and retrieve the right output.
                # Start indices for each sample
                index = tf.range(0, self.batch_size) * self.body_truncate_len + (self.body_len_placeholder - 1)
                # Indexing
                body_outputs = tf.gather(tf.reshape(body_outputs, [-1, self.lstm_units*2]), index)
                #### End of adaptation

                #repeat-stack column-wise HN
                e = tf.constant(1.0,shape=[self.headline_truncate_len,self.batch_size,self.lstm_units*2])
                block_HN = tf.transpose(tf.multiply(body_outputs,e),[1,2,0]) #(batchsize,lstm*2,length)

                block_Y = tf.transpose(h_bidi_outputs,[1,2,0]) #(batchsize,lstm*2,length)
                M_p1 = tf.matmul(block_Wy,block_Y)
                M_p2 = tf.matmul(block_Wh,block_HN)
                M = tf.tanh(M_p1+ M_p2) # (batch_size,lstm*2,length)
                alpha = tf.nn.softmax(tf.matmul(block_Wa,M)) # (batch_size,1,lstm*2) x (batch_size,lstm*2,length) = batch_size,1,length)

                r = tf.matmul(block_Y,tf.transpose(alpha,[0,2,1])) #(batch_size,lstm*2,1)
                attention_output = tf.tanh(tf.matmul(block_Wp,r) + tf.matmul(block_Wx,tf.reshape(b_bidi_outputs[-1],[-1,self.lstm_units*2,1])))
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
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads = optimizer.compute_gradients(self.loss)

        clipped_grads = [(tf.clip_by_value(grad,-1,1),var) for grad,var in grads]
        train_step = optimizer.apply_gradients(clipped_grads)
        return train_step


    def evaluate(self,output):
        eval_correct = tf.nn.in_top_k(output, self.label_placeholder, 1)

        correct_pred = tf.equal(tf.argmax(output, 1), self.label_placeholder)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        return eval_correct

    def train(self,train_filename,valid_filename,test_filename,num_epochs,session,run_test=False):
        for epoch in xrange(num_epochs):
            start_time = time.time()
            train_loss,acc,all_ids,predicted_labels,true_labels = self.run_epoch(train_filename,True,session,verbose=False)
            dur = time.time() - start_time
            dur = dur/float(60)
            if run_test:
                valid_loss,valid_acc = self.pred(session,valid_filename)
                print "=========== epoch = " + str(epoch) + "=============="
                print "train_loss: " + str(train_loss) + "\ttrain_acc: " + str(acc) + \
                    " valid_loss: " + str(valid_loss) + "\tvalid_acc: " + str(valid_acc)
                print "train time (min): " + str(dur)
            else:
                print "=========== epoch = " + str(epoch) + "=============="
                print "train_loss: " + str(train_loss) + "\ttrain_acc: " + str(acc)
                print "train time (min): " + str(dur)


    def run_epoch(self,data_filename,isTraining,session,verbose =False):
        data_len = len(list(open(data_filename,'rb')))
        if not isTraining:
            num_batches = math.ceil(data_len/self.batch_size)
        else:
            num_batches = data_len/self.batch_size
        losses = []
        total_count = 0
        acc = 0.0
        indices = []

        all_ids = np.empty(data_len,dtype=object)
        true_labels = np.zeros(data_len)
        predicted_labels = np.zeros(data_len)

        for index in xrange(int(num_batches)):
            start_time = time.time()
            loss,true_count,indices,summaries,logits,batch_ids,batch_true_labels = self.run_batch(data_filename,indices,isTraining,session)
            batch_predicted_labels = np.argmax(logits,axis=1)
            predicted_labels[index*self.batch_size:(index+1)*self.batch_size] = batch_predicted_labels
            true_labels[index*self.batch_size:(index+1)*self.batch_size] = batch_true_labels
            all_ids[index*self.batch_size:(index+1)*self.batch_size] = batch_ids
            dur = time.time() - start_time

            self.iterations += 1
            if isTraining:
                self.train_writer.add_summary(summaries,self.iterations)
            else:
                self.test_writer.add_summary(summaries,self.iterations)
            if self.iterations % 100 == 0 and isTraining:
                self.saver.save(session, os.path.join('nn_snapshots/', 'bicondv3'), global_step=self.iterations)
            if self.iterations + 1 % data_len == 0 and isTraining:
                print output
                self.saver.save(session, os.path.join('nn_snapshots/', 'bicondv3-epoch'+str(int(self.iterations/float(data_len)))))

            total_count += true_count.sum()
            acc = total_count/float((index+1) * self.batch_size)
            #print acc
            if verbose:
                print "batch " + str(index+1) + "/" + str(num_batches) + "\tbatch loss: " + str(loss) + "\t batch acc: " + str(float(true_count.sum())/self.batch_size) + "\t dur:" + str(dur)
            losses.append(loss)

        return np.mean(losses),acc,all_ids,predicted_labels,true_labels

    def run_batch(self,data_filename,indices,isTraining,session):
        batch,indices = nn_input_data.get_batch(data_filename,indices,self.batch_size,self.headline_truncate_len,self.body_truncate_len,self.data_dim,self.w2v,isTraining)

        feed = {self.headline_placeholder: batch['batch_headlines'],self.body_placeholder:batch['batch_bodies'], self.label_placeholder: batch['batch_labels'], self.head_len_placeholder : batch['batch_headline_len'],self.body_len_placeholder: batch['batch_body_len']}

        if isTraining:
            loss, true_count, _, output,summaries = session.run([self.loss, self.eval_correct, self.train_step, self.output,self.merged], feed_dict=feed)
        else:
            loss, true_count, output,summaries = session.run([self.loss, self.eval_correct, self.output,self.merged], feed_dict=feed)

        return loss,true_count,indices,summaries,output,batch['batch_ids'],batch['batch_labels']

    def pred(self,session,data_filename,model_filename=None):
        """Gets the loss, acc of a model in the current training session, if model_filename is provided, will use that model using the relative path
        e.g. nn_snapshots/lstm-1000"""
        
        if model_filename:
            self.load(session,model_filename)

        loss,acc,all_ids,predicted_labels,true_labels = self.run_epoch(data_filename,False,session,verbose =False)
        self.score(all_ids,predicted_labels,true_labels)

        return loss,acc
    
    def score(self, ids, all_predicted, all_true_labels):
        """Takes list of unique ids, all the predictions + true labels for every row in the input
            Iterates through list and get the indices of the current id
            Get the majority vote and assign as predicted label
            Uses Fake news scoring algo"""

        ids = ids[ids != np.array(None)]
        unique_ids = np.unique(ids)
        predicted = []
        true_labels = []

        for i in xrange(len(unique_ids)):
            current = unique_ids[i]
            if current != -1:
                current_inds = ids == current
                true_label = all_true_labels[current_inds][0]
                predicted_labels_instance = all_predicted[current_inds]
                vals,counts = np.unique(predicted_labels_instance,return_counts=True)
                predicted_label = vals[np.argmax(counts)]
                predicted_label = nn_input_data.reverse_encode(int(predicted_label))
                true_label = nn_input_data.reverse_encode(int(true_label))
                predicted.append(predicted_label)
                true_labels.append(true_label)

        score,cm = self.score_submission(true_labels,predicted)
        print score
        self.print_confusion_matrix(cm)
        null_score, max_score = self.score_defaults(true_labels)
        print "null: " + str(null_score) + "max: " + str(max_score)

        print "our score: " + str(score/float(max_score))
        return 1

    def load(self, session,filename):
        """loads a previously saved LSTM, uses relative path.
        e.g. nn_snapshots/lstm-1000"""

        self.saver.restore(session, filename)

    ###################################################
    ######  Fake News Challenge scoring code ##########
    ###################################################
    def score_submission(self,gold_labels, test_labels):
        FIELDNAMES = ['Headline', 'Body ID', 'Stance']
        LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
        RELATED = LABELS[0:3]

        score = 0.0
        cm = [[0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]

        for i, (g_stance, t_stance) in enumerate(zip(gold_labels, test_labels)):
            if g_stance == t_stance:
                score += 0.25
                if g_stance != 'unrelated':
                    score += 0.50
            if g_stance in RELATED and t_stance in RELATED:
                score += 0.25

            cm[LABELS.index(g_stance)][LABELS.index(t_stance)] += 1

        return score, cm

    def print_confusion_matrix(self,cm):
        FIELDNAMES = ['Headline', 'Body ID', 'Stance']
        LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
        RELATED = LABELS[0:3]

        lines = ['CONFUSION MATRIX:']
        header = "|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format('', *LABELS)
        line_len = len(header)
        lines.append("-"*line_len)
        lines.append(header)
        lines.append("-"*line_len)

        hit = 0
        total = 0
        for i, row in enumerate(cm):
            hit += row[i]
            total += sum(row)
            lines.append("|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format(LABELS[i],
                                                                       *row))
            lines.append("-"*line_len)
        lines.append("ACCURACY: {:.3f}".format(hit / float(total)))
        print('\n'.join(lines))

    def score_defaults(self,gold_labels):
        """
        Compute the "all false" baseline (all labels as unrelated) and the max
        possible score
        :param gold_labels: list containing the true labels
        :return: (null_score, best_score)
        """
        unrelated = [g for g in gold_labels if g == 'unrelated']
        null_score = 0.25 * len(unrelated)
        max_score = null_score + (len(gold_labels) - len(unrelated))
        return null_score, max_score
    #######################################################
    ######  END Fake News Challenge scoring code ##########
    #######################################################
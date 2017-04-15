import numpy as np
import random
import time

from svd import sentence_to_mat

def get_batch(data_filename,indices,batch_size,headline_truncate_len,body_truncate_len,data_dim,w2v,isTraining):
    """ Takes a data_filename csv and reads in a batch. Indices can be [] or a list of indices.
        If empty, the function will retrieve all the indices necessary, create a batch, 
        then return the remaining indices for further batches
        Batch_IDs is a list of tuples (stance_id,body_id)
        IsTraining is a boolean that decides whether to randomize the indices"""


    lines = open(data_filename,'r')
    lines = list(lines)
    lines = lines[1:] # skip header

    if len(indices) < batch_size:
        new_indices = range(len(lines))
        if isTraining:
            random.seed(time.time())
            random.shuffle(indices)
        indices.extend(new_indices)

    batch_headlines = np.zeros((batch_size,headline_truncate_len,data_dim))
    batch_bodies = np.zeros((batch_size,body_truncate_len,data_dim))
    batch_labels = []
    batch_body_len = []
    batch_headline_len = []
    batch_ids = []
    batch_input_len = []

    for i in xrange(batch_size):
        index = indices.pop(0)
        line = lines[index]
        line = line.split("|")
        #Index|Headline|Stance ID|Body ID|Stance|articleBody|New Body ID
        raw_headline = line[1]
        raw_body = line[5]
        raw_label = line[4]
        body_id = line[3]
        stance_id = line[2]

        headline = np.transpose(sentence_to_mat(raw_headline,w2v)) #num_words x w2v_dim
        body = np.transpose(sentence_to_mat(raw_body,w2v)) #num_words x w2v_dim

        #Format input into the batch
        if body.shape[0] > body_truncate_len:
            body_len = body_truncate_len
            batch_bodies[i,:,:] = body[:body_len,:]
            batch_body_len.append(body_len)
        else:
            body_len = body.shape[0]
            batch_body_len.append(body_len)
            batch_bodies[i,:body_len,:] = body[:body_len,:]

        if headline.shape[0] > headline_truncate_len:
            head_len = headline_truncate_len
            batch_headlines[i,:,:] = headline[:headline_truncate_len,:]
            batch_headline_len.append(headline_truncate_len)
        else:
            head_len = headline.shape[0]
            batch_headlines[i,:head_len,:] = headline[:head_len,:]
            batch_headline_len.append(head_len)

        batch_input_len.append(head_len + body_len)

        label = one_hot_encode(raw_label)
        batch_labels.append(label)
        batch_ids.append((stance_id,body_id))

    batch_headlines = np.array(batch_headlines).astype(np.float32)
    batch_bodies = np.array(batch_bodies).astype(np.float32)
    batch_headline_len = np.array(batch_headline_len).astype(np.int32)
    batch_body_len = np.array(batch_body_len).astype(np.int32)
    batch_input_len = np.array(batch_input_len).astype(np.int32)
    batch_labels = np.array(batch_labels).astype(np.int64)

    batch = {'batch_headlines': batch_headlines,
             'batch_bodies': batch_bodies,
             'batch_labels': batch_labels,
             'batch_ids': batch_ids,
             'batch_headline_len': batch_headline_len,
             'batch_body_len': batch_body_len,
             'batch_input_len': batch_input_len}


    return batch,indices


def one_hot_encode(label):
    mapping = {'unrelated':0, 'agree':1, 'discuss':2, 'disagree':3}
    return mapping[label]


def simple_combine(headlines,bodies):
    """Naively concatenates headlines and bodies"""

    input_batch = np.concatenate([headlines,bodies],axis = 1) #num_words x w2v_dim

    #Input placeholder (Batchsize, self.max_timesteps, self.data_dim)
    return input_batch


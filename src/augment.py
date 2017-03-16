import utils

def index_replacement_words(s_vec,w2v_model,p):
    '''
    chooses which words from the sentence to replace
    returns list of indecies of chosen words
    '''
    import numpy as np
    import random
    w2v_ind = get_w2v_words(s_vec,w2v_model)
    r = np.random.geometric(p)
    r = r%len(w2v_ind)
    chosen_index = random.sample(range(len(w2v_ind)),k=r)
    chosen_words=[w2v_ind[i] for i in chosen_index]
    return chosen_words

def get_w2v_words(s_vec,w2v_model):
    '''
    returns a list of words from the sentence that are in the word2vec dictionary
    '''
    w2v_words_index = []
    for i, word in enumerate(s_vec):
        if word in w2v_model.vocab:
            w2v_words_index.append(i)
    return w2v_words_index

def replace_chosen_words(s_vec,chosen,w2v_model,q,t):
    '''
    replaces chosen words from sentence with similar (word2vec cosine) words
    '''
    new_sent=s_vec[:]
    for index in chosen:
        word = s_vec[index]
        similar_words = w2v_model.most_similar(positive=[word],topn=10)
        for i,(similar_word,similarity) in enumerate(similar_words):
            if similarity<t:
                del similar_words[i]
        import numpy as np
        if len(similar_words)>0:
            s = np.random.geometric(q)
            s = s%len(similar_words)
            new_word = similar_words[s][0]
            new_sent[index]=new_word.encode('ascii','ignore')
    return new_sent

def replace_w2v(s_vec,w2v_model,p=0.5,q=0.5,t=0.25):
    '''
    returns an alternative sentence for data augmentation
    '''
    chosen = index_replacement_words(s_vec,w2v_model,p)
    return replace_chosen_words(s_vec, chosen, w2v_model,q,t)

def w2v_augment():
    w2v_file = 'googlenews-vectors-negative300.bin.gz'
    w2v_url = 'https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM'

    w2v = utils.load_w2v(w2v_file,w2v_url)

    sentence = 'this is a test sentence'
    split_sentence=sentence.split()
    print split_sentence

    new_sentence = replace_w2v(split_sentence,w2v)
    print new_sentence

if __name__ == '__main__':
    w2v_augment()

import numpy as np
import pandas as pd
import re
import utils
import time


from itertools import izip,tee 
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from gensim.models.keyedvectors import KeyedVectors
from nltk import wordnet 

def augment_dataset(df):
    """Takes a dataframe (train or test) and adds in new rows of sentences that have their words switched out for synonyms"""

    #for each sentence in df

    new_sentences = generate_similar_sentences(sentence)

def generate_similar_sentences(sentence,w2v,percentage_to_replace=.8,max_syn=5,num_outputs=20):
    """Takes a sentence, switches out non compond words, returning a list of similar sentences
        percentage_to_replace is the percentage of the words to replace (it will ignore compound words and stop words if picked)
        max_syn is the max number of synonyms to look at for a given word 
        Num_outputs is number of sentences to return. """

    list_of_sentences = []

    words = pairwise_tokenize(sentence,w2v,remove_stopwords=False) #This has combined any compound words found in word2vec

    print words
    #if word contains underscore don't sub in synonyms
    dont_sub_idx = []
    compound_word_idx = []
    deleted_idx = []
    for idx,word in enumerate(words):
        if "_" in word or word in stopwords.words('english'):
            dont_sub_idx.append(idx)
        if "_" in word:
            compound_word_idx.append(idx)
            deleted_idx.append(idx+1)

    pattern = re.compile('[\W_]+') 
    sentence = pattern.sub(" ",sentence).lower()
    tagged = pos_tag(sentence.split(" ")) #Pos_tag needs to use the original sentence to tag parts of speech, we will now delete indices that correspond to words that no longer exist b/c of compound
    
    for idx in reversed(compound_word_idx):
        tagged.pop(idx+1)
        
    for tag in tagged:
        if tag[1] == 'NNP':
            dont_sub_idx.append(idx)
            
    for i in xrange(num_outputs):
        new_words = words
        mask = np.random.randn(len(words))
        for j in xrange(len(words)):
            if mask[j] < percentage_to_replace and j not in dont_sub_idx:
                pos = wordnet_pos_code(tagged[j][1])
                synonyms = get_synonyms(words[j],pos,max=max_syn)
                if len(synonyms) != 0:
                    new_words[j] = synonyms[np.random.randint(0,min(max_syn,len(synonyms)))]
               # print words[j],synonyms
        list_of_sentences.append(" ".join(new_words))


    return list_of_sentences

def pairwise_tokenize(sentence,w2v,remove_stopwords=True):
    """ Returns list of valid words + compound words
        Naively looks at each pair of words in the sentence and checks if it is in word2vec
        If so, it'll merge it. Then it'll remove stopwords. For the remaining words, it'll add the single word representation"""

    ignore_words = stopwords.words('english')

    #Remove non-alphanumeric
    pattern = re.compile('[\W_]+') 
    sentence = pattern.sub(" ",sentence)  
    words = sentence.split(" ")

    compound_word_idx = []
    a_idx = 0
    for a,b in pairwise(words):
        combined = a +"_" + b
        try:
            w2v[combined]
            print combined
            compound_word_idx.append(a_idx) #append the index of the 1st compound word
            a_idx += 1
        except KeyError:
            a_idx += 1

    for idx in compound_word_idx:
        words[idx] = words[idx] + "_" + words[idx + 1] #compound_word_idx stores index of 1st word, so combine with the next word

    #This cannot be combined into another loop to maintain where indices point
    for idx in reversed(compound_word_idx):
        words.pop(idx+1)

    if remove_stopwords == True:
        filtered = []
        for word in words:
            word = word.decode("utf-8")
            if word not in ignore_words:
                filtered.append(word)

        words = filtered

    return words

def get_synonyms(word,pos,max=5):
    """returns list of synonyms"""
    synonyms = []
    count = 0
    synsets = wordnet.synsets(word,pos=pos)
    for synset in synsets:
        for lemma in synset.lemma_names():
            if count >= max:
                break
            synonyms.append(lemma)
            count += 1

    return synonyms

def get_hyponyms(word,pos,max=5):
    """returns list of synonyms"""
    hyponyms = []
    count = 0
    synset = wordnet.synset(word,pos=pos)
    for hyp in synset.hy:
        for lemma in synset.lemma_names():
            if count >= max:
                break
            hyponyms.append(lemma)
            count += 1

    return hyponyms

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

def wordnet_pos_code(tag):
    if tag.startswith('NN'):
        return wordnet.NOUN
    elif tag.startswith('VB'):
        return wordnet.VERB
    elif tag.startswith('JJ'):
        return wordnet.ADJ
    elif tag.startswith('RB'):
        return wordnet.ADV
    else:
        return ''

def run_test(training_filename):
    data = pd.read_csv(training_filename)
    w2v_file = 'GoogleNews-vectors-negative300.bin.gz'
    w2v_url = 'https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM'

    w2v = utils.load_w2v(w2v_file,w2v_url)
    #Comparison
    with open('tokenize_check.txt','wb') as output_file:
        for i in xrange(10):
            original = data['Headline'].iloc[i]
            p_words = pairwise_tokenize(original,True,w2v)
            pair_sent = "|".join(words)
            output_file.write(original + "\n")
            output_file.write("111: " + pair_sent +"\n")
            output_file.write("\n")

    # start = time()
    # for i in xrange(500):
    #   original = data['Headline'].iloc[i]
    #   p_words = pairwise_tokenize(original,True,w2v)

    # dur = time() - start
    # print "pairwise: " + str(dur)

if __name__ == "__main__":
    run_test("../data/training-all.csv")
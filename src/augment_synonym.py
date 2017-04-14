import numpy as np
import pandas as pd
import re
import utils
import time


from itertools import izip,tee 
from nltk.corpus import stopwords,wordnet
from nltk.tag import pos_tag
from gensim.models.keyedvectors import KeyedVectors

def augment_headlines(df,w2v,output_filename,nrows=None):
    """Takes a dataframe (train or test) and returns a new dataframe with new rows of sentences that have their words switched out for synonyms
        Only does this for the first nrows, defaults to all"""
    if nrows == None:
        nrows = len(df)

    ave_dur = []
    #header= "Index,Headline,Stance ID,Body ID,Stance,articleBody,New Body ID"
    header= "Index|Headline|Stance ID|Body ID|Stance|articleBody|New Body ID"

    Indices = df.index
    Headlines = df.Headline
    StanceIDs = df.loc[:,"Stance ID"]
    BodyIDs = df.loc[:,"Body ID"]
    Stances = df.loc[:,"Stance"]
    Bodies = df.loc[:,"articleBody"]
    NewBodyIDs = df.loc[:,"New Body ID"]
    
    with open(output_filename,'wb') as write_file:
        write_file.write(header + "\n")
        for i in xrange(nrows):
            if i % 100 == 0:
                print i
            index = str(Indices[i])
            headline = Headlines.iloc[i]
            stanceID = StanceIDs.iloc[i]
            bodyid = str(BodyIDs.iloc[i])
            stance = Stances.iloc[i]
            body = Bodies.iloc[i].replace('\n','').replace("|",'')
            newbodyid = str(NewBodyIDs.iloc[i])

            start_time = time.time()
            new_sentences = generate_similar_sentences(headline,w2v)
            row = "|".join([str(index),str(headline),str(stanceID),str(bodyid),str(stance),str(body),str(newbodyid)])
            write_file.write(row + "\n")

            for counter,new_sent in enumerate(new_sentences):
                amended_stance_id = str(stanceID) + "_" + str(counter)
                row = "|".join([str(index),str(new_sent),amended_stance_id,str(bodyid),str(stance),str(body),str(newbodyid)])
                write_file.write(row + "\n")

            dur = time.time() - start_time
            ave_dur.append(dur)
            
    print "generated dur" + str(sum(ave_dur)/len(ave_dur))
    
    return 

def generate_similar_sentences(sentence,w2v,percentage_to_replace=1,max_syn=10,num_outputs=50):
    """Takes a sentence, switches out non compond words, returning a list of similar sentences
        it will ignore compound words and stop words if picked
        max_syn is the max number of synonyms to look at for a given word 
        Num_outputs is number of sentences to return. """

    list_of_sentences = []

    words = pairwise_tokenize(sentence,w2v,remove_stopwords=False) #This has combined any compound words found in word2vec

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
        if not word:
            dont_sub_idx.append(idx)

    pattern = re.compile('[\W_]+') 
    sentence = pattern.sub(" ",sentence).lower().strip()
    tagged = pos_tag(sentence.split(" ")) #Pos_tag needs to use the original sentence to tag parts of speech, we will now delete indices that correspond to words that no longer exist b/c of compound
    
    for idx in reversed(compound_word_idx):
        tagged.pop(idx+1)
        
    for tag in tagged:
        if tag[1] == 'NNP':
            dont_sub_idx.append(idx)
            
    for i in xrange(num_outputs):
        new_words = words
        mask = np.random.random_sample(len(words))
        for j in xrange(len(words)):
            if mask[j] < .5 and j not in dont_sub_idx:
                pos = wordnet_pos_code(tagged[j][1])
                synonyms = get_synonyms(words[j],w2v,pos,max=max_syn)
                if len(synonyms) != 0:
                    new_words[j] = synonyms[np.random.randint(0,min(max_syn,len(synonyms)))]
        list_of_sentences.append(" ".join(new_words))

    list_of_sentences = set(list_of_sentences)
    return list(list_of_sentences)

def pairwise_tokenize(sentence,w2v,remove_stopwords=True):
    """ Returns list of valid words + compound words
        Naively looks at each pair of words in the sentence and checks if it is in word2vec
        If so, it'll merge it. Then it'll remove stopwords. For the remaining words, it'll add the single word representation"""

    ignore_words = stopwords.words('english')

    #Remove non-alphanumeric
    pattern = re.compile('[\W_]+') 
    sentence = pattern.sub(" ",sentence)  
    sentence = sentence.strip()
    words = sentence.split(" ")

    compound_word_idx = []
    a_idx = 0
    for a,b in pairwise(words):
        combined = a +"_" + b
        try:
            w2v[combined]
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

def get_synonyms(word,w2v,pos,max=20):
    """returns list of synonyms"""
    synonyms = []
    count = 0
    synsets = wordnet.synsets(word,pos=pos)
    for synset in synsets:
        candidate_names = []
        for lemma in synset.lemma_names():
            candidate_names.append(lemma)
        for hypo in synset.hyponyms():
            candidate_names.append(hypo)
        for hyper in synset.hypernyms():
            candidate_names.append(hyper)

        for lemma in candidate_names:
            if count >= max:
                break
            # print pos,word,lemma
            try:
                similarity = w2v.n_similarity([word.lower()],[lemma.lower() ])
                if similarity > .34 and lemma not in synonyms:
                    synonyms.append(lemma)

                    count += 1
            except:
                continue

    return synonyms

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
    
    ##Test for Pairwise_Tokenize
    # data = pd.read_csv(training_filename)
    # w2v_file = 'GoogleNews-vectors-negative300.bin.gz'
    # w2v_url = 'https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM'

    # w2v = utils.load_w2v(w2v_file,w2v_url)
    # #Comparison
    # with open('tokenize_check.txt','wb') as output_file:
    #     for i in xrange(10):
    #         original = data['Headline'].iloc[i]
    #         p_words = pairwise_tokenize(original,True,w2v)
    #         pair_sent = "|".join(words)
    #         output_file.write(original + "\n")
    #         output_file.write("111: " + pair_sent +"\n")
    #         output_file.write("\n")


    #Run Augmented headlines
    w2v_file = 'GoogleNews-vectors-negative300.bin.gz'
    w2v_url = 'https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM'
    print "loading w2v"
    w2v = utils.load_w2v(w2v_file,w2v_url)

    data = pd.read_csv(training_filename)
    print "generating headlines"
    augmented_temp = augment_headlines(data,w2v,"temp.txt")
    new_data = pd.read_csv("temp.txt",sep="|")


if __name__ == "__main__":
    run_test("../dataset/train_split.csv")

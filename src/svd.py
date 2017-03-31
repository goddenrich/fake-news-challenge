import numpy as np
import pandas as pd
import sklearn
import augment_synonym
import utils
from gensim.models.keyedvectors import KeyedVectors


def sentence_to_mat(sentence,w2vmodel):
	temp = []
	words = augment_synonym.pairwise_tokenize(sentence,w2vmodel,remove_stopwords=True)
	for j in words:
		try:
			temp.append(w2vmodel[j].reshape((300,1)))
		except KeyError:
			continue
	temp = np.concatenate(temp, axis=1)
	return temp

def normalise(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def cos_angle(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = normalise(v1)
    v2_u = normalise(v2)
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)


def run_test(headlines, bodies, stance, w2v):
	unrelated = []
	agree = []
	discuss = []
	disagree = []
	related = []

	for i in xrange(1000):
		head = sentence_to_mat(headlines[i], w2v)
		bod = sentence_to_mat(bodies[i],w2v)
		U_head, S_head, V_head = np.linalg.svd(head)
		mat_1 = U_head[:,:1]
		U_bod, S_bod, V_bod = np.linalg.svd(bod)
		mat_2 = U_bod[:,:1]
	

		#error = cos_angle(mat_1.reshape(300), mat_2.reshape(300))
		error = np.dot(mat_1.reshape(300), mat_2.reshape(300))

		#eigenValues,eigenVectors = linalg.eig(A)

		#idx = eigenValues.argsort()[::-1]   
		#eigenValues = eigenValues[idx]
		#eigenVectors = eigenVectors[:,idx]
		#print(error)
		#print(stance[i])
		
		if(stance[i] == 'unrelated'):
			unrelated.append(error)
		else :
			related.append(error)
			if(stance[i] == 'discuss'):
				discuss.append(error)
			elif(stance[i] == 'agree'):
				agree.append(error)
			else :
				disagree.append(error)
		
	print("unrelated median:", np.median(unrelated))
	print("unrelated average:", np.average(unrelated))
	print("related median:", np.median(related))
	print("related average:", np.average(related))
	print("agree median:", np.median(agree))
	print("agree average:", np.average(agree))
	print("disagree median:", np.median(disagree))
	print("disagree average:", np.average(disagree))
	print("discuss median:", np.median(discuss))
	print("discuss average:", np.average(discuss))

def main():

	try:
		data = pd.read_csv("../data/training-all.csv")
	except IOError:
		print("File not found, try running cleanup.py")

	bodies = np.array(data['articleBody'])
	headlines = np.array(data['Headline'])
	stance = np.array(data['Stance'])

	w2v_file = 'GoogleNews-vectors-negative300.bin.gz'
	w2v_url = 'https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM'


	w2v = utils.load_w2v(w2v_file,w2v_url)
	if w2v==False:
		print 'failed to load the file'
		return

	run_test(headlines, bodies, stance, w2v)

if __name__ == '__main__':
	main()

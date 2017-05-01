import sys
import os
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import svm
from sklearn import linear_model
from sklearn import neighbors
from sklearn import tree
from sklearn import ensemble
from sklearn.feature_selection import SelectFromModel
import operator
import re
import svd

#import augment_synonym
import utils
import sktensor

def prep_tensor_data(headlines, bodies, stance, w2v):
	"""prepare data to be fed into classifier for tensor methods
	take in numpy array of headlines bodies and stances and w2v model
	return X and y matrices"""

	X = []
	y = []
	
	for i in xrange(headlines.shape[0]):
		#print(i)
		head = svd.sentence_to_mat(headlines[i], w2v)
		head_avg = np.mean(head, axis=1)

		bod = svd.sentence_to_mat(bodies[i],w2v)
		bod_avg = np.mean(bod, axis=1)

		U_head, S_head, V_head = np.linalg.svd(head)
		mat_1 = U_head[:,:3]
		U_bod, S_bod, V_bod = np.linalg.svd(bod)
		mat_2 = U_bod[:,:7]
		
		tmp = np.concatenate((head_avg.reshape((head_avg.shape[0],1)), mat_1), axis=1)
		#print(tmp.shape)
		tmp2 = np.concatenate((bod_avg.reshape((bod_avg.shape[0],1)), mat_2), axis=1)
		#print(tmp2.shape)
		#print(np.concatenate((tmp,tmp2), axis=1).shape)
		X.append(np.concatenate((tmp, tmp2), axis=1))
		y.append(stance[i])

	return np.dstack(X).T, np.hstack(y)



def create_dataset(headlines_train, bodies_train, stance_train, headlines_test, bodies_test, stance_test):
	"""Given w2v model, train and test headlines, bodies and stance, return Dataset
	either by reading in already made csv's or calling data_splitting functions"""

	w2v_file = 'GoogleNews-vectors-negative300.bin.gz'
	w2v_url = 'https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM'


	w2v = utils.load_w2v(w2v_file,w2v_url)
	if w2v==False:
		print 'failed to load the file'
		return

	try:
		tmp = pd.read_csv('training-all.csv')
	except IOError:
		print("File not found, try running cleanup.py")

		"""
	for (idx, data) in enumerate(even_split(tmp)):
		
		try:
			data = pd.read_csv(path +'_split.csv')
		except IOError:
			print("File not found, try running cleanup.py")
		
		bodies = np.array(data['articleBody'])
		headlines = np.array(data['Headline'])
		stance = np.array(data['Stance'])

		#if(idx==0):
		X_train, y_train = prep_stanford(headlines, bodies, stance, w2v, concat)
		print(X_train.shape)
		print(y_train.shape)
		#else:
		X_test, y_test = prep_stanford(headlines, bodies, stance, w2v, concat)
		print(X_test.shape)
		print(y_test.shape)
		"""
	X_train, y_train = prep_tensor_data(headlines_train, bodies_train, stance_train, w2v)
	X_test, y_test = prep_tensor_data(headlines_test, bodies_test, stance_test, w2v)
	
	return X_train, y_train, X_test, y_test



def main():
	"""
	########run this code first to create dataset
	create_dataset()
	"""
	#X_train, y_train, X_test, y_test = create_dataset()
		
	open('tensor_results.csv', 'w')

	T_train = pd.read_csv('poly_train.csv', sep='|')
	T_test = pd.read_csv('poly_test.csv', sep='|')
	
	headlines_train = np.array(T_train['Headline'])
	bodies_train = np.array(T_train['articleBody'])
	stance_train = np.array(T_train['Stance'])
		
	headlines_test = np.array(T_test['Headline'])
	bodies_test = np.array(T_test['articleBody'])
	stance_test = np.array(T_test['Stance'])

	X_train, y_train, X_test, y_test = create_dataset(headlines_train, bodies_train, stance_train, headlines_test, bodies_test, stance_test)
	print(X_train.shape)
	print(X_test.shape)
	print(y_train.shape)
	print(y_test.shape)

	lookup = {}
	df = pd.DataFrame(X_train)
	df_y = pd.DataFrame(y_train)
	agree = df.loc[df_y['0'] == 'agree']
	disagree = df.loc[df_y['0'] == 'disagree']
	discuss = df.loc[df_y['0'] == 'discuss']
	unrelated = df.loc[df_y['0'] == 'unrelated']


	U_agree = sktensor.hsovd(np.array(agree), [5,5,5])
	U_disagree = sktensor.hsovd(np.array(disagree), [5,5,5])
	U_discuss = sktensor.hsovd(np.array(discuss), [5,5,5])
	U_unrelated = sktensor.hsovd(np.array(unrelated), [5,5,5])





if __name__ == '__main__':
	main()

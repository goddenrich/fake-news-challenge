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


#import augment_synonym
import utils
#from gensim.models.keyedvectors import KeyedVectors
import score
import svd
from data_splitting import even_split

"""
Possible feature functions
Calculate surface lexical similarity between two word snippets s1 and s2

def suface_lexical_similarity(s1, s2):
"""

def prepare_data(headlines, bodies, stance, w2v, function_1):
	"""Prepare data for Classifiers
	looks at top eigenvector"""

	X = []
	y = []
	
	for i in xrange(headlines.shape[0]):
		#print(i)
		head = svd.sentence_to_mat(headlines[i], w2v)
		bod = svd.sentence_to_mat(bodies[i],w2v)
		U_head, S_head, V_head = np.linalg.svd(head)
		mat_1 = U_head[:,:1]
		U_bod, S_bod, V_bod = np.linalg.svd(bod)
		mat_2 = U_bod[:,:1]
		
		X.append(function_1(mat_1, mat_2))
		y.append(stance[i])

	return np.hstack(X).T, np.hstack(y)
		#error = cos_angle(mat_1.reshape(300), mat_2.reshape(300))

def prep_data2(headlines, bodies, stance, w2v, function_1, flag=1):
	"""Prepare data for Classifiers
	looks at top eigenvector and average together concatenate
	adds augmentation by splitting in half"""

	X = []
	y = []
	
	for i in xrange(headlines.shape[0]):
		#print(i)
		head = svd.sentence_to_mat(headlines[i], w2v)
		head_avg = np.mean(head, axis=1)
		bod = svd.sentence_to_mat(bodies[i],w2v)
		U_head, S_head, V_head = np.linalg.svd(head)
		mat_1 = function_1(U_head[:,:1], head_avg.reshape((head_avg.shape[0],1)))
		if(flag==1 and bod is not None and bod.shape[1]>1):
			for j in np.array_split(bod,2,axis=1):
				#print(j)
				bod_avg = np.mean(j, axis=1)
				U_bod, S_bod, V_bod = np.linalg.svd(j)
				mat_2 = function_1(U_bod[:,:1], bod_avg.reshape((bod_avg.shape[0],1)))
			
				X.append(function_1(mat_1, mat_2))
				y.append(stance[i])
		elif(bod is not None):
			bod_avg = np.mean(bod, axis=1)
			U_bod, S_bod, V_bod = np.linalg.svd(bod)
			mat_2 = function_1(U_bod[:,:1], bod_avg.reshape((bod_avg.shape[0],1)))
		
			X.append(function_1(mat_1, mat_2))
			y.append(stance[i])

	return np.hstack(X).T, np.hstack(y)

def most_sim(eig, original):
	original_avg = np.mean(original, axis=1)
	tmp = {}
	for x in eig.T:
		tmp[tuple(x)] = svd.cos_angle(x, original_avg)

	#print(tmp)
	#print(sorted(tmp.iteritems(), key= operator.itemgetter(1), reverse=True)[:1][0][0])
	return sorted(tmp.iteritems(), key= operator.itemgetter(1), reverse=True)[:1][0][0]

def prep_stanford(headlines, bodies, stance, w2v, remove_stopwords=True):
	"""bag of words model
	preparation of data"""

	X = []
	y = []

	for i in xrange(headlines.shape[0]):
		#if (i % 100 == 0):
		#	print(i)
		head = svd.sentence_to_mat(headlines[i], w2v, remove_stopwords)
		head_avg = np.mean(head, axis=1)

		avgs = {}
		#for j in bodies[i].split('.'):
		for j in re.split('/\?|\.|!|;|:/', bodies[i]):
			temp = svd.sentence_to_mat(j, w2v, remove_stopwords)
			#print(temp.shape)
			if(temp is not None):
				bod_avg = np.mean(temp, axis=1)
			#print(bod_avg)
				avgs[tuple(bod_avg)] = svd.cos_angle(head_avg, bod_avg)

		newA = list(sorted(avgs.iteritems(), key=operator.itemgetter(1), reverse=True)[:1])

		final = []
		for k in newA:
			final.append(k[0])
		#while(len(final) < 3):
		#	final.append(final[-1])
		#print(np.concatenate((np.hstack(final), head_avg), axis=0).shape)
		tmp = np.concatenate((np.hstack(final), head_avg), axis=0)
		#print(tmp.shape)
		X.append(tmp.reshape(tmp.shape[0], 1))
		y.append(stance[i])
	#print(len(X))
	return np.hstack(X).T, np.hstack(y)

def concat(mat_1, mat_2):
	return np.concatenate((mat_1, mat_2), axis=0)

def subtract(mat_1, mat_2):
	return mat_1 - mat_2

def comb_sum(mat_1, mat_2):
	return mat_1 + mat_2


def grid_search(X_train, X_test, y_train, y_test, estimator, param_grid):
	clf = model_selection.GridSearchCV(estimator= estimator, param_grid=param_grid, scoring=scorer, cv=5)
	clf.fit(X_train, y_train)
	#print(clf.best_score_)
	#print(clf.best_params_)
	#print(clf.cv_results_)

	test_clf = clf.best_estimator_
	test_clf.fit(X_train, y_train)
	#print(test_clf)
	pred = test_clf.predict(X_test)
	temp = score.report_score(y_test, pred)
	#temp = np.argwhere(y_test==0)
	#temp1 = np.argwhere(y_test==1)
	#ix1 = np.in1d(y_test.ravel(), 1).reshape(y_test.shape)
	#ix0 = np.in1d(y_test.ravel(), 0).reshape(y_test.shape)
	#print(np.array(X_test[ix0]['A']))
	#print(X_test[np.array(temp1)])
	#print(temp)
	#plotDB(test_clf, X_test[ix0]['A'], X_test[ix0]['B'], X_test[ix1]['A'], X_test[ix1]['B'], "Decision Tree")
	#print(estimator, clf.best_score_)

	return test_clf, test_clf.score(X_train, y_train), test_clf.score(X_test, y_test), temp

def random_forest(X_train, X_test, y_train, y_test):
	estimator = ensemble.RandomForestClassifier()
	#param_grid = [{'min_samples_split': np.arange(2,11)}]
	param_grid = [{'min_samples_split': [7]}]
	return grid_search(X_train, X_test, y_train, y_test, estimator, param_grid)

def svm_linear_kernel(X_train, X_test, y_train, y_test):
	estimator = svm.SVC()
	#param_grid = [{'C': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0], 'kernel': ['linear']},]
	param_grid = [{'C': [0.1, 0.5, 1.0], 'kernel': ['linear']},]
	#should i include std of scores or just look at scores mean
	return grid_search(X_train, X_test, y_train, y_test, estimator, param_grid)

def svm_poly_kernel(X_train, X_test, y_train, y_test):
	estimator = svm.SVC()
	param_grid = [{'C': [0.1,0.5,1.0], 'kernel': ['poly'], 'degree': [3, 4], 'gamma':[0.5, 1.0]}]
	#should i include std of scores or just look at scores mean
	return grid_search(X_train, X_test, y_train, y_test, estimator, param_grid)

def svm_rbf_kernel(X_train, X_test, y_train, y_test):
	estimator = svm.SVC()
	#param_grid = [{'C': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0] , 'kernel': ['rbf'], 'gamma':[0.1, 0.5, 1.0, 3.0]}]
	#param_grid = [{'C': [1.0, 2.0] , 'kernel': ['rbf'], 'gamma':[0.8, 1.0]}]
	param_grid = [{'C': [1.0, 2.0] , 'kernel': ['rbf'], 'gamma':[0.8]}]
	#should i include std of scores or just look at scores mean
	return grid_search(X_train, X_test, y_train, y_test, estimator, param_grid)

def k_nn(X_train, X_test, y_train, y_test):
	estimator = neighbors.KNeighborsClassifier()
	#param_grid = [{'n_neighbors': np.arange(1,51), 'leaf_size': np.arange(5,65,5)}]
	#should i include std of scores or just look at scores mean
	param_grid = [{'leaf_size': [30]}]
	return grid_search(X_train, X_test, y_train, y_test, estimator, param_grid)

def decision_tree(X_train, X_test, y_train, y_test):
	estimator = tree.DecisionTreeClassifier()
	param_grid = [{'min_samples_split': np.arange(2,11)}]
	#should i include std of scores or just look at scores mean
	return grid_search(X_train, X_test, y_train, y_test, estimator, param_grid)

def scorer(estimator, X, y):
	y_pred = estimator.predict(X)
	return score.report_score(y, y_pred)/100.0

def output_text(file, estimator, best_score, test_score):
	d = {'1':estimator, '2': best_score, '3': test_score}
	df = pd.DataFrame(data=d, index=xrange(1))

	if not os.path.isfile(file):
		df.to_csv(file,header =False, index=False)
	else :
		df.to_csv(file,mode = 'a',header=False, index=False)

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
	X_train, y_train = prep_data2(headlines_train, bodies_train, stance_train, w2v, concat, flag=1)
	X_test, y_test = prep_data2(headlines_test, bodies_test, stance_test, w2v, concat, flag=0)
	
	return X_train, y_train, X_test, y_test
		#print(y)
		#np.savetxt('stanford_X_' + str(idx) + '.csv', X, delimiter=',')
		#np.savetxt('stanford_y_' + str(idx) + '.csv', y, delimiter=',', fmt="%s")

def main():

	"""
	########run this code first to create dataset
	create_dataset()
	"""
	#X_train, y_train, X_test, y_test = create_dataset()
		
	open('concat_aug_lasso_final2.csv', 'w')

	T_train = pd.read_csv('train_80.csv', sep='|')
	T_test = pd.read_csv('test_20.csv', sep='|')
	
	headlines_train = np.array(T_train['Headline'])
	bodies_train = np.array(T_train['articleBody'])
	stance_train = np.array(T_train['Stance'])
		
	headlines_test = np.array(T_test['Headline'])
	bodies_test = np.array(T_test['articleBody'])
	stance_test = np.array(T_test['Stance'])

	X_train, y_train, X_test, y_test = create_dataset(headlines_train, bodies_train, stance_train, headlines_test, bodies_test, stance_test)

		#print(T_train.shape)
		#print(T_test.shape)
		#print(T_train.columns)
		#print(T_test.columns)
		
		#print(X_train.shape)
		#print(y_train.shape)
		#print(X_test.shape)
		#print(y_test.shape)
		#T_train = np.array(T_train)
		#T_test = np.array(T_test)
		
		#y_train = np.array(pd.read_csv('stanford_y_train.csv', header=None))
	#y_test = np.array(pd.read_csv('stanford_y_test.csv', header=None))

	y_train = y_train.reshape(y_train.shape[0])
	y_test = y_test.reshape(y_test.shape[0])

	print(X_train.shape)
	print(y_train.shape)
	print(X_test.shape)
	print(y_test.shape)
		
	m, n, m_, n_ = random_forest(X_train, X_test, y_train, y_test)
	print(m)
	output_text('concat_aug_lasso_final2.csv', n, m_, n_)

	m.fit(X_train, y_train)
	model = SelectFromModel(m, prefit=True)
	X_train_new = model.transform(X_train)
	print(X_train_new.shape)
	X_test_new = model.transform(X_test)
	#a, b, a_, b_ = svm_linear_kernel(X_train, X_test, y_train, y_test)
	c, d, c_, d_ = svm_poly_kernel(X_train_new, X_test_new, y_train, y_test)
	print(c)
	output_text('concat_aug_lasso_final2.csv',d,c_,d_)

	
	"""
	for path in ['subtract', 'concat']:
		open(path + '_stanford_final.csv', 'w')
		X_train = np.array(pd.read_csv(path + '_X_train.csv', header=None))
		X_test = np.array(pd.read_csv(path + '_X_test.csv', header=None))
		y_train = np.array(pd.read_csv(path + '_y_train.csv', header=None))
		y_test = np.array(pd.read_csv(path + '_y_test.csv', header=None))

		y_train = y_train.reshape(y_train.shape[0])
		y_test = y_test.reshape(y_test.shape[0])



		#a, b, a_, b_ = svm_linear_kernel(X_train, X_test, y_train, y_test)
		#c, d, c_, d_ = svm_poly_kernel(X_train, X_test, y_train, y_test)
		e, f, e_, f_  = svm_rbf_kernel(X_train, X_test, y_train, y_test)
		#i, j, i_, j_ = k_nn(X_train, X_test, y_train, y_test)
		#k, l, k_, l_ = decision_tree(X_train, X_test, y_train, y_test)
		#m, n, m_, n_ = random_forest(X_train, X_test, y_train, y_test)

		#output_text(path + '_final.csv', a, b, a_, b_)
		#output_text(path + '_final.csv', c, d, c_, d_)
		output_text(path + '_stanford_final.csv', e, f, e_, f_)
		#output_text(path + '_final.csv', i, j, i_, j_)
		#output_text(path + '_final.csv', k, l, k_, l_)
		#output_text(path + '_final.csv', m, n, m_, n_)

		#print(y_train.shape)
		#print(y_test.shape)
		#print(random_forest(X_train, X_test, y_train.reshape(y_train.shape[0]), y_test.reshape(y_test.shape[0])))
		#clf = ensemble.RandomForestClassifier()
		#clf.fit(X_train, y_train)

		#print(clf.score(X_train, y_train))
		#print(clf.score(X_test, y_test))
		#np.savetxt('result1.csv', clf.predict(X_test), delimiter=',', fmt="%s")
		
		#tx = clf.predict(X_test)

		#print(score.report_score(y_test, tx))
	"""


if __name__ == '__main__':
	main()


	

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

#import augment_synonym
import utils
#from gensim.models.keyedvectors import KeyedVectors
import score
import svd

"""
Possible feature functions
Calculate surface lexical similarity between two word snippets s1 and s2

def suface_lexical_similarity(s1, s2):
"""


###try different operations on top eigeinvectors, concatenate, subtract, sum
def prepare_data(headlines, bodies, stance, w2v, function_1):
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
		

def concat(mat_1, mat_2):
	return np.concatenate((mat_1, mat_2), axis=0)

def subtract(mat_1, mat_2):
	return mat_1 - mat_2

def comb_sum(mat_1, mat_2):
	return mat_1 + mat_2


def grid_search(X_train, X_test, y_train, y_test, estimator, param_grid):
	clf = model_selection.GridSearchCV(estimator= estimator, param_grid=param_grid, cv=5)
	clf.fit(X_train, y_train)
	#print(clf.best_score_)
	#print(clf.best_params_)
	#print(clf.cv_results_)

	test_clf = clf.best_estimator_
	test_clf.fit(X_train, y_train)

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
	param_grid = [{'min_samples_split': np.arange(2,11)}]
	return grid_search(X_train, X_test, y_train, y_test, estimator, param_grid)

def svm_linear_kernel(X_train, X_test, y_train, y_test):
	estimator = svm.SVC()
	#param_grid = [{'C': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0], 'kernel': ['linear']},]
	param_grid = [{'C': [0.1, 0.5, 1.0], 'kernel': ['linear']},]
	#should i include std of scores or just look at scores mean
	return grid_search(X_train, X_test, y_train, y_test, estimator, param_grid)

def svm_poly_kernel(X_train, X_test, y_train, y_test):
	estimator = svm.SVC()
	param_grid = [{'C': [0.1, 1.0], 'kernel': ['poly'], 'degree': [4,5], 'gamma':[0.1,1.0]}]
	#should i include std of scores or just look at scores mean
	return grid_search(X_train, X_test, y_train, y_test, estimator, param_grid)

def svm_rbf_kernel(X_train, X_test, y_train, y_test):
	estimator = svm.SVC()
	#param_grid = [{'C': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0] , 'kernel': ['rbf'], 'gamma':[0.1, 0.5, 1.0, 3.0]}]
	param_grid = [{'C': [0.1, 1.0] , 'kernel': ['rbf'], 'gamma':[0.1, 0.5, 1.0]}]
	#should i include std of scores or just look at scores mean
	return grid_search(X_train, X_test, y_train, y_test, estimator, param_grid)

def k_nn(X_train, X_test, y_train, y_test):
	estimator = neighbors.KNeighborsClassifier()
	param_grid = [{'n_neighbors': np.arange(1,51), 'leaf_size': np.arange(5,65,5)}]
	#should i include std of scores or just look at scores mean
	return grid_search(X_train, X_test, y_train, y_test, estimator, param_grid)

def decision_tree(X_train, X_test, y_train, y_test):
	estimator = tree.DecisionTreeClassifier()
	param_grid = [{'min_samples_split': np.arange(2,11)}]
	#should i include std of scores or just look at scores mean
	return grid_search(X_train, X_test, y_train, y_test, estimator, param_grid)

def output_text(file, estimator, best_score, test_score, score_1):
	d = {'1':estimator, '2': best_score, '3': test_score, '4': score_1}
	df = pd.DataFrame(data=d, index=xrange(1))

	if not os.path.isfile(file):
   		df.to_csv(file,header =False, index=False)
	else :
		df.to_csv(file,mode = 'a',header=False, index=False)

def create_dataset():
	for path in ['test', 'train']:
		try:
			data = pd.read_csv('data/data_' + path +'.csv')
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

		X, y = prepare_data(headlines, bodies, stance, w2v, subtract)
		#print(y)
		np.savetxt('subtract_X_' + path + '.csv', X, delimiter=',')
		np.savetxt('subtract_y_' + path + '.csv', y, delimiter=',', fmt="%s")

def main():

	"""
	########run this code first to create dataset
	create_dataset()
	"""
	for path in ['subtract', 'concat']:
		open(path + '_final.csv', 'w')
		X_train = np.array(pd.read_csv(path + '_X_train.csv', header=None))
		X_test = np.array(pd.read_csv(path + '_X_test.csv', header=None))
		y_train = np.array(pd.read_csv(path + '_y_train.csv', header=None))
		y_test = np.array(pd.read_csv(path + '_y_test.csv', header=None))

		y_train = y_train.reshape(y_train.shape[0])
		y_test = y_test.reshape(y_test.shape[0])



		a, b, a_, b_ = svm_linear_kernel(X_train, X_test, y_train, y_test)
		c, d, c_, d_ = svm_poly_kernel(X_train, X_test, y_train, y_test)
		e, f, e_, f_  = svm_rbf_kernel(X_train, X_test, y_train, y_test)
		i, j, i_, j_ = k_nn(X_train, X_test, y_train, y_test)
		k, l, k_, l_ = decision_tree(X_train, X_test, y_train, y_test)
		m, n, m_, n_ = random_forest(X_train, X_test, y_train, y_test)

		output_text(path + '_final.csv', a, b, a_, b_)
		output_text(path + '_final.csv', c, d, c_, d_)
		output_text(path + '_final.csv', e, f, e_, f_)
		output_text(path + '_final.csv', i, j, i_, j_)
		output_text(path + '_final.csv', k, l, k_, l_)
		output_text(path + '_final.csv', m, n, m_, n_)

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
	




if __name__ == '__main__':
	main()


	
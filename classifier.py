#Naive Bayes

import random
import pandas as pd
import numpy as np
import string
import time
from collections import *
from operator import itemgetter
import sklearn.metrics as sk
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

def create_dict(DS, dict_size):
	DS = DS.as_matrix()
	freq = Counter()
	for row in DS:
		# row 0 is class (Y), row 1 is title, row 2 is article
		text = row[1] + row[2]
		lowerReview = text.lower()
		removePunc = str.maketrans("","", string.punctuation)
		pReview = lowerReview.translate(removePunc)
		review = lowerReview.translate(removePunc).split()
		for word in review:
			freq[word] += 1
			
	dictionary = freq.most_common(dict_size)
	return dictionary

def create_binaryBOW (DS, mostCom):
	DS = DS.as_matrix()
	BoW = {}
	for i in range(len(mostCom)):
		key = mostCom[i][0]
		BoW[key] = i
		
	bin_BOW = []

	for row in DS:
		bF = np.zeros((5000,), dtype=int)
		text = row[1] + row[2]
		lowerReview = text.lower()
		removePunc = str.maketrans("","", string.punctuation)
		review = lowerReview.translate(removePunc).split()
		for word in review:
			if word in BoW:
				bF[BoW[word]] = 1
		bin_BOW.append(bF)

	return np.array(bin_BOW)

# Bernoulli Naive Bayes
def do_BNB(train_BoW, testing_BoW, train_true_rating, testing_true_rating, param_tuning):
	
	acc_list = []
	for i in param_tuning:
		print ("i:	",i)
		clf = BernoulliNB(alpha = i)
		clf.fit(train_BoW,train_true_rating)
		pred_arr = clf.predict(testing_BoW)
		acc = sk.accuracy_score(testing_true_rating, pred_arr)
		acc_list.append(acc)
		print (acc)
	
	best_acc = np.amax(acc_list)
	best_param = param_tuning[np.argmax(acc_list)]
	print (best_acc,best_param)
	return (best_acc,best_param)


# KNN
# def do_KNN(train_BoW, testing_BoW, train_true_rating, testing_true_rating, param_tuning):

# 	acc_list = []
# 	for i in param_tuning:
# 		print ("i:	",i)
# 		KNN = KNeighborsClassifier(n_neighbors=param_tuning)
# 		KNN.fit(train_BoW,train_true_rating)
# 		KNN.predict(testing_BoW)
# 		return


if __name__ == "__main__":
	start = time.clock()

	# read / setup
	OG_train_x = pd.read_csv(r'/Users/vivek/git/A5_COMP_551/Datasets/train.csv',dtype='str', header = None)
	OG_train_y = pd.to_numeric((pd.read_csv(r'/Users/vivek/git/A5_COMP_551/Datasets/train.csv', dtype='str', header = None)).iloc[:,0]).values
	test_x = pd.read_csv(r'/Users/vivek/git/A5_COMP_551/Datasets/test.csv',dtype='str', header = None)
	test_y = pd.to_numeric((pd.read_csv(r'/Users/vivek/git/A5_COMP_551/Datasets/test.csv', dtype='str', header = None)).iloc[:,0]).values

	# for cross validation (80-20 split)
	# create train set
	train_x = OG_train_x.loc[:23999]
	train_y = OG_train_y[:24000]
	# create validation set 
	valid_x = OG_train_x.drop(train_x.index)
	valid_y = OG_train_y[24000:]

	
	dict_size = 5000 #potential hyper-param
	dictionary = create_dict(train_x,dict_size)

	train_bin_BOW = create_binaryBOW(train_x,dictionary)
	print ("created Bin Bag of Words for training ")
	print (time.clock() - start)
	
	valid_bin_BOW = create_binaryBOW(valid_x,dictionary)
	print ("created Bin Bag of Words for validation ")
	print (time.clock() - start)
	
	test_bin_BOW = create_binaryBOW(test_x,dictionary)
	print ("created Bin Bag of Words for testing ")
	print (time.clock() - start)

	print ("start param_tuning")
	param_tuning = np.linspace(1.0e-10,1,10) #not the most extensive search (takes 10 mins for 10 values)
	best_val_acc,best_param = do_BNB(train_bin_BOW, valid_bin_BOW, train_y, valid_y, param_tuning)
	print ("run on test")
	do_BNB(train_bin_BOW, test_bin_BOW, train_y, test_y, [best_param])

	# KNN
	# dictionary = create_dict(OG_train_x,dict_size)
	# train_bin_BOW = create_binaryBOW(OG_train_x,dictionary)
	# test_bin_BOW = create_binaryBOW(OG_test_x,dictionary)



	# do_Lin_SVM(train_bin_BOW, test_bin_BOW, train_y, test_y, param_tuning)
	
	print (time.clock() - start)











#!/usr/bin/evn python

"""
-------------------------------------------------------------------------------
About: Implement RIDGE optimization for Logistic Regression
		Expected acc: 92%
Author: Ruiyu Xu
Timestamp: Nov. 2014
-------------------------------------------------------------------------------
"""

import sys
import operator
import numpy as NP
from numpy.linalg import inv
import re
import math as M
import random as R
import matplotlib.pyplot as plt

ftr_no = 0
run_round = 10
iteration = 210
init_w_val = 0.0
norm_param_list = []
MEAN_INDEX = 0
SD_INDEX = 1
step = 0.01

X_list = []
y_list = []
test_X_list = []
testY_y_list = []
ds_size = 0.0
test_size = 0.0
penalty_enu = 0.01	# lambda
w0 = 0.0

def logistic_regression():
	"""
	It takes train_lines read from train data file and test_lines from test
	data file, feature number of the dataset as input arguments.
	"""
	determine_norm_param(X_list)
	norm_X_list = []
	for X in X_list:
		norm_X = norm_data(X)
		norm_X_list.append(norm_X)
	X_matrix = NP.array(norm_X_list)	
	W_matrix = stoch_GA_estimation(X_matrix,y_list)
	print "For training------->>>>"
	train_MSE,train_acc,train_rtuple_list = 	\
			predict_val(W_matrix,X_list,y_list)
	print "For testing------->>>>"
	test_MSE,test_acc,test_rtuple_list = 	\
			predict_val(W_matrix,test_X_list,test_y_list)
	return train_MSE,train_acc,train_rtuple_list,	\
			test_MSE,test_acc,test_rtuple_list

def stoch_GA_estimation(X_matrix,Y_matrix):
	"""
	Stochastic Gradient Ascent
	"""
	global w0
	W_matrix = NP.tile(init_w_val,(1,ftr_no))[0]
	w0 = init_w_val
	penalty = penalty_enu/ds_size
	for it in range(0,iteration):
		print "++++++++++++++++++++++++++iteration",it
		ds_diff_list = [(cal_h_W_Xt(W_matrix,X_matrix[t])-Y_matrix[t]) 	\
						for t in range(ds_size)]
		w0 = w0 - step*(1.0/ds_size)*sum(ds_diff_list)
		#print "The w0 is",w0
		avg_prod_sum_list = []	# len = len of W
		for j in range(0,ftr_no):
			prod_sum = sum([ds_diff_list[i]*X_matrix[i][j] 	\
						for i in range(ds_size)])
			avg_prod_sum_list.append((1.0/ds_size)*prod_sum)
		new_w_list = map(lambda w_j,aps_j:w_j-step*(aps_j+penalty*w_j),	\
					W_matrix,avg_prod_sum_list)
		W_matrix = NP.array(new_w_list)

	print "The final W matrix is:"
	print W_matrix
	return W_matrix

def cal_h_W_Xt(W_matrix,Xt):
	"""
	Calculate and return h_W(Xt)
	h_W(Xt) = 1/(1+e^-(sum(d)(wd*xd)))
	"""
	try:
		power = -(NP.dot(W_matrix,Xt)+w0)
		power = 709.0 if power >= 709.0 else power
		pred_val = 1.0/(1.0+M.pow(M.e,power))
	except:
		power = -NP.dot(W_matrix,Xt+w0)
		print "-NP.dot(w,t) is",power
		print "M.pow(e,...) is",M.pow(M.e,power)
	return pred_val

def predict_val(W_matrix,X,y):
	"""
	predicts the value by using the given W_matrix
	"""
	count = len(y)
	diff_sqr_sum = 0.0
	TP,FP,TN,FN = [0.0,0.0,0.0,0.0]
	rtup_list = []
	for i,entry in enumerate(X):
		real_val = y[i]
		data_entry = norm_data(entry)
		pred_val = cal_h_W_Xt(W_matrix,NP.array(data_entry))
		rtup_list.append((real_val,pred_val))
		pred_val = 1.0 if pred_val >= 0.5 else 0.0
		if real_val == 0.0:
			if pred_val == 0.0:
				TP += 1
			else:
				FN += 1
		else:
			if pred_val == 1.0:
				TN += 1
			else:
				FP += 1
		diff_sqr_sum += M.pow(pred_val-real_val,2)
	print "TP,FN,TN,FP ---",TP,FN,TN,FP
	MSE = diff_sqr_sum/count
	TOT = TN+FP+FN+TP
	P = TP + FN
	N = TN + FP
	err_rate = (FP+FN)/TOT
	accuracy = 1-err_rate
	print "Accuracy is",accuracy
	return MSE,accuracy,rtup_list

def get_var_list(lines):
	y_list = []
	X_list = []
	for i,line in enumerate(lines):
		line = line.strip("\n\r")
		split_line = line.split()
		split_line = [float(s) for s in split_line if s]
		label = split_line[-1]
		y_list.append(label)
		X_list.append(split_line[:-1])
	return X_list,y_list

def norm_data(data_entry):
	"""
	Z-score normalization the a data attr values by the norm_param_list.
	Returns an normalized X_matrix
	"""
	norm_attr_list = []
	for i,attr in enumerate(data_entry):
		norm_params = norm_param_list[i]
		mean = norm_params[MEAN_INDEX]
		sd = norm_params[SD_INDEX]
		norm_attr = 0 if sd == 0 else (attr-mean)/sd
		norm_attr_list.append(norm_attr)
	return norm_attr_list

def determine_norm_param(train_dataset):
	"""
	This method calculates and determines the normalization parameters from
	the given training data and then stores them into the norm_param_list:
	[(mean_0,sd_0),(mean_1,sd_1),...,(mean_56,sd_56)]
	"""
	global norm_param_list
	norm_param_list = []
	set_len = len(train_dataset)
	for i in range(0,ftr_no):
		val_list = []
		for ds in train_dataset[:]:
			val_list.append(ds[i])
		mean = sum(val_list)/float(set_len)
		sd = M.sqrt(sum([M.pow(x-mean,2) for x in val_list])/float(set_len))
		norm_param_list.append((mean,sd))
	
def run():
	global X_list,y_list,test_X_list,test_y_list,ds_size,ftr_no
	path = "spam_polluted/"
	f_train_ftr = open(path+"train_feature.txt","r")
	train_lines = f_train_ftr.readlines()
	ftr_no = len(train_lines[0].split())
	f_train_ftr.close()
	f_train_label = open(path+"train_label.txt","r")
	label_lines = f_train_label.readlines()
	f_train_label.close()
	all_train_lines = []
	for i,line in enumerate(train_lines):
		line = line.strip("\r\n")
		label = label_lines[i]
		line = line+" "+label
		all_train_lines.append(line)
	R.shuffle(all_train_lines)
	X_list,y_list = get_var_list(all_train_lines)
	ds_size = len(y_list)

	f_test_ftr = open(path+"test_feature.txt","r")
	test_lines = f_test_ftr.readlines()
	f_test_ftr.close()
	f_test_label = open(path+"test_label.txt","r")
	label_lines = f_test_label.readlines()
	f_test_label.close()
	all_test_lines = []
	for i,line in enumerate(test_lines):
		line = line.strip("\r\n")
		label = label_lines[i]
		line = line+" "+label
		all_test_lines.append(line)
	test_X_list,test_y_list = get_var_list(all_test_lines)

	train_MSE,train_acc,train_rtuple_list,test_MSE,test_acc,	\
		test_rtuple_list = logistic_regression()
	print "Training MSE,acc--------->>>>>>"
	print train_MSE,train_acc
	print "Testing MSE,acc ---------->>>>>>"
	print test_MSE,test_acc

def record_roc(f,rtuple_list,label):
	"""
	rtuple is (real_lable,predict_prob)
	"""
	sort_rtuple_list = sorted(rtuple_list,key=operator.itemgetter(1))
	total_no = len(sort_rtuple_list)
	print "total number of dataentry is:",total_no
	poscount_list = []
	TP,FP = [0.0,0.0]
	ROC_list = []
	TPR_list = []
	FPR_list = []
	for i,rtuple in enumerate(sort_rtuple_list):
		if rtuple[0] == 0.0:
			TP += 1
		else:
			FP += 1
		poscount_list.append((TP,FP))
	total_pos,total_neg = poscount_list[-1]
	for i,poscount in enumerate(poscount_list):
		TP,FP = poscount
		TPR = 0.0 if total_pos==0.0 else TP/total_pos
		FPR = 0.0 if total_neg==0.0 else FP/total_neg
		TPR_list.append(TPR)
		FPR_list.append(FPR)
		ROC_list.append((TPR,FPR))
		f.write(str(TPR)+","+str(FPR)+"\n")
	cal_AUC(ROC_list)
	plt.plot(FPR_list,TPR_list,'ro')
	plt.xlabel(label+" Data False Positive Rate")
	plt.ylabel(label+" Data True Positive Rate")
	plt.axis([0.0,1.0,0.0,1.0])
	plt.show()

def cal_AUC(ROC_list):
	auc = 0.0
	ledge,last_y = ROC_list[0]
	for d in ROC_list[:-1]:
		redge,cur_y = d
		auc += (ledge+redge)*(cur_y-last_y)/2.0
		last_y = cur_y
		ledge = redge
	print "AUC is:",auc
	
run()

#!/usr/bin/evn python

"""
-------------------------------------------------------------------------------
About: Train Logistic Regression using Gradient Ascent (and then test)
	on Spambase data.
Author: Ruiyu Xu
Timestamp: Sept. 2014
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

ftr_no = 57
run_round = 10
iteration = 30
init_w_val = 0.0
norm_param_list = []
MEAN_INDEX = 0
SD_INDEX = 1
step = 0.01

def logistic_regression(train_lines,test_lines):
	"""
	It takes train_lines read from train data file and test_lines from test
	data file, feature number of the dataset as input arguments.
	"""
	X_list,y_list = get_var_list(train_lines)
	determine_norm_param(X_list)
	norm_X_list = []
	for X in X_list:
		norm_X = norm_data(X)
		norm_X_list.append(norm_X)
	X_matrix = NP.array(norm_X_list)	
	W_matrix = stoch_GA_estimation(X_matrix,y_list)
	print "For training------->>>>"
	train_MSE,train_acc,train_rtuple_list = predict_val(W_matrix,train_lines)
	print "For testing------->>>>"
	test_MSE,test_acc,test_rtuple_list = predict_val(W_matrix,test_lines)
	return train_MSE,train_acc,train_rtuple_list,	\
			test_MSE,test_acc,test_rtuple_list

def stoch_GA_estimation(X_matrix,Y_matrix):
	"""
	Stochastic Gradient Ascent
	"""
	W_matrix = NP.tile(init_w_val,(1,ftr_no))[0]
	for it in range(0,iteration):
		print "++++++++++++++++++++++++++iteration",it
		for t,Xt in enumerate(X_matrix):
			diff_t = cal_h_W_Xt(W_matrix,Xt)-Y_matrix[t]
			new_w_list = map(lambda w_j,xt_j:w_j-(step*diff_t*xt_j),W_matrix,Xt)
			W_matrix = NP.array(new_w_list)
	print "The final W matrix is:"
	print W_matrix
	return W_matrix

def cal_h_W_Xt(W_matrix,Xt):
	"""
	Calculate and return h_W(Xt)
	h_W(Xt) = 1/(1+e^-(sum(d)(wd*xd)))
	"""
	pred_val = 1.0/(1.0+M.pow(M.e,-NP.dot(W_matrix,Xt)))
	return pred_val

def predict_val(W_matrix,lines):
	"""
	predicts the value by using the given W_matrix
	"""
	count = 0
	diff_sqr_sum = 0.0
	TP,FP,TN,FN = [0.0,0.0,0.0,0.0]
	valid_entry_list = []
	for i,line in enumerate(lines):
		line = line.strip("\n\r")
		split_line = line.split(",")
		attr_list = [float(s) for s in split_line if s]
		if len(attr_list) == 0:
			continue
		else:
			valid_entry_list.append(attr_list)
			count += 1
	rtup_list = []
	for entry in valid_entry_list:
		real_val = float(entry[-1])
		to_norm_entry = []
		to_norm_entry.extend(entry[:-1])
		data_entry = norm_data(to_norm_entry)
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
	"""
	Get X matrix and y matrix
	"""
	y_list = []
	X_list = []
	for i,line in enumerate(lines):
		line = line.strip("\n\r")
		split_line = line.split(",")
		split_line = [float(s) for s in split_line if s]
		if not len(split_line) == ftr_no + 1:
			continue
		else:
			X_i = []
			X_i.extend([float(x) for x in split_line[:-1]])
			X_list.append(X_i)
			y_list.append(float(split_line[-1]))
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
	
def run_cross_folder(data_file):
	"""
	Use corss folder method to test linear regression. It separates the given
	data file into 10 line-length-equal parts. It uses 9/10 of the dataset
	for training and the remaining 1/10 for testing.
	"""
	f_data = open(data_file,"r")
	all_lines = f_data.readlines()
	R.shuffle(all_lines)
	data_size = len(all_lines)
	last_end_no = -1
	r = 1
	f_roc_train = open("roc_train_"+str(r)+".txt","w")
	f_roc_test = open("roc_test_"+str(r)+".txt","w")
	print "-----------------------------------------------"
	train_data_lines = []
	test_data_lines = []
	start_lines_no = last_end_no+1
	percent = float(r)/run_round
	end_line_no = int(percent*data_size)
	for i,line in enumerate(all_lines):
		if i>=start_lines_no and i<=end_line_no:
			test_data_lines.append(line)
		else:
			train_data_lines.append(line)
	train_MSE,train_acc,train_rtuple_list,test_MSE,test_acc,	\
		test_rtuple_list = logistic_regression(train_data_lines,	\
							test_data_lines)
	record_roc(f_roc_train,train_rtuple_list,"Training")
	record_roc(f_roc_test,test_rtuple_list,"Testing")
	f_roc_train.close()
	f_roc_test.close()
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
	
print " Spambase data"
run_cross_folder("spambase.data")

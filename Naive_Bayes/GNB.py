#!/usr/bin/evn python

"""
-------------------------------------------------------------------------------
About: Naive Bayes with features modeled as Gaussian random variables
		Jelinek-Mercer smoothing is used for variance
		Laplace smoothing is used for avoiding 0 probability
Author: Ruiyu Xu
Timestamp: Oct. 2014
-------------------------------------------------------------------------------
"""

import sys
import operator
import numpy as NP
from numpy.linalg import inv
import re
import math as M
import random
import matplotlib.pyplot as plt

ftr_no = 57
run_round = 10
fold_size = 10
NONSPAM_MODL_INDEX = 0
SPAM_MODL_INDEX = 1
MEAN_INDEX = 0
VAR_INDEX = 1

class data_set():
	"""
	"""
	def __init__(self,x_list,y_list):
		self.x_list = x_list
		self.y_list = y_list
		self.spam_set,self.nonspam_set = self.__separate_spam_set__()
		self.total_count,self.spam_count,self.nonspam_count = 	\
			len(self.y_list),len(self.spam_set),len(self.nonspam_set)
		self.lam = float(self.total_count)/(self.total_count+2.0)
		self.P_0 = float(self.nonspam_count)/float(self.total_count)
		self.P_1 = float(self.spam_count)/float(self.total_count)
		self.__build_model__()

	def __separate_spam_set__(self):
		spam_set = []
		nonspam_set = []
		for i,x in enumerate(self.x_list):
			if self.y_list[i] == 1.0:
				spam_set.append(x)
			else:
				nonspam_set.append(x)
		return spam_set,nonspam_set

	def __build_model__(self):
		self.model = []
		for i in range(0,ftr_no):
			sval_sum,nval_sum,gen_sum = 0.0,0.0,0.0
			for j,Xj in enumerate(self.x_list):
				ftr_val = Xj[i]
				gen_sum += ftr_val
				if self.y_list[j] == 0.0:
					nval_sum += ftr_val
				else:
					sval_sum += ftr_val
			mean0 = nval_sum/self.nonspam_count
			mean1 = sval_sum/self.spam_count
			mean = gen_sum/self.total_count
			ssqr_diff_sum,nsqr_diff_sum,gen_diff_sum = 0.0,0.0,0.0
			for j,Xj in enumerate(self.x_list):
				ftr_val = Xj[i]
				gen_diff_sum += M.pow(ftr_val-mean,2)
				if self.y_list[j] == 0.0:
					nsqr_diff_sum += M.pow(ftr_val-mean0,2)
				else:
					ssqr_diff_sum += M.pow(ftr_val-mean1,2)
			var0 = nsqr_diff_sum/self.nonspam_count
			var1 = ssqr_diff_sum/self.spam_count
			var = gen_diff_sum/self.total_count
			smoothed_var0 = self.__smooth_var__(var0,var)
			smoothed_var1 = self.__smooth_var__(var1,var)
			self.model.append([(mean0,smoothed_var0),(mean1,smoothed_var1)])

	def __smooth_var__(self,foreground,background):
		"""
		Smoothing for variance
		smoothed = lambda*foreground+(1-lambda)*background
		lambda = N/(N+2) where N is the length of the whole training set
		foreground: class conditional variance
		background: overall variance
		"""
		smoothed = self.lam*foreground+(1.0-self.lam)*background
		return smoothed

	def get_model(self,attr_index):
		modellist = self.model[attr_index]
		nonspam_model = modellist[NONSPAM_MODL_INDEX]
		spam_model = modellist[SPAM_MODL_INDEX]
		return nonspam_model,spam_model

def GNB(train_lines,test_lines):
	X_list,y_list = get_var_list(train_lines)
	ds = data_set(X_list,y_list)
	print "Finally training result--------->>>"
	train_MSE,train_acc,train_rtuple_list = predict_val(ds,train_lines)
	print "Finally testing result-------->>>>"
	test_MSE,test_acc,test_rtuple_list = predict_val(ds,test_lines)
	return train_MSE,train_acc,train_rtuple_list,	\
				test_MSE,test_acc,test_rtuple_list	

def predict_val(ds,lines):
	"""
	predicts the value
	"""
	count = 0
	diff_sqr_sum = 0.0
	TP,FP,TN,FN = [0.0,0.0,0.0,0.0]
	valid_entry_list = []
	for i,line in enumerate(lines):
		line = line.strip("\n\r")
		split_line = ""
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
		data_entry = []
		data_entry.extend(entry[:-1])
		pred_val = cal_pred_val(ds,data_entry)
		rtup_list.append((real_val,pred_val))
		if real_val == 0.0:
			if pred_val == 0.0:
				TP += 1.0
			else:
				FN += 1.0
		else:
			if pred_val == 1.0:
				TN += 1.0
			else:
				FP += 1.0
		diff_sqr_sum += M.pow(pred_val-real_val,2)
	MSE = diff_sqr_sum/count
	TOT = TN+FP+FN+TP
	err_rate = (FP+FN)/TOT
	accuracy = 1-err_rate
	print "TP,FN,TN,FP ----",TP,FN,TN,FP
	print "Error rate is",err_rate
	print "Accuracy is",accuracy
	return MSE,accuracy,rtup_list

def cal_pred_val(ds,data_entry):
	"""
	Use gaussian distribution to calculate independent probability of each
	value.
	Use Laplace smoothing to avoid any possible 0 probability 
	"""
	prod_P0,prod_P1 = 1.0,1.0
	for i,Xji in enumerate(data_entry):
		nonspam_modl,spam_modl = ds.get_model(i)
		g0 = gaussian(Xji,nonspam_modl[MEAN_INDEX],nonspam_modl[VAR_INDEX])
		g1 = gaussian(Xji,spam_modl[MEAN_INDEX],spam_modl[VAR_INDEX])
		prod_P0 *= g0
		prod_P1 *= g1
	if M.isinf(prod_P1):
		prod_P1 = sys.float_info.max if Prod_P1 > 1.0 else sys.float_info.min
	gram0 = ds.P_0*prod_P0
	gram1 = ds.P_1*prod_P1
	P_0 = (gram0+1.0)/(gram0+gram1+2.0)
	P_1 = (gram1+1.0)/(gram0+gram1+2.0)
	if P_1 > P_0:
		return 1.0
	return 0.0

def gaussian(x,u,var):
	power = -M.pow(x-u,2)/(2.0*var)
	denom = M.sqrt(2.0*M.pi*var)
	g = M.pow(M.e,power)/denom
	return g

def get_var_list(lines):
	"""
	Get X matrix and y matrix
	"""
	y_list = []
	X_list = []
	for i,line in enumerate(lines):
		line = line.strip("\n\r")
		split_line = ""
		split_line = line.split(",")
		split_line = [float(s) for s in split_line if s]
		if not len(split_line) == ftr_no + 1:
			continue
		else:
			X_i = [float(x) for x in split_line[:-1]]
			X_list.append(X_i)
			y_list.append(float(split_line[-1]))
	return X_list,y_list

def run_cross_folder(data_file):
	"""
	Use corss folder method to test linear regression. It separates the given
	data file into 10 line-length-equal parts. It uses 9/10 of the dataset
	for training and the remaining 1/10 for testing.
	"""
	f_data = open(data_file,"r")
	all_lines = f_data.readlines()
	random.shuffle(all_lines)
	data_size = len(all_lines)
	f_roc_train = open("roc_train.txt","w")
	f_roc_test = open("roc_test.txt","w")
	total_train_MSE,total_train_acc,total_test_MSE,	\
		total_test_acc = [0.0,0.0,0.0,0.0]
	for r in range(1,run_round+1):
		print "-----------------------------------------------"
		print "round",r
		train_data_lines = []
		test_data_lines = []
		for i,line in enumerate(all_lines):
			if i%fold_size == r-1:
				test_data_lines.append(line)
			else:
				train_data_lines.append(line)
		train_MSE,train_acc,train_rtuple_list,test_MSE,test_acc,	\
			test_rtuple_list = GNB(train_data_lines,test_data_lines)
		print "Training MSE,acc--------->>>>>>",train_MSE,train_acc
		print "Testing MSE,acc ---------->>>>>>",test_MSE,test_acc		
		total_test_MSE += test_MSE
		total_test_acc += test_acc
		total_train_MSE += train_MSE
		total_train_acc += train_acc
		if r == 1:
			record_roc(f_roc_train,train_rtuple_list,"Training")
			record_roc(f_roc_test,test_rtuple_list,"Testing")
			f_roc_train.close()
			f_roc_test.close()
	print "Average training MSE:",total_train_MSE/run_round
	print "Average training acc:",total_train_acc/run_round
	print "Average training Error Rate:",1.0-total_train_acc/run_round
	print "Average testing MSE:",total_test_MSE/run_round
	print "Average testing acc:",total_test_acc/run_round
	print "Average testing Error Rate:",1.0-total_test_acc/run_round

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

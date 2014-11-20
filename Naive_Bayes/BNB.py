#!/usr/bin/evn python

"""
-------------------------------------------------------------------------------
About: Naive Bayes with ffeatures modeled as simple Bernoulli (Boolean)
		random variables.
		Laplace smoothing is used to avoid 0 probability
        Background-Foreground smoothing is used for variance
Author: Ruiyu Xu
Time: Oct. 2014
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
fold_size = 10
stats_filename = "spam_stat.txt"
MEAN0_INDEX = 0
MEAN1_INDEX = 1
MEAN_INDEX = 2
LE_INDEX = 0
G_INDEX = 1
NONSPAM_MODL_INDEX = 0
SPAM_MODL_INDEX = 1

class data_set():
	"""
	"""
	def __init__(self,x_list,y_list):
		self.x_list = x_list
		self.y_list = y_list
		self.spam_set,self.nonspam_set = self.__separate_spam_set__()
		self.total_count,self.spam_count,self.nonspam_count = 	\
			len(self.y_list),len(self.spam_set),len(self.nonspam_set)
		self.P_0 = float(len(self.nonspam_set))/float(len(self.y_list))
		self.P_1 = float(len(self.spam_set))/float(len(self.y_list))
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
		self.__cal_means__()	
		self.model = []
		for i in range(0,ftr_no):
			spam_le_no,spam_g_no,nonspam_le_no,nonspam_g_no = NP.tile(0.0,4)
			mean = self.mean_list[i]
			for j,Xj in enumerate(self.x_list):
				ftr_val = Xj[i]
				if ftr_val <= mean:
					if self.y_list[j] == 0.0:
						nonspam_le_no += 1
					else:
						spam_le_no += 1
				else:
					if self.y_list[j] == 0.0:
						nonspam_g_no += 1
					else:
						spam_g_no += 1
			spam_no = (spam_le_no,spam_g_no)
			nonspam_no = (nonspam_le_no,nonspam_g_no)
			spam_probs = self.__smooth_probs__(spam_no,self.spam_count)
			nonspam_probs = self.__smooth_probs__(nonspam_no,self.nonspam_count)
			self.model.append((nonspam_probs,spam_probs))

	def __cal_means__(self):
		self.mean_list = []
		for i in range(0,ftr_no):
			total = 0.0
			for j,Xj in enumerate(self.x_list):
				total += Xj[i]
			mean = total/(float(self.total_count))
			self.mean_list.append(mean)

	def __smooth_probs__(self,cond_no,data_count):
		"""
		Smooth the probability with Laplace smoothing method
		"""
		cond_s_count = NP.array(cond_no)+1
		data_s_count = NP.array(data_count)+2
		smoothed_probs = cond_s_count/data_s_count
		return smoothed_probs

	def get_val_probs(self,val,attr_index):
		probs_list = self.model[attr_index]
		prob_index = self.get_prob_index(val,attr_index)
		prob0 = probs_list[NONSPAM_MODL_INDEX][prob_index]
		prob1 = probs_list[SPAM_MODL_INDEX][prob_index]
		return prob0,prob1

	def get_prob_index(self,val,attr_index):
		"""
		Given a value and its attribute index, check which probability
		ie. le or g, it should use. Return the index
		"""
		mean = self.mean_list[attr_index]
		if val <= mean:
			return LE_INDEX
		else:
			return G_INDEX

def NB(train_lines,test_lines):
	"""
	Naive Bayes with Bernoulli
	"""
	x_list,y_list = get_var_list(train_lines)
	ds = data_set(x_list,y_list)
	train_MSE,train_acc,train_rtuple_list = predict_val(ds,train_lines)
	test_MSE,test_acc,test_rtuple_list = predict_val(ds,test_lines)
	return train_MSE,train_acc,train_rtuple_list,	\
			test_MSE,test_acc,test_rtuple_list

def get_var_list(lines):
	"""
	Get X list and y list
	"""
	y_list = []
	x_list = []
	for i,line in enumerate(lines):
		line = line.strip("\n\r")
		split_line = [float(s) for s in line.split(",") if s]
		if not len(split_line) == ftr_no + 1:
			continue
		else:
			x_list.append([float(x) for x in split_line[:-1]])
			y_list.append(float(split_line[-1]))
	return x_list,y_list

def predict_val(ds,lines):
	"""
	predicts the value by using the given dataset.
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
				TN += 1.0
			else:
				FP += 1.0
		else:
			if pred_val == 1.0:
				TP += 1.0
			else:
				FN += 1.0
		diff_sqr_sum += M.pow(pred_val-real_val,2)
	MSE = diff_sqr_sum/count
	TOT = TN+FP+FN+TP
	err_rate = (FP+FN)/TOT
	accuracy = 1-err_rate
	print "TP,FN,TN,FP ----",TP,FN,TN,FP
	print "ERror rate is",err_rate
	print "Accuracy is",accuracy
	return MSE,accuracy,rtup_list

def cal_pred_val(ds,Xj):
	"""
	P(Y=yk|X1,...,Xn) = P(Y=yk)TTiP(Xi|Y=yk)/SUMjP(Y=yj)TTjP(Xi|Y=yj)	
	"""
	prod_P0,prod_P1 = 1.0,1.0
	for i,Xji in enumerate(Xj):
		prob0,prob1 = ds.get_val_probs(Xji,i)
		prod_P0 *= prob0
		prod_P1 *= prob1
	enum0 = prod_P0 * ds.P_0
	enum1 = prod_P1 * ds.P_1
	denom = enum0+enum1
	pred_P0 = enum0/denom
	pred_P1 = enum1/denom
	if pred_P1 > pred_P0:
		return 1.0
	return 0.0

def run_cross_folder(data_file):
	f_data = open(data_file,"r")
	all_lines = f_data.readlines()
	R.shuffle(all_lines)
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
			test_rtuple_list = NB(train_data_lines,test_data_lines)
		print "Training mse,acc --------->>>>>>",train_MSE,train_acc
		print "Testing mse,acc ---------->>>>>>",test_MSE,test_acc
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

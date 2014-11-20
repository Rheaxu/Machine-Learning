#!/usr/bin/evn python

"""
-------------------------------------------------------------------------------
About: Naive Bayes with features modeled via histogram
		4 bins:
		[min-value,low-mean-value]
		(low-mean-value,overall-mean-value]
		(overall-mean-value,high-mean-value]
		(high-mean-value,max-value]
		9 bins:
		Equally split the [min_val,max_val] interval to 9 bins
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
MIN_INDEX = 0
LM_INDEX = 1
OM_INDEX = 2
HM_INDEX = 3
MAX_INDEX = 4
NONSPAM_MODL_INDEX = 0
SPAM_MODL_INDEX = 1
BINS4 = 4
BINS9 = 9

class data_set():
	def __init__(self,x_list,y_list,bins_no):
		self.x_list = x_list
		self.y_list = y_list
		self.spam_set,self.nonspam_set = self.__separate_spam_set__()
		self.total_count,self.spam_count,self.nonspam_count = 	\
					len(self.y_list),len(self.spam_set),len(self.nonspam_set)
		self.P_0 = float(len(self.nonspam_set))/float(len(self.y_list))
		self.P_1 = float(len(self.spam_set))/float(len(self.y_list))
		self.bins_no = bins_no
		self.__build_bins_model__()

	def __separate_spam_set__(self):
		spam_set = []
		nonspam_set = []
		for i,x in enumerate(self.x_list):
			if self.y_list[i] == 1.0:
				spam_set.append(x)
			else:
				nonspam_set.append(x)
		return spam_set,nonspam_set

	def __build_bins_model__(self):
		if self.bins_no == BINS4:
			self.bins_list = self.__set_4bins__()
		else:
			self.bins_list = self.__set_9bins__()
		self.model = []
		for i in range(0,ftr_no):
			spam_bins_count = NP.tile(0.0,self.bins_no)
			nonspam_bins_count = NP.tile(0.0,self.bins_no)
			for j,Xj in enumerate(self.x_list):
				ftr_val = Xj[i]
				bin_index = self.get_bin_index(ftr_val,i)
				if self.y_list[j] == 0.0:
					nonspam_bins_count[bin_index] += 1
				else:
					spam_bins_count[bin_index] += 1
			spam_probs = 	\
				self.__bins_smoothed_probs__(spam_bins_count,self.spam_count)
			nonspam_probs = self.__bins_smoothed_probs__(	\
					nonspam_bins_count,self.nonspam_count)
			self.model.append((nonspam_probs,spam_probs))

	def __bins_smoothed_probs__(self,bins_count,data_count):
		"""
		Smooth the probability with Laplace smoothing method
		"""
		bin_s_count = bins_count+1
		data_s_count = data_count+2
		smoothed_probs = bin_s_count/data_s_count
		return smoothed_probs

	def __set_4bins__(self):
		bins_list = []
		for i in range(0,ftr_no):
			first_val =self.x_list[0][i] 
			min_value,max_value,overall_sum = first_val,first_val,first_val
			spam_sum,nonspam_sum = 0.0,0.0
			if self.y_list[0] == 0.0:
				nonspam_sum += first_val
			else:
				spam_sum += first_val
			for j,Xj in enumerate(self.x_list[1:]):
				ftr_val = Xj[i]
				min_value = ftr_val if ftr_val < min_value else min_value
				max_value = ftr_val if ftr_val > max_value else max_value
				overall_sum += ftr_val
				if self.y_list[j+1] == 0.0:
					nonspam_sum += ftr_val
				else:
					spam_sum += ftr_val
			nonspam_mean = nonspam_sum/float(self.nonspam_count)
			spam_mean = spam_sum/float(self.spam_count)
			overall_mean = overall_sum/float(self.total_count)
			low_mean,high_mean = [nonspam_mean,spam_mean] if 	\
				nonspam_mean < spam_mean else [spam_mean,nonspam_mean]
			bins_list.append((min_value,low_mean,overall_mean,	\
							high_mean,max_value))
		return bins_list
	
	def __set_9bins__(self):
		bins_list = []
		for i in range(0,ftr_no):
			first_val = self.x_list[0][i]
			min_value,max_value = first_val,first_val
			spam_sum,nonspam_sum = 0.0,0.0
			if self.y_list[0] == 0.0:
				nonspam_sum += first_val
			else:
				spam_sum += first_val
			for j,Xj in enumerate(self.x_list[1:]):
				ftr_val = Xj[i]
				min_value = ftr_val if ftr_val < min_value else min_value
				max_value = ftr_val if ftr_val > max_value else max_value
			interval = (min_value - max_value)/9
			bins = []
			for j in range(0,self.bins_no+1):
				bins.append(min_value+j*interval)
			bins_list.append(bins)
		return bins_list
		
	def get_val_probs(self,val,attr_index):
		bins_probs = self.model[attr_index]
		bin_index = self.get_bin_index(val,attr_index)
		prob0 = bins_probs[NONSPAM_MODL_INDEX][bin_index]
		prob1 = bins_probs[SPAM_MODL_INDEX][bin_index]
		return prob0,prob1

	def get_bin_index(self,val,attr_index):
		"""
		Given a values and its attribute index, check which bin does it belong
		to. Return the index of the bin
		"""
		bins = self.bins_list[attr_index]
		lower_bound = bins[0]
		if val == lower_bound:
			return 0
		for i,higher_bound in enumerate(bins[1:]):
			if val > lower_bound and val <= higher_bound:
				return i
			else:
				lower_bound = higher_bound
		return self.bins_no-1

def HNB(train_lines,test_lines,bins_no):
	X_list,y_list = get_var_list(train_lines)
	ds = data_set(X_list,y_list,bins_no)
	print "Finally training result--------->>>"
	train_MSE,train_acc,train_rtuple_list = predict_val(ds,train_lines)
	print "Finally testing result-------->>>>"
	test_MSE,test_acc,test_rtuple_list = predict_val(ds,test_lines)
	return train_MSE,train_acc,train_rtuple_list,	\
				test_MSE,test_acc,test_rtuple_list	

def predict_val(ds,lines):
	"""
	predicts the value by using the given params
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
	print diff_sqr_sum
	MSE = diff_sqr_sum/count
	TOT = TN+FP+FN+TP
	err_rate = (FP+FN)/TOT
	accuracy = 1-err_rate
	print "TP,FN,TN,FP ----",TP,FN,TN,FP
	print "Error rate is",err_rate
	print "Accuracy is",accuracy
	return MSE,accuracy,rtup_list

def cal_pred_val(ds,data_entry):
	prod_P0,prod_P1 = 1.0,1.0
	for i,Xji in enumerate(data_entry):
		prob0,prob1 = ds.get_val_probs(Xji,i)		
		prod_P0 *= prob0
		prod_P1 *= prob1
	enum0 = prod_P0 * ds.P_0
	enum1 = prod_P1 * ds.P_1
	denom = enum0+enum1
	P_0 = enum0/denom
	P_1 = enum1/denom
	if P_1 > P_0:
		return 1.0
	return 0.0

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

def run_cross_folder(data_file,bins_no):
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
			test_rtuple_list = HNB(train_data_lines,test_data_lines,bins_no)
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
run_cross_folder("spambase.data",BINS9)

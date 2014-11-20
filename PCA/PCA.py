#!/usr/bin/evn python

"""
-------------------------------------------------------------------------------
About: Run PCA to get 100 features
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
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

ftr_no = 0
run_round = 10
fold_size = 10
NONSPAM_MODL_INDEX = 0
SPAM_MODL_INDEX = 1
MEAN_INDEX = 0
VAR_INDEX = 1
comp_no = 100

X_list = []
y_list = []
test_X_list = []
test_y_list = []
ds_size = 0.0
test_size = 0.0

class data_set():
	"""
	"""
	def __init__(self,x_list,y_list):
		self.x_list = x_list
		self.y_list = y_list
		self.spam_set,self.nonspam_set = self.__separate_spam_set__()
		self.total_count,self.spam_count,self.nonspam_count = 	\
			len(self.y_list),len(self.spam_set),len(self.nonspam_set)
		self.lam = 0.999999
		#######self.lam = float(self.total_count)/(self.total_count+2.0)
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
		return NP.matrix(spam_set),NP.matrix(nonspam_set)

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
		print "The length of model is",len(self.model)

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

def GNB():
	ds = data_set(X_list,y_list)
	print "Finally training result--------->>>"
	train_MSE,train_acc,train_rtuple_list = predict_val(ds,X_list,y_list)
	print "Finally testing result-------->>>>"
	test_MSE,test_acc,test_rtuple_list = predict_val(ds,test_X_list,test_y_list)
	return train_MSE,train_acc,train_rtuple_list,	\
				test_MSE,test_acc,test_rtuple_list	

def predict_val(ds,X,y):
	"""
	predicts the value
	"""
	count = len(y)
	diff_sqr_sum = 0.0
	TP,FP,TN,FN = [0.0,0.0,0.0,0.0]
	rtup_list = []
	for i,entry in enumerate(X):
		real_val = y[i]
		pred_val = cal_pred_val(ds,entry)
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
	err_rate = (FP+FN)/count
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
		prod_P1 = sys.float_info.max if prod_P1 > 1.0 else sys.float_info.min
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
	g = M.exp(power)/denom
	return g

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

def run_cross_folder():
	"""
	Use corss folder method to test linear regression. It separates the given
	data file into 10 line-length-equal parts. It uses 9/10 of the dataset
	for training and the remaining 1/10 for testing.
	"""
	global X_list,y_list,test_X_list,test_y_list,ds_size,test_size,ftr_no
	path = "spam_polluted/"
	f_train_ftr = open(path+"train_feature.txt","r")
	train_lines = f_train_ftr.readlines()
	ftr_no = comp_no
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
	random.shuffle(all_train_lines)
	all_X_list,y_list = get_var_list(all_train_lines)
	ds_size = len(y_list)

	f_test_ftr = open(path+"test_feature.txt","r")
	test_lines = f_test_ftr.readlines()
	f_test_ftr.close()
	f_test_label = open(path+"test_label.txt","r")
	test_label_lines = f_test_label.readlines()
	f_test_label.close()
	all_test_lines = []
	for i,line in enumerate(test_lines):
		line = line.strip("\r\n")
		label = test_label_lines[i]
		line = line+" "+label
		all_test_lines.append(line)
	all_test_X_list,test_y_list = get_var_list(all_test_lines)
	test_size = len(test_y_list)

	X_list,test_X_list = apply_pca(all_X_list,all_test_X_list)

	train_MSE,train_acc,train_rtuple_list,test_MSE,test_acc,	\
		test_rtuple_list = GNB()
	print "Training MSE,acc--------->>>>>>",train_MSE,train_acc
	print "Testing MSE,acc ---------->>>>>>",test_MSE,test_acc		

def apply_pca(all_X_list,all_test_X_list):
	pca = PCA(n_components = comp_no)
	all_X = NP.array(all_X_list + all_test_X_list)
	pca.fit(all_X)
	transformed_X = pca.transform(all_X)
	train_X = transformed_X[:ds_size]
	
	test_X = transformed_X[ds_size:]
	return train_X,test_X

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

run_cross_folder()

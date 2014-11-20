#!/usr/bin/env python

"""
-------------------------------------------------------------------------------
About: Active Learning
Author: Ruiyu Xu
Time: Oct. 2014
-------------------------------------------------------------------------------
"""

import numpy as NP
import math as M
import random
import operator
import matplotlib.pyplot as plt
import pylab as pl
import time

ftr_no = 57
T = 20
fold_size = 10
CHOOSE_OPT = "Opt_stump"
CHOOSE_RAND = "Rand_stump"
all_X_list = []
all_y_list = []
X_list = []
y_list = []
remain_X_list = []
remain_y_list = []
test_X_list = []
test_y_list = []
to_add_index_list = []
episode_add_line_no = 0.0
ds_size = 0.0
total_ds_size = 0.0
opt_err_list = []
rand_err_list = []

def Adaboost(boost_type,plot_flag):
	"""
	Do adaboost on the given weak learner WL which contains several
	predictors.
	"""
	global D,ds_size
	WL = Weak_Learner(boost_type)
	ds_size = float(len(y_list))
	D = init_D(ds_size)
	round_err_list,train_err_list,test_err_list,auc_list = [[],[],[],[]]
	train_round_result = NP.zeros(ds_size)
	remain_train_round_result = NP.zeros(len(remain_y_list))
	test_round_result = NP.zeros(len(test_y_list))
	test_err = 0.0
	t = 1
	for t in range(1,T+1):
		ipslon_t,best_stump = WL.get_ht_min_werr(D,boost_type,t)
		round_err_list.append(ipslon_t)
		alpha = 0.5*M.log((1.0-ipslon_t)/ipslon_t,M.e)
		D = update_D(D,alpha,best_stump,ipslon_t)
		train_round_result += cal_round_result(alpha,best_stump.pred_vals)		
		remain_train_round_result += cal_remain_round_result(alpha,best_stump)
		test_round_result += cal_test_round_result(alpha,best_stump)
		if t == T:
			get_to_add_data(boost_type,remain_train_round_result)
		train_err,train_rtuple_list = 	\
						predict(best_stump,train_round_result,y_list)
		train_err_list.append(train_err)
		test_err,test_rtuple_list = 	\
					predict(best_stump,test_round_result,test_y_list)
		TRP_list,FPR_list,auc = record_roc(test_rtuple_list)
		train_acc = 1.0-train_err
		test_acc = 1.0-test_err
		print "Round:",t,"Ftr:",best_stump.ftr_index+1,"Threshold:",	\
			best_stump.threshold,"Round_err:",ipslon_t,"Train_acc:",	\
			train_acc,"Test_acc:",test_acc,"AUC:",auc
	return test_err

def init_D(dataset_size):
	"""
	Initialize the data point distribution.
	The initial values are 1/|data_set|
	"""
	val = 1.0/float(dataset_size)
	D = NP.tile(val,dataset_size)
	return D

def cal_accum_hyp(round_result):
	finalH = NP.sign(round_result)
	return finalH

def update_D(D,alpha,best_stump,ipslon_t):
	new_D = []
	ht = best_stump.pred_vals
	for i,Di in enumerate(D):
		power = -alpha*y_list[i]*ht[i]
		exp = M.exp(power)
		new_Di = Di*exp
		new_D.append(new_Di)
	newD_sum = sum(new_D)
	normed_D = NP.array(new_D)/newD_sum
	return normed_D

def get_to_add_data(boost_type,remain_train_result):
	global X_list,y_list,remain_X_list,remain_y_list
	candits = []
	if boost_type == CHOOSE_OPT:
		for i,result in enumerate(remain_train_result):
			candits.append((abs(result),remain_X_list[i],remain_y_list[i]))
		sort_candits = sorted(candits,key=operator.itemgetter(0))
		for i in range(0,episode_add_line_no):
			X_list.append(candits[i][1])
			y_list.append(candits[i][2])
		for i in range(episode_add_line_no,len(remain_y_list)):
			remain_X_list.append(candits[i][1])
			remain_y_list.append(candits[i][2])
	else:
		for i in range(0,episode_add_line_no):
			X_list.append(remain_X_list[i])
			y_list.append(remain_y_list[i])
		remian_X_list = remain_X_list[episode_add_line_no:]
		remian_y_list = remain_y_list[episode_add_line_no:]

def cal_round_result(alpha,hx):
	rr = []
	for hi in hx:
		rr.append(alpha*hi)
	return NP.array(rr)

def cal_remain_round_result(alpha,best_stump):
	hx = []
	for X in remain_X_list:
		ftr = X[best_stump.ftr_index]
		hx.append(1.0 if ftr > best_stump.threshold else -1.0)
	return cal_round_result(alpha,hx)

def cal_test_round_result(alpha,best_stump):
	hx = []
	for X in test_X_list:
		ftr = X[best_stump.ftr_index]
		hx.append(1.0 if ftr > best_stump.threshold else -1.0)
	return cal_round_result(alpha,hx)

def predict(best_stump,round_result,y):
	pred_vals = cal_accum_hyp(round_result)
	TP,FP,TN,FN = [0.0,0.0,0.0,0.0]
	rtuple_list = []
	for i,pred in enumerate(pred_vals):
		real = y[i]
		rtuple_list.append((real,pred))
		if real == -1.0:
			if pred == -1.0:
				TP += 1.0
			else:
				FN += 1.0
		else:
			if pred == -1.0:
				FP += 1.0
			else:
				TN += 1.0
	train_err = (FP+FN)/ds_size
	return train_err,rtuple_list

def record_roc(rtuple_list):
	"""
	rtuple is (real_lable,predict_prob)
	"""
	sort_rtuple_list = sorted(rtuple_list,key=operator.itemgetter(1))
	total_no = len(sort_rtuple_list)
	poscount_list = []
	TP,FP = [0.0,0.0]
	ROC_list = []
	TPR_list = []
	FPR_list = []
	for i,rtuple in enumerate(sort_rtuple_list):
		if rtuple[0] == -1.0:
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
	return TPR_list,FPR_list,cal_AUC(ROC_list)

def cal_AUC(ROC_list):
	auc = 0.0
	ledge,last_y = ROC_list[0]
	for d in ROC_list[:-1]:
		redge,cur_y = d
		auc += (ledge+redge)*(cur_y-last_y)/2.0
		last_y = cur_y
		ledge = redge
	return auc

def plot():
	pl.plot(opt_err_list,label="Add data by boosting score")
	pl.plot(rand_err_list,label="Add data randomly")
	pl.legend(loc="upper right")
	pl.xlabel("train_data_percent")
	pl.ylabel("test_error")
	pl.show()

def plot_helper(data_list,line_name,xlabel,ylabel):
	for i,data in enumerate(data_list):
		pl.plot(data,label=line_name[i])
	pl.legend(loc="upper left")
	pl.xlabel(xlabel)
	pl.ylabel(ylabel)
	pl.show()

def draw_test_roc(TPR_list,FPR_list):
	plt.plot(FPR_list,TPR_list,'ro')
	plt.xlabel("Testing Data False Positive Rate")
	plt.ylabel("Testing Data True Positive Rate")
	plt.axis([0.0,1.0,0.0,1.0])
	plt.show()

class Weak_Learner():
	"""
	This weak learner returns the "best" decision stump with respect to the
	weighted training set given
	"""
	def __init__(self,predictor_type):
		self.__predictors__ = self.__construct_predictors__(predictor_type)

	def __construct_predictors__(self,predictor_type):
		predictors = []
		for i in range(0,ftr_no):
			ftr_vals = []
			for X in X_list:
				ftr_vals.append(X[i])
			predictors.append(Decision_Stumps(ftr_vals,i,predictor_type))
		return predictors

	def get_ht_min_werr(self,D,predictor_type,t):
		"""
		In each round, boosting provides a weighted data set to the weak
		learner, and the weak learner provides the best decision stump
		with respect to the weighted training set.
		"""
		if predictor_type == CHOOSE_OPT:
			return self.choose_best_predictor(D,t)
		else:
			return self.choose_rand_predictor(D)

	def choose_best_predictor(self,D,t):
		"""
		This methods get the optimal decision stump for each feature (57 opt
		decision stumps), and compare them, then return the best one among the
		57 ones.
		Return: the hypothesis and index list of wrong hyp from the best
				predictor
		"""
		farest_err,best_stump = 	\
				self.__predictors__[0].get_best_stump(D)
		for p in self.__predictors__[1:]:
			err,cur_stump = p.get_best_stump(D)
			if abs(0.5-err)>abs(0.5-farest_err):
				farest_err = err
				best_stump = cur_stump
		return farest_err,best_stump

	def choose_rand_predictor(self,D):
		"""
		Randomly choose a predictor
		Return: the hypothesis from the random predictor
		"""
		rand_ftr_no = random.randrange(ftr_no)
		stump_list = self.__predictors__[rand_ftr_no]
		return stump_list.get_rand_stump(D)

class Decision_Stumps():
	"""
	This is for one feature
	It contains decision stumps created by all possible thresholds
	"""

	def __init__(self,ftr_vals,ftr_index,predictor_type):
		"""
		The predictors contains 57 stump lists, where each list is for a
		feature.
		A list contains as many stumps as thresholds that a feature has.
		"""
		self.ftr_index = ftr_index
		self.stumps = self.__create_stumps__(ftr_index,ftr_vals)
		self.predictor_type = predictor_type
		self.stumps_no = len(self.stumps)

	def __create_stumps__(self,ftr_index,ftr_vals):
		"""
		To create the various thresholds for the given feature fi:
		1. sort the values of fi
		2. remove duplicate values
		3. construct thresholds that are midway between successive feature
			values.
		4. add two thresholds for fi: one below all values for the fi
			and one above all values for fi
		"""
		max_val = ftr_vals[0]
		min_val = ftr_vals[0]
		sorted_vals = sorted(ftr_vals)
		last_val = sorted_vals[0]
		candit_t = [last_val]
		for val in sorted_vals[1:]:
			if not val == last_val:
				max_val = val if val > max_val else max_val
				min_val = val if val < min_val else min_val
				candit_t.append(val)
				last_val = val
		stumps = map(lambda x,y:stump(ftr_index,(x+y)/2.0),candit_t[:-1],	\
				candit_t[1:])
		allstumps = [stump(ftr_index,min_val-0.1)]
		allstumps.extend(stumps)
		allstumps.append(stump(ftr_index,max_val+0.1))
		return allstumps

	def get_best_stump(self,D):
		"""
		Get the best stump for from the stump list of a feature
		"""
		best_stump = self.stumps[0]
		farest_err = best_stump.cal_werr(D)
		for s in self.stumps[1:]:
			err = s.cal_werr(D)
			if abs(0.5-err) > abs(0.5-farest_err):
				farest_err = err
				best_stump = s
		return farest_err,best_stump

	def get_rand_stump(self,D):
		rand_index = random.randrange(self.stumps_no)
		cur_stump = self.stumps[rand_index]
		err = cur_stump.cal_werr(D)
		return err,cur_stump

class stump():
	"""
	"""
	def __init__(self,ftr_index,threshold):
		self.ftr_index = ftr_index
		self.threshold = threshold
		self.pred_vals,self.wrong_indices = self.__predict__()

	def __predict__(self):
		pred_vals = []
		wrong_indices = []
		for i,X in enumerate(X_list):
			ftr = X[self.ftr_index]
			pred_val = 1.0 if ftr > self.threshold else -1.0
			pred_vals.append(pred_val)
			if not pred_val == y_list[i]:
				wrong_indices.append(i)
		return pred_vals,wrong_indices

	def cal_werr(self,D):
		ipslon = 0.0
		for ind in self.wrong_indices:
			ipslon += D[ind]
		return ipslon

def get_var_list(lines):
	"""
	Get X matrix and y matrix
	"""
	X_list = []
	y_list = []
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
			y_list.append(1.0 if split_line[-1] == 1.0 else -1.0)
	return X_list, y_list

def run(data_file,boost_type):
	global all_X_list,all_y_list,X_list,y_list,test_X_list,remain_X_list,	\
			remain_y_list,test_y_list,episode_add_line_no,total_ds_size,	\
			opt_err_list,rand_err_list
	f_data = open(data_file,"r")
	all_lines = f_data.readlines()
	random.shuffle(all_lines)
	total_ds_size = len(all_lines)
	plot_flag = False
	start_trainlines_no =int(0.05*total_ds_size)
	test_endindex = int(0.1*total_ds_size)	#10% test data
	all_trainlines = all_lines[test_endindex:]
	test_lines = all_lines[:test_endindex]
	all_X_list,all_y_list = get_var_list(all_trainlines)
	test_X_list,test_y_list = get_var_list(test_lines)
	episode_add_line_no = int(0.05*total_ds_size)
	train_per = 0.05
	init_train_lines_no = int(0.05*total_ds_size)
	X_list = all_X_list[:init_train_lines_no]
	y_list = all_y_list[:init_train_lines_no]
	remain_X_list = all_X_list[init_train_lines_no:]
	remain_y_list = all_y_list[init_train_lines_no:]
	while True:
		start_tick = time.time()
		print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run Episode "+	\
				str(train_per*100)+"% training data"
		test_err = Adaboost(boost_type,plot_flag)
		if boost_type == CHOOSE_OPT:
			opt_err_list.append(test_err)
		else:
			rand_err_list.append(test_err)
		if train_per >= 0.5:
			break
		train_per += 0.05
		plot_flag = False
		end_tick = time.time()

run("spambase.data",CHOOSE_OPT)
print opt_err_list
run("spambase.data",CHOOSE_RAND)
print rand_err_list
plot()


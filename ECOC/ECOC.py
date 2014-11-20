#!/usr/bin/env python

"""
-------------------------------------------------------------------------------
About: ECOC (Error Correcting Output Codes)
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
import itertools

ftr_no = 1754
T = 200
fold_size = 10
CHOOSE_OPT = "Opt_stump"
CHOOSE_RAND = "Rand_stump"
X_list = []
orig_y_list = []
orig_y_invert_list = []
y_list = []
invert_y_list = []
train_ftr_vals_list = []
ftrs_threshold_list = []		# The stumps list
TFPNlist = []		# [TPlist,FPlist,TNlist,FNlist] for each stump
TPINDEX,FPINDEX,TNINDEX,FNINDEX = [0,1,2,3]
SWIL = []				# stump wrong indices list
test_X_list = []
test_y_list = []				# Changes for each ECOC func
k = 8
func_no = 20
class_list = [0,1,2,3,4,5,6,7]
func_list  =[]
code_list = []
ds_size = 0
test_size = 0
train_result_list = []
test_result_list = []

def Adaboost(boost_type,plot_flag):
	"""
	Do adaboost on the given weak learner WL which contains several
	predictors.
	"""
	global D,train_result_list,test_result_list,	\
			train_result_list,test_result_list
	WL = Weak_Learner(boost_type)
	D = init_D(ds_size)
	round_err_list,train_err_list,test_err_list = [[],[],[]]
	train_round_result = NP.zeros(ds_size)
	test_round_result = NP.zeros(len(test_y_list))
	t = 1
	for t in range(1,T+1):
		if t%5 == 0:
			print "================Running for round",t
		ipslon_t,best_stump = WL.get_ht_min_werr(D,boost_type,t)
		"""
		The best_stump contains (ftr_index,threshold_index)
		"""
		best_ftr_index,threshold_index = best_stump
		round_err_list.append(ipslon_t)
		alpha = 0.5*M.log((1.0-ipslon_t)/ipslon_t,M.e)
		D = update_D(D,alpha,best_stump,ipslon_t)
		train_round_result += cal_train_round_result(alpha,best_stump)
		test_round_result += cal_test_round_result(alpha,best_stump)
	final_test_result = NP.sign(test_round_result)
	return final_test_result

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
	"""
	best_stump = [best_ftr_index,threshold_index]
	"""
	new_D = D[:]
	ftr_index = best_stump[0]
	threshold_index = best_stump[1]
	stump_TFPNlist = TFPNlist[ftr_index][threshold_index]
	FPlist = stump_TFPNlist[FPINDEX]
	FNlist = stump_TFPNlist[FNINDEX]
	TPlist = stump_TFPNlist[TPINDEX]
	TNlist = stump_TFPNlist[TNINDEX]
	wrong_exp = M.exp(alpha)
	right_exp = M.exp(-alpha)
	for ind in FPlist:
		new_D[ind] *= wrong_exp
	for ind in FNlist:
		new_D[ind] *= wrong_exp
	for ind in TPlist:
		new_D[ind] *= right_exp
	for ind in TNlist:
		new_D[ind] *= right_exp
	newD_sum = sum(new_D)
	normed_D = NP.array(new_D)/newD_sum
	return normed_D

def cal_train_round_result(alpha,best_stump):
	TPlist,FPlist,TNlist,FNlist = TFPNlist[best_stump[0]][best_stump[1]]
	rr = [alpha]*ds_size
	for i in TPlist:
		rr[i] *= y_list[i]
	for i in FPlist:
		rr[i] *= -y_list[i]
	for i in TNlist:
		rr[i] *= y_list[i]
	for i in FNlist:
		rr[i] *= -y_list[i]
	return NP.array(rr)
		

def cal_round_result(alpha,hx):
	rr = []
	for hi in hx:
		rr.append(alpha*hi)
	return NP.array(rr)

def cal_test_round_result(alpha,best_stump):
	hx = []
	ftr_index = best_stump[0]
	threshold_index = best_stump[1]
	threshold = ftrs_threshold_list[ftr_index][threshold_index]
	for X in test_X_list:
		val = 0.0
		for ind_val in X:
			if ind_val[0] == ftr_index:
				val = ind_val[1]
				break
		hx.append(1.0 if val > threshold else -1.0)
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
	return pred_vals,train_err,rtuple_list

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

def plot(round_err_list,train_err_list,test_err_list,auc_list):
	plot_helper([round_err_list],["Round error"],"Round number","Round error")
	plot_helper([train_err_list,test_err_list],	\
				["train error","test error"],	\
				"Round number","Train and Test error",)
	plot_helper([auc_list],["AUC"],"Round number","AUC")

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
		"""
		"""

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
		Choose the best feature
		"""
		farest_err,best_threshold_index = self.choose_best_stump(D,0)
		best_stump = [0,best_threshold_index]
		time1 = time.time()
		for i in range(1,ftr_no):
			err,cur_threshold_index = self.choose_best_stump(D,i)
			cur_stump = [i,cur_threshold_index]
			if abs(0.5-err)>abs(0.5-farest_err):
				farest_err = err
				best_stump = cur_stump
		time2 = time.time()
		return farest_err,best_stump
				
	def choose_rand_predictor(self,D,ftr_index):
		"""
		Randomly choose a predictor
		Return: the hypothesis from the random predictor
		"""
		rand_ftr_no = random.randrange(ftr_no)
		stump_list = self.__predictors__[rand_ftr_no]
		return stump_list.get_rand_stump(D)

	def choose_best_stump(self,D,ftr_index):
		"""
		This method gets the best stump for a feature
		"""
		farest_err = self.cal_stump_err(D,ftr_index,0)
		best_threshold_index = 0
		threshold_list_len = len(ftrs_threshold_list[ftr_index])
		for i in range(1,threshold_list_len):
			err = self.cal_stump_err(D,ftr_index,i)
			if abs(0.5-err) > abs(0.5-farest_err):
				farest_err = err
				best_threshold_index = i
		return farest_err,best_threshold_index

	def cal_stump_err(self,D,ftr_index,threshold_index):
		stump_TFPNlist = TFPNlist[ftr_index][threshold_index]
		FPlist = stump_TFPNlist[FPINDEX]
		FNlist = stump_TFPNlist[FNINDEX]
		werr = 0.0
		for ind in FPlist:
			werr += D[ind]
		for ind in FNlist:
			werr += D[ind]
		if werr >= 1.0 or werr <= 0.0:
			print "The werr is",
			print werr
			print "The D is"
			print D
			print "sum of D is",sum(D)
			print "FPlist is"
			print FPlist
			print "FPlist len is",len(FPlist)
			print "FNlist is"
			print FNlist
			print "FNlist len is",len(FNlist)
			print ""
		return werr

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

def get_train_var_list(lines):
	"""
	Get X matrix and y matrix
	"""
	global train_ftr_vals_list
	train_ftr_vals_list = [[] for i in range(0,ftr_no)]
	X_list = []
	y_list = []
	invert_y_list = [[] for i in range(0,k)]
	for i,line in enumerate(lines):
		line = line.strip("\n\r")
		split_line = ""
		split_line = line.split()
		label = split_line[0]
		y_list.append(label)
		invert_y_list[int(label)].append(i)
		X = []
		for index_val in split_line[1:]:
			ind,val = index_val.split(":")
			X.append((int(ind),float(val)))
			train_ftr_vals_list[int(ind)].append((i,float(val)))
		X_list.append(X)
	return X_list,y_list,invert_y_list

def get_test_var_list(lines):
	X_list = []
	y_list = []
	for i,line in enumerate(lines):
		line = line.strip("\r\n")
		split_line = line.split()
		label = split_line[0]
		y_list.append(label)
		X = []
		for index_val in split_line[1:]:
			ind,val = index_val.split(":")
			X.append((int(ind),float(val)))
		X_list.append(X)
	return X_list,y_list

def generate_stumps():
	global ftrs_threshold_list
	for i,vals_list in enumerate(train_ftr_vals_list):
		non0_list = [0.0]
		non0_list.extend(sorted([v[1] for v in vals_list]))
		if len(non0_list) == 0:
			ftrs_threshold_list.append([-0.05,0.0,0.05])
			continue
		threshold_list = [-0.05]
		last_val = non0_list[0]
		for cur_val in non0_list[1:]:
			if not cur_val == last_val:
				threshold_list.append((last_val+cur_val)/2.0)
				last_val = cur_val
		threshold_list.append(threshold_list[-1]+0.05)
		ftrs_threshold_list.append(threshold_list)
	
def generate_SWIL():
	global TFPNlist
	TFPNlist = []
	poslist = invert_y_list[1]
	neglist = invert_y_list[0]
	for i,stumps in enumerate(ftrs_threshold_list):
		ftr_vals = train_ftr_vals_list[i]
		stumps_len = len(stumps)
		stump_TFPNlist = []
		for j,stump in enumerate(stumps):
			TPlist,FPlist,TNlist,FNlist = [[],[],[],[]]
			if j == 0:
				FPlist = neglist
				TPlist = poslist
			elif j == stumps_len-1:
				TNlist = neglist
				FNlist = poslist
			else:
				cur_start = 0
				for val in ftr_vals:
					X_ind = val[0]
					cur_part_y_list = y_list[cur_start:X_ind]
					for i,y in enumerate(cur_part_y_list):
						if y == 1.0:
							FNlist.append(i+cur_start)
						else:
							TNlist.append(i+cur_start)
					if val[1] > stump:
						if y_list[X_ind] == 1.0:
							TPlist.append(X_ind)
						else:
							FPlist.append(X_ind)
					else:
						if y_list[X_ind] == 1.0:
							FNlist.append(X_ind)
						else:
							TNlist.append(X_ind)
					cur_start = X_ind+1
				cur_part_y_list = y_list[cur_start:]
				for i,y in enumerate(cur_part_y_list):
					if y == 1.0:
						FNlist.append(i+cur_start)
					else:
						TNlist.append(i+cur_start)
			stump_TFPNlist.append([TPlist,FPlist,TNlist,FNlist])
		TFPNlist.append(stump_TFPNlist)

def get_binary_y_list(func):
	pos_label_list = []
	for i,bit in enumerate(func):
		if bit == 1:
			pos_label_list.append(i)
	cur_y_list = []
	cur_invert_y_list = [[],[]]
	for i,y in enumerate(orig_y_list):
		if int(y) in pos_label_list:
			cur_y_list.append(1.0)
			cur_invert_y_list[1].append(i)
		else:
			cur_y_list.append(-1.0)
			cur_invert_y_list[0].append(i)
	return cur_y_list,cur_invert_y_list

def generate_funcs():
	"""
	Exhaustive Dodes: Generate 8 functions
	"""
	global func_list,code_list
	all_func_list = ["".join(req) for req in itertools.product("01",repeat=k)]
	random.shuffle(all_func_list)
	func_list = all_func_list[0:func_no]
	func_list = [[int(n) for n in strf] for strf in func_list]
	for i in range(0,k):
		codeword = []
		for f in func_list:
			codeword.append(f[i])
		code_list.append(codeword)

def run():
	global X_list,orig_y_list,orig_y_inver_list,y_list,	\
			invert_y_list,test_X_list,test_y_list,ds_size,test_size
	generate_funcs()
	train_file = "8newsgroup/train.trec/feature_matrix.txt"
	test_file = "8newsgroup/test.trec/feature_matrix.txt"
	f_train = open(train_file,"r")
	alltrainlines = f_train.readlines()
	random.shuffle(alltrainlines)
	f_test = open(test_file,"r")
	testlines = f_test.readlines()
	f_train.close()
	f_test.close()

	ds_size = 1000
	trainlines = alltrainlines[0:ds_size]
	X_list,orig_y_list,orig_y_invert_list = get_train_var_list(trainlines)
	test_X_list,test_y_list = get_test_var_list(testlines)
	test_size = len(test_y_list)
	generate_stumps()
	plot_flag = False

	train_result_list,test_result_list = [[],[]]
	for i,func in enumerate(func_list):
		print "In the",i,"func loop"
		y_list,invert_y_list = get_binary_y_list(func)
		tick1 = time.time()
		generate_SWIL()
		tick2 = time.time()
		start_tick = time.time()
		print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"+	\
				" function",i
		test_result=Adaboost(CHOOSE_OPT,plot_flag)
		print "The test_result is",
		print test_result
		test_result_list.append(test_result)
		plot_flag = False
		end_tick = time.time()
		print "******** Time used:",end_tick-start_tick
	pred_list = translate_pred_result(test_result_list)
	test_err = cal_test_err(pred_list)
	print "The final Test_err is",test_err

def cal_test_err(pred_list):
	diff_count = 0.0
	for i,pred_val in enumerate(pred_list):
		if not pred_val == int(test_y_list[i]):
			diff_count += 1
	err_rate = diff_count/test_size
	return err_rate

def translate_pred_result(test_result_list):
	pred_vals = []
	for i in range(0,test_size):
		test_code = []
		for f in range(0,func_no):
			test_code.append(test_result_list[f][i])
		pred_val = cal_final_pred_val(test_code)
		pred_vals.append(pred_val)
	return pred_vals

def cal_final_pred_val(test_code):
	"""
	Calculate pred label for one data point
	"""
	min_diff_count = 21
	label_ind = -1
	for i,code in enumerate(code_list):
		diff_count = 0
		for f in range(0,func_no):
			if (test_code[f] == 1.0 and code[f] == 1) or 	\
				(test_code[f] == -1.0 and code[f] == 0):
				pass
			else:
				diff_count += 1
		if diff_count < min_diff_count:
			min_diff_count = diff_count
			label_ind = i
	return label_ind
	
def decide_func(result_list):
	"""
	Decide the label for each data point
	"""

run()

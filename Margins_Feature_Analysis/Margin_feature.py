#!/usr/bin/env python

"""
-------------------------------------------------------------------------------
About: Rank features by the fraction of average margin
Author: Ruiyu Xu
Time: Nov. 2014
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
T = 300
fold_size = 10
CHOOSE_OPT = "Opt_stump"
CHOOSE_RAND = "Rand_stump"
X_list = []
y_list = []
invert_y_list = []
train_ftr_vals_list = []
ftrs_threshold_list = []
TFPNlist = []
TPINDEX,FPINDEX,TNINDEX,FNINDEX = [0,1,2,3]
SWIL = []	
test_X_list = []
test_y_list = []
ftr_margin_record = []
"""
ftr_margin_record : list of (ftr_index,alpha,pred_val_list)
"""
ds_size = 0.0

def Adaboost(boost_type,plot_flag):
	"""
	Do adaboost on the given weak learner WL which contains several
	predictors.
	"""
	global D,ds_size
	start_t = time.time()
	WL = Weak_Learner(boost_type)
	fi_t = time.time()
	print "++++++++++Finished creating Weak_learner,time:",fi_t-start_t
	D = init_D(ds_size)
	round_err_list,train_err_list,test_err_list,auc_list = [[],[],[],[]]
	train_round_result = NP.zeros(ds_size)
	test_round_result = NP.zeros(len(test_y_list))
	alpha_list = []
	gamaf_list = []
	t = 1
	for t in range(1,T+1):
		ipslon_t,best_stump = WL.get_ht_min_werr(D,boost_type,t)
		best_ftr_index,threshold_index = best_stump
		round_err_list.append(ipslon_t)
		alpha = 0.5*M.log((1.0-ipslon_t)/ipslon_t,M.e)
		D = update_D(D,alpha,best_stump,ipslon_t)
		train_round_result += cal_train_round_result(alpha,best_stump)
		test_round_result += cal_test_round_result(alpha,best_stump)
		train_err,train_rtuple_list = 	\
						predict(best_stump,train_round_result,y_list)
		train_err_list.append(train_err)
		test_err,test_rtuple_list = 	\
					predict(best_stump,test_round_result,test_y_list)
		test_err_list.append(test_err)
		TPR_list,FPR_list,auc = record_roc(test_rtuple_list)
		auc_list.append(auc)
		ind = best_stump[0]
		thre = ftrs_threshold_list[ind][best_stump[1]]
		print "Round:",t,"Ftr:",ind+1,"Threshold:",thre,"Round_err:",	\
			ipslon_t,"Train_err:",train_err,"Test_err:",test_err,"Auc:",auc
	ftr_rank_list = cal_fraction()
	print ftr_rank_list
	sort_fraction = sorted(ftr_rank_list,key=operator.itemgetter(1),reverse=True)
	for rank in range(0,10):
		ftr_frac = sort_fraction[rank]
		print "rank",rank,": ftr",ftr_frac[0]
	if plot_flag:
		plot(round_err_list,train_err_list,test_err_list,auc_list)
		draw_test_roc(TPR_list,FPR_list)

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
	"""
	calculate alpha_t*h_t_x
	"""
	global ftr_margin_record
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
	ftr_margin_record.append((best_stump[0],alpha,rr))
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
		val = X[ftr_index]
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
	return train_err,rtuple_list

def cal_fraction():
	f_margin_list = []	# group by feature, 	
						#list of (ftr_index,list of margin_f_x)
	gama_f_list = []
	sort_ftr_margin_record = sorted(ftr_margin_record,	\
							key=operator.itemgetter(0))	
	print "The ftr_margin_record length is",len(ftr_margin_record)
	first_rec = sort_ftr_margin_record[0]
	last_ind = first_rec[0]
	rr = NP.array(first_rec[2])
	f_alpha = abs(first_rec[1])
	up_sum = rr
	f_alpha_sum = f_alpha
	for rec in sort_ftr_margin_record[1:]:
		cur_ind = rec[0]
		f_alpha = abs(rec[1])
		cur_rr = rec[2]
		if not cur_ind == last_ind:
			lx = NP.array(y_list)
			up = lx*up_sum
			f_margin_list.append((last_ind,up/f_alpha))	#TODO
			gama_f_list.append(f_alpha_sum)
			up_sum = rr
			f_alpha_sum = f_alpha
			last_ind = cur_ind
		else:
			up_sum += rr
			f_alpha_sum += f_alpha
	f_margin_list.append((last_ind,up/f_alpha))
	gama_f_list.append(f_alpha_sum)
	gama_f_list_sum = sum(gama_f_list)
	gama_f_list = [gama_f/gama_f_list_sum for gama_f in gama_f_list]
	margin_x_list = []	# TODO 1-dim only for data point
	for d in range(0,ds_size):
		margin_x = 0.0
		for j,gama_f in enumerate(gama_f_list):
			margin_x+= f_margin_list[j][1][d]
		margin_x_list.append(margin_x)

	frac_list = []	#TODO:return
	total_margin = sum(margin_x_list)
	print "gama_f_list is"
	print gama_f_list
	for j,gama_f in enumerate(gama_f_list):
		ftr_ind = f_margin_list[j][0]
		f_margin_sum = sum(f_margin_list[j][1])
		frac_up = gama_f*f_margin_sum
		frac = frac_up/total_margin
		frac_list.append((ftr_ind,frac))
	return frac_list
		
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
		This methods get the optimal decision stump for each feature (57 opt
		decision stumps), and compare them, then return the best one among the
		57 ones.
		Return: the hypothesis and index list of wrong hyp from the best
				predictor
		"""
		farest_err,best_threshold_index = 	\
				self.choose_best_stump(D,0)
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

	def choose_rand_predictor(self,D):
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
		
def generate_stumps():
	global ftrs_threshold_list
	for i,vals_list in enumerate(train_ftr_vals_list):
		non0_list = [0.0]
		non0_list.extend(sorted(vals_list))
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
				for X_ind,val in enumerate(ftr_vals):
					if val > stump:
						if y_list[X_ind] == 1.0:
							TPlist.append(X_ind)
						else:
							FPlist.append(X_ind)
					else:
						if y_list[X_ind] == 1.0:
							FNlist.append(X_ind)
						else:
							TNlist.append(X_ind)
			stump_TFPNlist.append([TPlist,FPlist,TNlist,FNlist])
		TFPNlist.append(stump_TFPNlist)

def get_train_var_list(lines):
	"""
	Get X matrix and y matrix
	"""
	global train_ftr_vals_list
	train_ftr_vals_list = [[] for i in range(0,ftr_no)]
	X_list = []
	y_list = []
	invert_y_list = [[],[]]
	for i,line in enumerate(lines):
		line = line.strip("\n\r")
		split_line = ""
		split_line = line.split(",")
		split_line = [float(s) for s in split_line if s]
		label = split_line[-1]
		invert_y_list[int(label)].append(i)
		label = 1.0 if split_line[-1] == 1.0 else -1.0
		y_list.append(label)
		X = []
		for ind,val in enumerate(split_line[:-1]):	
			X.append(X)
			train_ftr_vals_list[ind].append(float(val))
		X_list.append(X)
	return X_list, y_list,invert_y_list

def get_test_var_list(lines):
	X_list = []
	y_list = []
	for i,line in enumerate(lines):
		line = line.strip("\r\n")
		split_line = line.split(",")
		split_line = [float(s) for s in split_line if s]
		label = 1.0 if split_line[-1]==1.0 else -1.0
		y_list.append(label)
		X = []
		for val in split_line[:-1]:
			X.append(val)
		X_list.append(X)
	return X_list,y_list
	
def run(data_file,boost_type):
	global X_list,y_list,invert_y_list,invert_y_list,test_X_list,	\
			test_y_list,ds_size,test_size
	f_data = open(data_file,"r")
	all_lines = f_data.readlines()
	random.shuffle(all_lines)
	plot_flag = False
	r = 1

	train_data_lines = []
	test_data_lines = []
	for i,line in enumerate(all_lines):
		if i%fold_size == r-1:
			test_data_lines.append(line)
		else:
			train_data_lines.append(line)
	ds_size = len(train_data_lines)
	train_result_list,test_result_list = [[],[]]
	X_list,y_list,invert_y_list = get_train_var_list(train_data_lines)
	test_X_list,test_y_list = get_test_var_list(test_data_lines)
	test_size = len(test_y_list)
	generate_stumps()
	generate_SWIL()
	Adaboost(boost_type,plot_flag)
	plot_flag = False

run("spambase.data",CHOOSE_OPT)

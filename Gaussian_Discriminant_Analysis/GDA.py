#!/usr/bin/evn python

"""
-------------------------------------------------------------------------------
About: This is about the Gaussian Discriminant Analysis.
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

ftr_no = 57
run_round = 10

def GDA(train_lines,test_lines,feature_no,f_type):
	"""
	linear regreassion for house
	"""
	global ftr_no,file_type
	ftr_no = feature_no
	x_list,y_list = get_var_list(train_lines)
	trainset_len = len(train_lines)
	testset_len = len(test_lines)
	PHI,MIU0,MIU1,SIGMA = estimate(NP.array(x_list),	\
					NP.array(y_list),trainset_len)
	train_MSE,train_acc = predict_val(PHI,MIU0,MIU1,	\
					SIGMA,train_lines)
	test_MSE,test_acc = predict_val(PHI,MIU0,MIU1,	\
					SIGMA,test_lines)
	return train_MSE,train_acc,test_MSE,test_acc

def predict_val(PHI,MIU0,MIU1,SIGMA,lines):
	"""
	predicts the value by using the given PHI, MIU0, MIU1, SIGMA
	"""
	count = 0
	diff_sqr_sum = 0.0
	TP,FP,TN,FN = [0.0,0.0,0.0,0.0]
	for i,line in enumerate(lines):
		line = line.strip("\n\r")
		split_line = ""
		split_line = line.split(",")
		data_entry = [float(s) for s in split_line if s]
		if len(data_entry) == 0:
			continue
		else:
			real_val = float(data_entry[-1])
			data_entry = data_entry[:-1]
			count += 1
			pred_val = cal_pred_val(PHI,MIU0,MIU1,SIGMA,data_entry)
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
	print "ERror rate is",err_rate
	print "Accuracy is",accuracy
	return MSE,accuracy

def cal_pred_val(PHI,MIU0,MIU1,SIGMA,X):
	"""
	
	"""
	det = NP.linalg.det(SIGMA)
	gram1 = 1.0/(M.pow(2*M.pi,ftr_no/2.0)*M.pow(det,0.5))
	gram2_0 = 0.5*NP.dot(NP.dot((X-MIU0),SIGMA.I),(X-MIU0).T)
	gram2_1 = 0.5*NP.dot(NP.dot((X-MIU1),SIGMA.I),(X-MIU1).T)
	P_0 = M.pow(M.e,-gram2_0)
	P_1 = M.pow(M.e,-gram2_1)
	P_0 = gram1*M.pow(M.e,-gram2_0)
	P_1 = gram1*M.pow(M.e,-gram2_1)
	if P_1 >= P_0:
		return 1.0
	return 0.0

def get_var_list(lines):
	"""
	Get x matrix and its transpose matrix
	"""
	x_map = {i:[] for i in range(0,ftr_no)}
	x_T_matrix = []
	y_list = []
	x_list = []
	for i,line in enumerate(lines):
		line = line.strip("\n\r")
		split_line = ""
		split_line = line.split(",")
		split_line = [float(s) for s in split_line if s]
		if not len(split_line) == ftr_no + 1:
			continue
		else:
			x_list.append([float(x) for x in split_line[:-1]])
			y_list.append(float(split_line[-1]))
	return x_list,y_list

def estimate(X_list,y_list,m):
	"""
	m is the length of the whole dataset
	PHI = sum({y_i=1})/m	i=1 to m
	MIU0 = sum({y_i=0})*X_i/sum({y_i=0})
	MIU1 = sum({y_i=1})*X_i/sum({y_i=1})
	SIGMA = sum((X_i-u_yi)*(X_i-u_yi)T)/m	i=1 to m
	"""
	count_y1 = len([y for y in y_list if y==1.0])
	count_y0 = m-count_y1
	PHI = float(count_y1)/float(m)
	X_y0 = []
	X_y1 = []
	for i,y in enumerate(y_list):
		if y == 0.0:
			X_y0.append(X_list[i])
		else:
			X_y1.append(X_list[i])
	
	u0_numerator = NP.array(X_y0).sum(axis=0)
	u0_denominator = float(count_y0)
	print "The u0_denominator is",u0_denominator
	u1_numerator = NP.array(X_y1).sum(axis=0)
	u1_denominator = float(count_y1)
	print "The u1_denominator is",u1_denominator
	MIU0 = NP.tile(0.0,(1,ftr_no)) if u0_denominator == 0.0	\
			else NP.divide(u0_denominator,u0_numerator)
	MIU1 = NP.tile(0.0,(1,ftr_no)) if u1_denominator == 0.0	\
			else NP.divide(u1_denominator,u1_numerator)

	Xi_uy_list = []
	for i,X in enumerate(X_list):
		uy = MIU0 if y_list[i] == 0.0 else MIU1
		Xi_uy = NP.matrix(X-uy)
		Xi_uy_list.append(NP.dot(Xi_uy.T,Xi_uy))
	SIGMA = sum(Xi_uy_list)/float(m)
	return PHI,MIU0,MIU1,SIGMA

def matrix_mul(matrix1,matrix2):
	"""
	"""
	m1T_m2_inv = inv(m1.T)

def run_cross_folder(data_file,attr_no,f_type):
	f_data = open(data_file,"r")
	all_lines = f_data.readlines()
	R.shuffle(all_lines)
	data_size = len(all_lines)
	last_end_no = -1
	total_train_MSE,total_train_acc,total_test_MSE,total_test_acc = 	\
													[0.0,0.0,0.0,0.0]
	for r in range(1,run_round+1):
		print "-----------------------------------------------"
		print "round",r
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
		last_end_no = end_line_no
		train_MSE,train_acc,test_MSE,test_acc = 	\
			GDA(train_data_lines,test_data_lines,attr_no,f_type)
		print "Training mse,acc --------->>>>>>",train_MSE,train_acc
		print "Testing mse,acc ---------->>>>>>",test_MSE,test_acc
		total_test_MSE += test_MSE
		total_test_acc += test_acc
		total_train_MSE += train_MSE
		total_train_acc += train_acc
	print "Average training MSE:",total_train_MSE/run_round
	print "Average training acc:",total_train_acc/run_round
	print "Average testing MSE:",total_test_MSE/run_round
	print "Average testing acc:",total_test_acc/run_round

print " Spambase data"
run_cross_folder("spambase.data",57,"spam")

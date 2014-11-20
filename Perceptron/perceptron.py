#!/usr/bin/evn python

"""
-------------------------------------------------------------------------------
About: Create a perceptron learning algorithm
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

ftr_no = 4
init_w_val = 3.0
step =0.1 

def perceptron(train_lines):
	"""
	It takes train_lines read from train data file as the input argument.
	"""
	X_list,y_list = get_var_list(train_lines)
	X_matrix = NP.array(X_list)
	raw_W_matrix = percep_estimation(X_matrix,y_list,train_lines)
	print "The raw W_matrix is:"
	print raw_W_matrix
	norm_W_matrix = normalize_W(raw_W_matrix)
	print "The normalized W_matrix is:"
	print norm_W_matrix

def normalize_W(W):
	"""
	create the normalized weights by dividing your perceptron weights 
	w1, w2, w3, and w4 by -w0
	"""
	_w0 = -W[0]
	return [w/_w0 for w in W[1:]] 

def percep_estimation(X_matrix,y_list,train_lines):
	"""
	Perceptron estimation
	"""
	W_matrix = NP.tile(3.0,(1,ftr_no+1))[0]
	M = get_M(W_matrix,X_matrix,y_list)
	M_len = len(M)
	count = 0
	update_count = 0
	while(len(M) > 0):
		count += 1
		for t,x in enumerate(M):
			update_count += 1
			new_w_list = map(lambda w_j,x_j:w_j+step*x_j,W_matrix,x)
			W_matrix = NP.array(new_w_list)
		M = get_M(W_matrix,X_matrix,y_list)
		M_len = len(M)
		print "iteration:",count,"----","total mistake:",M_len
	print "The total update number is:",update_count
	return W_matrix

def get_M(W,X,y_list):
	"""
	Calculate the predict values and return missclassified data points set M
	"""
	M = []
	for i,xt in enumerate(X):
		pred_val = cal_h_W_Xt(W,xt)
		if not pred_val == y_list[i]:
			M.append(xt)
	return M

def cal_h_W_Xt(W_matrix,Xt):
	"""
	Calculate and return h_W(Xt)
	h_W(Xt) = W*Xt
	"""
	pred_val = NP.dot(W_matrix,Xt)
	pred_val = 1.0 if pred_val >= 0.0 else -1.0
	return pred_val

def get_var_list(lines):
	"""
	Get X list and y list
	"""
	y_list = []
	X_list = []
	for i,line in enumerate(lines):
		line = line.strip("\n\r")
		split_line = line.split("\t")
		split_line = [float(s) for s in split_line if s]
		if not len(split_line) == ftr_no + 1:
			continue
		else:
			X_i = []
			y = float(split_line[-1])
			if y == -1.0:
				X_i = [-1.0]
				X_i.extend([-float(x) for x in split_line[:-1]])
				y_list.append(-y)
			else:
				X_i = [1.0]
				X_i.extend([float(x) for x in split_line[:-1]])
				y_list.append(y)
			X_list.append(X_i)
	return X_list,y_list

def run_perceptron(data_file):
	"""
	Main function to run perceptron
	"""
	f_data = open(data_file,"r")
	train_data_lines = f_data.readlines()
	perceptron(train_data_lines)

run_perceptron("perceptronData.txt")

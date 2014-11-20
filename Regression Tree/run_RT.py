#!/usr/bin/env python

import RT_train
import RT_test

run_round = 10

def run_RT(train_data_file,test_data_file):
	"""

	"""
	f_train = open(train_data_file,"r")
	f_test = open(test_data_file,"r")
	train_lines = f_train.readlines()
	print "Train data set size,",len(train_lines)
	test_lines = f_test.readlines()
	print "Test data set size,",len(test_lines)
	f_train.close()
	f_test.close()
	print "============================================"
	RT_train.create_RT(train_lines)
	train_MSE = RT_test.predict_data(train_lines)
	print "------->>>>Training Error",train_MSE
	test_MSE = RT_test.predict_data(test_lines)
	print "------->>>>Testing Error",test_MSE
	print "============================================"

run_RT("housing_train.txt","housing_test.txt")

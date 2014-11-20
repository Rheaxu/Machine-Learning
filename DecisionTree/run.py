#!/usr/bin/env python
"""
------------------------------------------------------------------- 
About: This script run the Decision tree train and test procedure
-------------------------------------------------------------------
"""

import DT_train
import DT_test
import random

run_round = 10

def run_cross_folder(data_file):
	"""

	"""
	f_data = open(data_file,"r")
	all_lines = f_data.readlines()
	random.shuffle(all_lines)
	data_size = len(all_lines)
	print "data size",data_size
	last_end_no = -1
	total_train_MSE = 0.0
	total_train_er = 0.0
	total_train_acc = 0.0
	total_test_MSE = 0.0
	total_test_er = 0.0
	total_test_acc = 0.0
	for r in range(1,run_round+1):
		print "============================================"
		train_data_lines = []
		test_data_lines = []
		start_line_no = last_end_no+1
		percent = float(r)/run_round
		end_line_no = int(percent*data_size)
		for i,line in enumerate(all_lines):
			if i>=start_line_no and i<=end_line_no:
				test_data_lines.append(line)
			else:
				train_data_lines.append(line)
		last_end_no = end_line_no
		DT_train.create_DT(train_data_lines,run_round = r)
		print "Testing error----------->>>>>>>"
		test_MSE,test_er,test_acc =  DT_test.classify_data(test_data_lines,run_round = r)
		total_test_MSE += test_MSE
		total_test_er += test_er
		total_test_acc += test_acc
		print "Training error----------->>>>>>>"
		train_MSE,train_er,train_acc = DT_test.classify_data(train_data_lines,run_round = r)
		total_train_MSE += train_MSE
		total_train_er += train_er
		total_train_acc += train_acc
		print "============================================"
	print "Average trainning MSE:",total_train_MSE/run_round
	print "Average training error rate:",total_train_er/run_round
	print "Average training accuracy:",total_train_acc/run_round
	print "Averate testing MSE:",total_test_MSE/run_round
	print "Average testing error rate:",total_test_er/run_round
	print "Average testing accuracy:",total_test_acc/run_round
                      
run_cross_folder("spambase.data")

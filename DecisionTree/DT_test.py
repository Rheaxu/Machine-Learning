#!/usr/bin/env python

import sys
import math as M

def classify_data(test_data_lines,**kwargs):
	"""
	Classify the data from the given test data file
	Args:
		run_round: int number
	"""
	data_set = []
	f_result_name = "DT_test_result"
	f_tree_name = "decision_tree"
	if "run_round" in kwargs:
		r = kwargs["run_round"]
		f_result_name = f_result_name+"_"+str(r)+".txt"
		f_tree_name = f_tree_name+"_"+str(r)+".tree"
	else:
		f_result_name = f_result_name+".txt"
		f_tree_name = f_tree_name+".tree"
	f_result = open(f_result_name,"w")
	DT_tree = create_DT_tree(f_tree_name)
	TP = 0.0
	FP = 0.0
	TN = 0.0
	FN = 0.0
	count = 0
	diff_sqr_sum = 0.0
	for line in test_data_lines:
		line = line.strip("\r\n")
		data_entry = line.split(",")
		result = classify_data_helper(data_entry,DT_tree)
		count += 1
		real_val = data_entry[-1]
		if real_val == "1" and result == "1":
			TN += 1
		elif real_val == "0" and result == "1":
			FN += 1
		elif real_val == "0" and result == "0":
			TP += 1
		elif real_val == "1" and result == "0":
			FP += 1
		diff_sqr_sum += M.pow(float(result)-float(real_val),2)
		f_result.write(result+"\n")
	f_result.close()
	MSE = diff_sqr_sum/count
	print "MSE is,",MSE
	er,acc = evaluation(TP,FP,TN,FN)
	return MSE,er,acc

def classify_data_helper(data_entry,DT_tree):
	"""
	"""
	left_branch = True
	last_layer = -1
	label = ""
	for n in DT_tree:
		if not n.layer == last_layer+1:
			continue
		elif not left_branch:
			left_branch = True
			continue
		else:
			last_layer = n.layer
			if n.n_type == "leaf":
				label = n.name
				break
			else:
				index = int(n.name.split("_")[1])
				attr_val = float(data_entry[index])
				split_point = float(n.split_point)
				if attr_val < split_point:
					left_branch = True
				else:
					left_branch = False
	return label

def evaluation(TP,FP,TN,FN):
	"""
	"""
	print "TP--"+str(TP),"FP--"+str(FP),"TN--"+str(TN),"FN--"+str(FN)
	TOT = TN+FP+FN+TP
	error_rate = (FP+FN)/TOT
	print "error rate =",error_rate
	accuracy = 1-error_rate
	print "accuracy =",accuracy
	return error_rate,accuracy

class node():
	"""
	"""
	def __init__(self,layer,name,n_type,split_point):
		self.layer =layer
		self.name = name
		self.n_type = n_type
		self.split_point = split_point

def create_DT_tree(f_tree_name):
	"""
	Create DT_tree from the given decision tree file
	DT_tree is a list of node stored in Depth First method
	"""
	f_tree = open(f_tree_name,"r")
	lines = f_tree.readlines()
	DT_tree = []
	for line in lines:
		node = get_line_content(line)
		DT_tree.append(node)
	return DT_tree

def get_line_content(line):
	"""
	Split the line, return layer,node
	"""
	line = line.strip("\r\n")
	split1 = line.split("-")
	split2 = split1[1].split(",")
	node_layer = int(split1[0])
	node_name = split2[0]
	n_type = split2[1]
	split_point = split2[2]

	return node(node_layer,node_name,n_type,split_point)


#!/usr/bin/env python

"""
-------------------------------------------------------------------------------
About : Regression Tree Algorithm, this module creates regression tree from 
		given training data set (raw lines read from data file) adn store them
		into .tree file
Author : Ruiyu Xu
-------------------------------------------------------------------------------
"""
import operator
import math as M
import re

RT_LAYERS = 4
attr_map = {}
attr_no = 13
all_attr_names = ["attr_"+str(i) for i in range(0,attr_no)]
default_labl = 0.0
BRANCH_HOLDER = "++"

def create_RT(train_data_lines,**kwargs):
	"""
	This method create the regression tree.
	Args:
		run_round: int number
	"""
	global attr_map
	data_set,attr_map = readin_data(train_data_lines)
	tree_file_name = "regression_tree"
	finish_msg = "Finished"
	if "run_round" in kwargs:
		r = kwargs["run_round"]
		tree_file_name = tree_file_name + "_"+str(r)+".tree"
		finish_msg += " round "+str(r)
	else:
		tree_file_name = tree_file_name+".tree"
	f_tree = open(tree_file_name,"w")
	RT = Rtree(data_set,all_attr_names,f_tree)
	f_tree.close()
	print finish_msg

def readin_data(train_data_lines):
	"""
	Read in training data from file, generate and return the data set
	and update the attr_map which is the dict of all values for all attr.
	"""
	data_set = []
	attr_map = {labl:[] for labl in all_attr_names}
	for line in train_data_lines:
		line = line.strip("\r\n")
		data_entry = re.compile(r'[\t\s]').split(line) 
		data_entry = [float(s) for s in data_entry if s]
		if not len(data_entry)==attr_no+1:
			continue
		else:
			data_set.append(data_entry)
			attr_map = update_attr_map(data_entry,attr_map)
	return data_set,attr_map

def update_attr_map(data_entry,old_attr_map):
	"""
	Generates the attribute map. (attr_name:value_list)
	"""
	new_attr_map = {}
	for i,val in enumerate(data_entry):
		if i >= attr_no:
			break
		labl = all_attr_names[i]
		vals_list = old_attr_map[labl]
		vals_list.append(val)
		new_attr_map[labl] = vals_list
	return new_attr_map

class Rtree():
	"""
	Regression Tree class.
	"""
	def __init__(self,data_set,attr_list,f_tree,**kwargs):
		"""
		Initiates the regression tree
		The attr_no is the number of features, not including the label
		Args:
		layer: The layer number of the current subtree. Root is layer 0
		"""
		self.layer = 0 if not "layer" in kwargs else int(kwargs["layer"])
		self.data_set = data_set
		self.attr_list = attr_list[:]
		self.__create_RT__(f_tree)

	def __create_RT__(self,f_tree):
		"""
		Create regression tree by using the given training data file
		"""
		current_data_set = self.data_set
		candit_attrs = self.attr_list
		label_value_list = self.get_labl_value_list(current_data_set)
		if len(current_data_set)==0 or len(candit_attrs)==0:
			self.root = node(default_labl,self.layer,0.0,isleaf=True)
			n_type = "leaf" if self.root.isleaf else "node"
			node_str = str(self.layer)+"-"+str(self.root.name)+","+	\
						n_type+","+	str(self.root.split_point)
			f_tree.write(node_str+"\n")
			print BRANCH_HOLDER+BRANCH_HOLDER*self.layer+"Layer:"+	\
					str(self.layer)+","+"label:"+str(self.root.name)
		elif self.layer > RT_LAYERS:
			n_type = "leaf"
			labels = []
			for d in self.data_set:
				labels.append(d[-1])
			self.root = node(cal_g(labels),self.layer,0.0,isleaf=True)
			node_str = str(self.layer)+"-"+str(self.root.name)+","+	\
						n_type+","+	str(self.root.split_point)
			f_tree.write(node_str+"\n")
			print BRANCH_HOLDER+BRANCH_HOLDER*self.layer+"Layer:"+	\
					str(self.layer)+","+"label:"+str(self.root.name)
		else:
			self.root = choose_best_node(current_data_set,candit_attrs,self.layer)
			n_type = "node"
			node_str = str(self.root.layer)+"-"+str(self.root.name)+","+	\
						n_type+","+str(self.root.split_point)
			f_tree.write(node_str+"\n")
			print BRANCH_HOLDER+BRANCH_HOLDER*self.layer+"Layer:"+	\
					str(self.layer)+","+"attr:"+str(self.root.name)+","	\
					"split_point:"+str(self.root.split_point)
			lchild_data_set = []
			rchild_data_set = []
			attr_name = self.root.name
			split_point = self.root.split_point
			attr_index = all_attr_names.index(attr_name)
			for d in current_data_set:
				a = float(d[attr_index])
				if a<split_point:
					lchild_data_set.append(d)
				else:
					rchild_data_set.append(d)
			remain_attrs = self.attr_list[:]
			remain_attrs.remove(attr_name)
			new_layer = self.layer+1
			self.lchild = Rtree(lchild_data_set,remain_attrs,f_tree,	\
						layer=new_layer)
			self.rchild = Rtree(rchild_data_set,remain_attrs,f_tree,	\
						layer=new_layer)

	def get_labl_value_list(self,current_data_set):
		"""
		Collects the label values and put them into a list
		"""
		value_list = []
		for d in current_data_set:
			value_list.append(d[-1])
		return value_list

def choose_best_node(data_set,candit_attrs,layer):
	"""
	This method chooses the best attributes to be the node
	The attr_TE_map records the total error caused by splitting the the node
	by each attribute
	"""
	total = len(data_set)
	attr_TE_map = {}
	attr_name = candit_attrs[0]
	attr_val_list = get_attr_val_list(attr_name,data_set)
	split_point,attr_TE = cal_TE(attr_val_list,data_set)
	for attr in candit_attrs[1:]:
		attr_val_list = get_attr_val_list(attr,data_set)
		next_split_point,next_attr_TE = cal_TE(attr_val_list,data_set)
		if next_attr_TE < attr_TE:
			attr_name = attr
			attr_TE = next_attr_TE
			split_point = next_split_point
	return node(attr_name,layer,split_point)

def get_attr_val_list(attr_name,data_set):
	"""
	Generate the list of value of an attribute from the given data_set
	"""
	index = all_attr_names.index(attr_name)
	val_list = []
	for d in data_set:
		val_list.append(float(d[index]))
	return val_list

def cal_TE(attr_val_list,data_set):
	"""
	Calculates the total error caused by splitting the tree by the given
	attibute. Returns the total error and the split point
	"""
	split_point,TE = choose_best_split_point(attr_val_list,data_set);
	return split_point,TE

def cal_error(labels):
	"""
	Error = sum(i)(labeli-g)^2
	"""
	g = cal_g(labels)
	total = 0.0
	for label in labels:
		total += M.pow((label-g),2)
	return float(total)

def cal_g(labels):
	"""
	g = sum(labels)/|labels|
	"""
	size = len(labels)
	if size == 0:
		return 0.0
	total = 0.0
	for label in labels:
		total += label
	return float(total)/size

def choose_best_split_point(attr_val_list,data_set):
	"""
	Select the best split point for the given attribute
	"""
	candit_point_list = cal_candit_split_point_list(attr_val_list)
	TE_map = {}	#map of expected information of candit points
	p = candit_point_list[0]
	TE = cal_point_TE(attr_val_list,p,data_set)
	for next_p in candit_point_list[1:]:
		next_TE = cal_point_TE(attr_val_list,next_p,data_set)
		if next_TE < TE:
			TE = next_TE
			p = next_p
	return p,TE

def cal_candit_split_point_list(attr_val_list):
	"""
	Calculates candidate split points from the given attribute value list
	"""
	candit_point_set = sorted(set(attr_val_list))
	if len(candit_point_set) < 2:
		return list(candit_point_set)
	split_point_list = map(lambda x,y:(x+y)/2.0,candit_point_set[:-1],	\
						candit_point_set[1:])
	return split_point_list

def cal_point_TE(attr_val_list,point,data_set):
	"""
	Calculates the total error of two branch splitted by the given point
	"""
	lt_data_label_set = []
	ge_data_label_set = []
	for i,d in enumerate(data_set):
		if attr_val_list[i] < point:
			lt_data_label_set.append(d[-1])
		else:
			ge_data_label_set.append(d[-1])
	return cal_error(lt_data_label_set)+cal_error(ge_data_label_set)

class node():
	"""
	The node of regression tree
	PARAMETERS:
		name: the name of the attribute the node represents
	"""
	def __init__(self,name,layer,split_point,**kwargs):
		"""
		args: isleaf = True/False
		"""
		self.isleaf=False if not "isleaf" in kwargs else True
		self.name = name
		self.layer = layer
		self.split_point = split_point

	def fall_to_lchild(self,value):
		return value<split_point

	def fall_to_rchild(self,value):
		return value >= split_point


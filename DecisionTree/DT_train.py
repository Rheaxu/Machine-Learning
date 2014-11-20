#!/usr/bin/env python

"""
-------------------------------------------------------------------------------
About : Decision Tree Algorithm, this module creates decision tree from given
		training data set (raw lines read from data file) adn store them into
		.tree file
Author : Ruiyu Xu
Date: Sept. 2014
-------------------------------------------------------------------------------
"""
import operator
import math as M

DT_LAYERS = 5
attr_no = 57
all_attr_names = ["attr_"+str(i) for i in range(0,attr_no)]
default_labl = "0"
BRANCH_HOLDER = "++"

def create_DT(train_data_lines,**kwargs):
	"""
	This method create the decision tree.
	Args:
		run_round: int number
	"""
	data_set = readin_data(train_data_lines)
	tree_file_name = "decision_tree"
	finish_msg = "Finished"
	if "run_round" in kwargs:
		r = kwargs["run_round"]
		tree_file_name = tree_file_name + "_"+str(r)+".tree"
		finish_msg += " round "+str(r)
	else:
		tree_file_name = tree_file_name+".tree"
	f_tree = open(tree_file_name,"w")
	DT = Dtree(data_set,all_attr_names,f_tree)
	f_tree.close()
	print finish_msg

def readin_data(train_data_lines):
	"""
	Read in training data from file, generate and return the data set
	and update the attr_map which is the dict of all values for all attr.
	"""
	data_set = []
	for line in train_data_lines:
		line = line.strip("\r\n")
		data_entry = line.split(",")
		if not len(data_entry)==attr_no+1:
			continue
		else:
			data_set.append(data_entry)
	return data_set

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

class Dtree():
	"""
	Decision Tree class.
	"""
	def __init__(self,whole_data_list,attr_list,f_tree,**kwargs):
		"""
		Initiates the decision tree
		The attr_no is the number of features, not including the label
		Args:
		layer: The layer number of the current subtree. Root is layer 0
		"""
		self.layer = 0 if not "layer" in kwargs else int(kwargs["layer"])
		self.data_set = data_set(whole_data_list)
		self.attr_list = attr_list[:]
		self.__create_DT__(f_tree)

	def __create_DT__(self,f_tree):
		"""
		Create decision tree by using the given training data file
		"""
		if self.data_set.total_num==0 or len(self.attr_list)==0 	\
			or self.data_set.set_1_num == 0:
			self.__generate_write_leaf_node__("0",f_tree)
		elif self.data_set.set_0_num==0:
			self.__generate_write_leaf_node__("1",f_tree)
		elif self.layer > DT_LAYERS:
			leaf_label = default_labl
			if self.data_set.set_1_num >= self.data_set.set_0_num:	
				leaf_label = "1"
			self.__generate_write_leaf_node__(leaf_label,f_tree)
		else:
			self.root = 	\
					self.data_set.choose_best_node(self.attr_list,self.layer)
			self.__generate_write_node__(f_tree)
			self.__create_sub_tree__(f_tree)

	def __create_sub_tree__(self,f_tree):
		"""
		Create left and right sub tree
		"""
		lchild_data_list = []
		rchild_data_list = []
		attr_name = self.root.name
		sp = self.root.sp.value
		attr_index = int(attr_name.split("_")[1])
		for d in self.data_set.whole_set:
			a = float(d[attr_index])
			if a<sp:
				lchild_data_list.append(d)
			else:
				rchild_data_list.append(d)
		remain_attrs = self.attr_list[:]
		remain_attrs.remove(attr_name)
		new_layer = self.layer+1
		self.lchild = Dtree(lchild_data_list,remain_attrs,f_tree,	\
					layer=new_layer)
		self.rchild = Dtree(rchild_data_list,remain_attrs,f_tree,	\
					layer=new_layer)

	def __generate_write_leaf_node__(self,label,f_tree):
		sp = split_point(0.0,[],[],0.0,0.0,[],-1)
		n_type = "leaf"
		self.root = node(label,self.layer,sp,isleaf = True)
		node_str = str(self.layer)+"-"+self.root.name+","+n_type+	\
					","+str(self.root.sp.value)
		f_tree.write(node_str+"\n")
		print BRANCH_HOLDER+BRANCH_HOLDER*self.layer+"Layer:"+	\
				str(self.layer)+","+"label:"+self.root.name
	
	def __generate_write_node__(self,f_tree):
		n_type = "node"
		node_str = str(self.root.layer)+"-"+self.root.name+","+	\
					n_type+","+str(self.root.sp.value)
		f_tree.write(node_str+"\n")
		print BRANCH_HOLDER+BRANCH_HOLDER*self.layer+"Layer:"+	\
				str(self.layer)+","+"attr:"+self.root.name+","+	\
				"split_point:"+str(self.root.sp.value)

class data_set():
	"""
	The data_set class contains the number of data labelled as "1" and "0",
	its entropy
	"""
	def __init__(self,whole_set):
		self.whole_set = whole_set
		self.set_0,self.set_1 = self.__sep_dataset__()
		self.set_0_num = len(self.set_0)
		self.set_1_num = len(self.set_1)
		self.total_num = self.set_1_num + self.set_0_num
		self.entropy = 0.0 if self.total_num == 0 else self.__cal_entropy__() 

	def __sep_dataset__(self):
		"""
		Separate the dataset to a set all labeled as "1" and another "0"
		"""
		set_1 = []
		set_0 = []
		for d in self.whole_set:
			if d[-1] == "0":
				set_0.append(d)
			else:
				set_1.append(d)
		return set_0,set_1

	def __cal_entropy__(self):
		"""
		H(Y) = sum(j)P(yj)*log2(1/P(yj))
		"""
		P_1 = float(self.set_1_num)/self.total_num
		P_0 = float(self.set_0_num)/self.total_num
		log_1 = 0.0 if P_1==0.0 else M.log(1.0/P_1,2.0)
		log_0 = 0.0 if P_0==0.0 else M.log(1.0/P_0,2.0)
		return P_1*log_1 + P_0*log_0

	def choose_best_node(self,candit_attrs,layer):
		"""
		This method chooses the best attributes to be the split node for
		the data_set
		"""
		attr_name = candit_attrs[0]
		attr_index = int(attr_name.split("_")[1])
		attr_val_list = self.__get_attr_val_list__(attr_index)
		sp = self.choose_best_split_point(attr_val_list,attr_index)
		split_node = node(attr_name,layer,sp)
		max_IG = self.entropy - split_node.sp.cond_entropy
		for attr in candit_attrs[1:]:
			attr_index = int(attr.split("_")[1])
			attr_val_list = self.__get_attr_val_list__(attr_index)
			next_split_point = self.choose_best_split_point(attr_val_list,attr_index)
			next_split_node = node(attr,layer,next_split_point)
			next_IG = self.entropy - next_split_node.sp.cond_entropy
			if next_IG > max_IG:
				attr_name = attr
				max_IG = next_IG
				split_node = next_split_node
		return split_node

	def __get_attr_val_list__(self,attr_index):
		"""
		Generate the list of value of an attribute from the given data_set
		"""
		val_list = []
		for d in self.whole_set:
			val_list.append(float(d[attr_index]))
		return val_list

	def cal_IG(self,attr_val_list,attr_index):
		"""
		Calculated the information gain of an attribute from the given data_set
		IG(X)=H(Y)-H(Y|X)
		"""
		sp = self.choose_best_split_point(attr_val_list,attr_index);
		IG = self.entropy - sp.cond_entropy
		return sp,IG

	def choose_best_split_point(self,attr_val_list,attr_index):
		"""
		Select the best split point for the given attribute
		"""
		candit_point_list = self.cal_candit_split_point_list(attr_val_list,attr_index)
		sp = candit_point_list[0]
		min_entropy = sp.cond_entropy
		for cp in candit_point_list[1:]:
			if cp.cond_entropy < min_entropy:
				min_entropy = cp.cond_entropy
				sp = cp
		return sp

	def cal_candit_split_point_list(self,attr_val_list,attr_index):
		"""
		Calculates candidate split points from the given attribute value list
		"""
		candit_val_list = sorted(set(attr_val_list))
		split_point_obj_list = []
		candit_len = len(candit_val_list)
		if candit_len < 2:
			p_value = candit_val_list[0]
			p_obj = self.__generate_p_obj__(p_value,attr_val_list,attr_index)
			split_point_obj_list.append(p_obj)
		else:
			for i in range(0,candit_len-1):
				p_value = (candit_val_list[i]+candit_val_list[i+1])/2.0
				p_obj = self.__generate_p_obj__(p_value,attr_val_list,attr_index)
				split_point_obj_list.append(p_obj)

		return split_point_obj_list
		
	def __generate_p_obj__(self,p_value,attr_list,attr_index):
		l_data_list,r_data_list,P_l,P_ge = self.__split_dataset_by_p__(p_value,attr_list)
		return split_point(p_value,l_data_list,r_data_list,P_l,P_ge,attr_list,attr_index)

	def __split_dataset_by_p__(self,p_value,attr_list):
		"""
		Split the data set to dataset(attr<p) and dataset(attr>=p)
		Return two data_set object and their probability
		"""
		l_data_list = []
		ge_data_list = []
		count_l = 0.0
		count_ge = 0.0
		for i,v in enumerate(attr_list):
			if v<p_value:
				count_l += 1
				l_data_list.append(self.whole_set[i])
			else:
				count_ge += 1
				ge_data_list.append(self.whole_set[i])
		P_l = count_l/(count_l+count_ge)
		P_ge = 1.0-P_l
		return l_data_list,ge_data_list,P_l,P_ge

class split_point():
	"""
    A class of split point, it contains the current value list of the 	\
    attribute, the index of the attribute and the length of the value list
    """
	def __init__(self,value,l_data_list,r_data_list,P_l,P_ge,	\
                 attr_list,attr_index):
		self.value = value
		self.l_data_set = data_set(l_data_list)
		self.r_data_set = data_set(r_data_list)
		self.attr_list = attr_list
		self.attr_index = attr_index
		self.attr_list_len = len(self.attr_list)
		self.prob_l = self.__cal_split_prob__()
		self.prob_ge = 1.0-self.prob_l
		self.cond_entropy = self.prob_l*self.l_data_set.entropy+	\
            self.prob_ge*self.r_data_set.entropy
    
	def __cal_split_prob__(self):
		count_l = 0.0
		if self.attr_list_len == 0:
			return 0.0
		for i,v in enumerate(self.attr_list):
			if v < self.value:
				count_l += 1
		prob_l = float(count_l)/self.attr_list_len
		return prob_l

class node():
	"""
	The node of decision tree
	PARAMETERS:
		name: the name of the attribute the node represents
	"""
	def __init__(self,name,layer,sp,**kwargs):
		"""
		args: isleaf = True/False
		"""
		self.isleaf=False if not "isleaf" in kwargs else True
		self.name = name
		self.layer = layer
		self.sp = sp



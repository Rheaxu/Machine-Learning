#!/usr/bin/evn python
"""
-------------------------------------------------------------------------------
About: Implementation of Neural Network
Author: Ruiyu Xu
Sept. 2014
-------------------------------------------------------------------------------
"""

import numpy as NP
import math as M
import numpy as NP
import random as R

w_start_val = 1
b_start_val = 1
INPUT_LAYR_INDEX = 0
HID_LAYR_INDEX = 1
OUTPUT_LAYR_INDEX = 2
INPUT_UNIT_NO = 8
HID_UNIT_NO = 3
OUTPUT_UNIT_NO = 8
T = 0.9	#Threshold
train_set = [[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],	\
				[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],	\
				[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]]
hidden_values = [[.89,.04,.08],[.15,.99,.99],[.01,.97,.27],[.99,.97,.71],	\
				[.03,.05,.02],[.01,.11,.88],[.80,.01,.98],[.60,.94,.01]]
l = 0.2	#learning rate

def run(train_set):
	"""
	Main algorithm of NN
	"""
	inputs = [[float(x) for x in inp] for inp in train_set]
	n_network = NN(inputs,hidden_values)
	W_matrix = n_network.train_NN(l)
	print W_matrix
	for inp in inputs:
		n_network.test_NN(inp)

class NN():
	"""
	"""
	def __init__(self,data_set,hid_targets):
		"""
		"""
		self.train_set = data_set
		self.layers = self.__init_layrs__()
        #TODO: TO generate difference initial weights, uncomment this code
		########self.w_list,self.biases= self.__init_weights_bias__()
		self.w_list,self.biases= self.__best_init_weights_bias__()
		print "*******>>>Initial w_list is"
		print self.w_list
		print "*******>>>Initial biases is"
		print self.biases
		self.func = sigmoid
	
	def __init_layrs__(self):
		"""
		"""
		layers = []
		input_layr = layer(NP.tile(0.0,INPUT_UNIT_NO))
		layers.append(input_layr)
		hid_layr = layer(NP.tile(0.0,HID_UNIT_NO))
		layers.append(hid_layr)
		output_layr = layer(NP.tile(0.0,OUTPUT_UNIT_NO))
		layers.append(output_layr)
		return layers
	
	def __best_init_weights_bias__(self):
		"""
		There are several optimum initial value for weights, here uses one
		"""
		w_list = [[[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
				[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
				[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]],[[ 0.,  0.,  0.],
				[ 1.,  1.,  1.],[ 0.,  0.,  0.], [1.,  1.,  1.],
				[ 1.,  1.,  1.],[ 0.,  0.,  0.],[ 0.,  0.,  0.],
				[ 1.,  1.,  1.]]]
		b_list = [[], [0.0, 1.0, 1.0],
				[0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]] 
		return w_list,b_list

	def __init_weights_bias__(self):
		"""
		wn_ij: n refers to the layer index, 0:input layer, 1:hiddent layer
			2:output layer; i refers to the unit index of the lower layer and 
			j refers to the unit index of the higher layer
		"""
		##### for input layer
		w_list = []
		b_list = []
		inp_layr_w = self.__init_weights_helper__(w_start_val,INPUT_UNIT_NO,HID_UNIT_NO)
		inp_layr_b = []
		w_list.append(inp_layr_w)
		b_list.append(inp_layr_b)
		###### for hid layer
		hid_layr_b = self.__init_biases__(b_start_val,HID_UNIT_NO)
		hid_layr_w = self.__init_weights_helper__(w_start_val+HID_UNIT_NO,HID_UNIT_NO,OUTPUT_UNIT_NO)
		b_list.append(hid_layr_b)
		w_list.append(hid_layr_w)
		##### for output layer
		out_layr_b = self.__init_biases__(b_start_val+HID_UNIT_NO,OUTPUT_UNIT_NO)
		b_list.append(out_layr_b)
		return w_list,b_list
		
	def __init_weights_helper__(self,start_value,layr_unit_no,	\
							higher_layr_unit_no):
		"""
		The wn_ij matrix is:
		j: higher_layr_unit_no
		i: layr_unit_no
		eg: For input layer
		w0 = [[1,1,1,1,1,1,1,1],[2,2,2,2,2,2,2,2],[3,3,3,3,3,3,3,3]]
		For hidden layer
		w1 = [[4,4,4],[5,5,5],[6,6,6],[7,7,7],
				[8,8,8],[9,9,9],[10,10,10],[11,11,11]]
		"""
		wn = []	
		for v in range(start_value,start_value+higher_layr_unit_no):
			wn.append(NP.tile(float(R.randint(0,1)),layr_unit_no))
		return wn

	def __init_biases__(self,start_value,layr_unit_no):
		"""
		The biases matrix is:
		for input layer: b0 = []
		for hid layer: b1 = [1.0,1.0,1.0]
		for output layer: b2 = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,]
		"""
		bvs = []
		for v in range(start_value,start_value+layr_unit_no):
			bvs.append(float(R.randint(0,1)))
		return bvs

	def train_NN(self,l):
		for r in range(0,10000):
			for i,X in enumerate(self.train_set):
				self.update_inputs_layr(X)
				self.feed_forward(X)
				self.back_propagate()
			if r%1000 == 0:
				print "*********************************************"
				print "The w_list after round",r,"is"
				print self.w_list
				print "The biases after round",r,"is"
				print self.biases
		return self.w_list

	def test_NN(self,inputs):
		#####get hiden value
		wn = self.w_list[0]
		hid_inp = []
		hid_out = []
		for i in range(0,3):
			wn_i = wn[i]
			hid_i = 0.0
			for j,inp in enumerate(inputs):
				hid_i += wn[i][j]*inp
			hid_i += self.biases[1][i]
			hid_inp.append(hid_i)
			hid_out.append(self.func(hid_i))
		#####get output value
		wn = self.w_list[1]
		out_inp = []
		out_out = []
		for i in range(0,8):
			wn_i = wn[i]
			out_i = 0.0
			for j,o in enumerate(hid_out):
				out_i += wn[i][j]*o
			out_i += self.biases[2][i]
			out_inp.append(out_i)
			result = 1.0 if self.func(out_i) >= T else 0.0
			out_out.append(result)
		int_inputs = [int(inp) for inp in inputs]
		int_outs = [int(out) for out in out_out]
		print "".join([str(i) for i in int_inputs]),"-->","  ".join([str(i) for i in hid_out]),"-->","".join([str(i) for i in int_outs])
		
	def update_inputs_layr(self,X):
		"""
		Updates input layer's input and output layer's target
		"""
		for i,unit_i in enumerate(self.layers[INPUT_LAYR_INDEX].units):
			unit_i.I = X[i]
			self.layers[OUTPUT_LAYR_INDEX].units[i].t = X[i]

	def feed_forward(self,X):
		count = 0
		for unit_j in self.layers[INPUT_LAYR_INDEX].units:
			unit_j.O = unit_j.I
			count += 0
		for n,cur_layr in enumerate(self.layers):
			if n == 0:
				continue
			prev_layr = self.layers[n-1]
			wn = self.w_list[n-1]
			biases = self.biases[n]
			for j,unit_j in enumerate(cur_layr.units):
				a = 0.0
				for i,unit_i in enumerate(prev_layr.units):
					a += wn[j][i]*unit_i.O
				unit_j.I = a+biases[j]
				unit_j.O = self.func(unit_j.I)
		#######

	def back_propagate(self):
		for unit_j in self.layers[OUTPUT_LAYR_INDEX].units:
			unit_j.Err = unit_j.O*(1.0-unit_j.O)*(unit_j.t-unit_j.O)
		higher_layr = self.layers[OUTPUT_LAYR_INDEX]
		wn = self.w_list[HID_LAYR_INDEX]
		for j,unit_j in enumerate(self.layers[HID_LAYR_INDEX].units):
			err_gram = 0.0
			for k,unit_k in enumerate(higher_layr.units):
				err_gram += unit_k.Err*wn[k][j]
			unit_j.Err = unit_j.O*(1.0-unit_j.O)*err_gram
		for n,wn in enumerate(self.w_list):
			l_layr = self.layers[n]
			h_layr = self.layers[n+1]
			for j,unit_j in enumerate(h_layr.units):
				for i,unit_i in enumerate(l_layr.units):
					delta_wnij = l*unit_j.Err*unit_i.O
					self.w_list[n][j][i] += delta_wnij
		for n,bn in enumerate(self.biases):
			if n == 0:
				continue
			layr = self.layers[n]
			for j,bn_j in enumerate(bn):
				delta_b = l*layr.units[j].Err
				self.biases[n][j] += delta_b

class layer():
	"""
	Each layer contains its original inputs, weights, weighted inputs and
	outputs
	"""
	def __init__(self,inputs,**kwargs):
		"""
		Parameters:
		1) targets: The targets of each unit in this layer
		"""
		biased_inputs = inputs
		if "targets" in kwargs:
			self.units = self.__generate_unit_list__(inputs,kwargs["targets"])
		else:
			self.units = self.__generate_unit_list__(inputs)

	def __generate_unit_list__(self,inputs):
		unit_list = []
		for inp in inputs:
			unit_list.append(unit(inp))
		return unit_list

	def __generate_unit_list_wtarget__(self,inputs,targets):
		unit_list = []
		for i,inp in enumerate(inputs):
			unit_list.add(unit(inp,target=targets[i]))

class unit():
	"""
	Each node in the network
	"""
	def __init__(self,inp,**kwargs):
		"""
		Parameters:
		1) target : The target value of the unit
		"""
		if "target" in kwargs:
			self.t = kwargs[target]
		self.I = inp
		self.Err = 0.0
		self.O = 0.0

	def update_target(self,new_t):
		self.t = new_t

def init_weights():
	w_list = []
	net1_w,net2_w = [[],[]]	# For net1 and net2
	node_w = []
	for j in range(1,hide_no+1):
		node_w = NP.tile(j,input_no+1)
		net1_w.append(node_w)
	w_list.append(net1_w)
	node_w = []
	for k in range(1,output_no+1):
		node_w = NP.tile(k,hide_no+1)
		net2_w.append(node_w)
	w_list.append(net2_w)
	return w_list

def sigmoid(z):
	"""
	g(z) = 1.0/(1.0+e^(-z))
	"""
	out = 1.0/(1.0+M.pow(M.e,-z))
	return 1.0/(1.0+M.pow(M.e,-z))


run(train_set)

#!/usr/bin/env python

"""
-------------------------------------------------------------------------------
About: EM on a mixture of several Gaussian
Author: Ruiyu Xu
Oct. 2014
-------------------------------------------------------------------------------
"""

import numpy as NP
import math as MA
import random as R
import sys

K = 0	# The number of components
N = 0	# The number of data set in training sample Y = Y1,Y2,...,Y3
M = 100	# Total number of round
d = 2	# Dimension of data point
converge_cond =0.00001 

def EM(data_file,model_no):
	global N,K
	Y_list = get_var_list(data_file)
	N = len(Y_list)
	K = model_no
	W,MIU,SIGMA = Initialization(Y_list)
	nj_list = []
	llh = cal_llh(Y_list,W,MIU,SIGMA)
	print ">>>>>>>>>>>>>>>>>>>>>>>>The Initials are"
	printinfo(W,MIU,SIGMA,llh)
	for m in range(0,M):
		print "*********************After Iteration",m
		gamaij_list,nj_list = Expectation(Y_list,W,MIU,SIGMA)
		W,MIU,SIGMA = Maximization(Y_list,gamaij_list,nj_list)
		new_llh = cal_llh(Y_list,W,MIU,SIGMA)
		printinfo(W,MIU,SIGMA,new_llh)
		if converged(llh,new_llh):
			break
		else:
			llh = new_llh
	print "\n"
	print "The converge condition is",converge_cond
	print ">>>>>>>>>>>>>>>>>>>>>>>>>The Final result is"
	printinfo(W,MIU,SIGMA,llh)

def printinfo(W,MIU,SIGMA,llh):
	for j in range(0,K):
		mean_name = "mean_"+str(j+1)
		print mean_name+" =",MIU[j]
		cov_name = "cov_"+str(j+1)
		print cov_name+" =",SIGMA[j]	
		n_name = "n_"+str(j+1)
		print n_name+" =",W[j]*N
		print "-------------------"
	print "Loglikelihood =",llh

def converged(llh,new_llh):
	diff = abs(llh - new_llh)
	if diff < converge_cond:
		return True
	else:
		return False

def cal_llh(Y_list,W,MIU,SIGMA):
	DATA_SUM = 0.0
	for i in range(0,N):
		MODL_SUM = 0.0
		Yi = Y_list[i]
		for j in range(0,K):
			wj = W[j]
			uj = MIU[j]
			sigmaj = SIGMA[j]
			phid = MGaussian(Yi,uj,sigmaj)
			prod = wj*phid
			MODL_SUM += prod
		DATA_SUM += MA.log(MODL_SUM,2)
	llh = DATA_SUM/N
	return llh

def Expectation(Y_list,W,MIU,SIGMA):
	nj_list = []
	gamaij_list = []
	for i in range(0,N):
		gamaj_list = []
		Yi = Y_list[i]
		weightedProb_list = []
		for j in range(0,K):
			wj = W[j]
			uj = MIU[j]
			sigmaj = SIGMA[j]
			Prob = MGaussian(Yi,uj,sigmaj)
			weightedProb = wj*Prob
			weightedProb_list.append(weightedProb)
		weightedProb_sum = sum(weightedProb_list)
		for j in range(0,K):
			gamaj = weightedProb_list[j]/weightedProb_sum
			gamaj_list.append(gamaj)
		gamaij_list.append(gamaj_list)
	return gamaij_list,cal_nj_list(gamaij_list)

def cal_nj_list(gamaij_list):
	nj_list = []
	for j in range(0,K):
		nj = 0.0
		for i in range(0,N):
			nj += gamaij_list[i][j]
		nj_list.append(nj)
	return nj_list

def Maximization(Y_list,gamaij_list,nj_list):
	W = []
	MIU = []
	SIGMA = []
	for j in range(0,K):
		#====== calculate weights
		wj = nj_list[j]/N
		W.append(wj)
		#====== calculate means
		miu_sum = gamaij_list[0][j]*NP.array(Y_list[0])
		nj = nj_list[j]
		for i in range(1,N):
			gamaij = gamaij_list[i][j]
			Yi = Y_list[i]
			prod = gamaij*NP.array(Yi)
			miu_sum += prod
		uj = miu_sum/nj
		MIU.append(uj)
		#====== calculate sigmas
		diff = NP.matrix(Y_list[0]-uj)
		gamaij = gamaij_list[0][j]
		sigmaSUM = gamaij*diff.T*diff
		for i in range(1,N):
			diff = NP.matrix(Y_list[i]-uj)
			gamaij = gamaij_list[i][j]
			sigmaSUM += gamaij*diff.T*diff
		SIGMAj = sigmaSUM/nj
		SIGMA.append(SIGMAj)
	return W,MIU,SIGMA

def MGaussian(Y,u,sigma):
	"""
	Density of a d-dimensional multivariate Gaussian
	"""
	y_u = NP.matrix(Y-u)
	sigma = NP.matrix(sigma)
	power = -0.5*y_u*sigma.I*y_u.T
	dete = NP.linalg.det(sigma)
	denom = MA.sqrt(MA.pow(2*MA.pi,d)*dete)
	phid = MA.pow(MA.e,power)/denom
	return phid

def Initialization(Y_list):
	"""
	Initialize W, MIUs and SIGMA
	"""
	init_gamaij_list = []
	for i in range(0,N):
		gamaj_list = []
		for j in range(0,K):
			gamaj_list.append(float(R.randint(0,1)))
		init_gamaij_list.append(gamaj_list)
	init_nj_list = cal_nj_list(init_gamaij_list)
	init_W,init_MIU,init_SIGMA = Maximization(Y_list,	\
				init_gamaij_list,init_nj_list)
	return init_W,init_MIU,init_SIGMA
	
def get_var_list(data_file):
	"""
	Get Y list
	"""
	Y_list = []
	with open(data_file,"r") as f_data:
		all_lines = f_data.readlines()
		for i,line in enumerate(all_lines):
			line = line.strip("\n\r")
			splitline = line.split()
			split_line = [float(s) for s in line.split() if s]
			if not len(split_line) == d:
				continue
			else:
				Y_list.append([float(y) for y in split_line])
		i_list = []
	f_data.close()
	return Y_list

model_no = int(sys.argv[1])
filename = str(model_no)+"gaussian.txt"
EM(filename,model_no)

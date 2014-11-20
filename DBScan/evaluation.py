#!/usr/bin/python

# Time : Nov. 2013
# Author: Ruiyu Xu

import math as M
import operator


def purity(clusters,N):
    total = 0
    for cls in clusters:
	countDict ={}
	for data in cls:
	    if countDict.has_key(data.cls):
		countDict[data.cls] += 1
	    else:
		countDict[data.cls] = 1
	sortMajority = sorted(countDict.iteritems(),key = operator.itemgetter(1),reverse=True)
	total += sortMajority[0][1]
    return float(total)/float(N)

def NMI(clusters,dataset):
    N = float(len(dataset))
    groundTruthDict = {}    # record the data of each ground truth
    
    for data in dataset:
	if groundTruthDict.has_key(data.cls):
	    dataList = groundTruthDict[data.cls]
	    dataList.append(data)
	    groundTruthDict[data.cls] = dataList
	else:
	    groundTruthDict[data.cls] = [data]
    IWC = 0.0
    HW = 0.0
    HC = 0.0

    for label,dataList in groundTruthDict.iteritems():
	for cj in clusters:
	    wkANDcj = 0.0
	    for data in cj:
		if label == data.cls:
		    wkANDcj += 1.0
	    if wkANDcj > 0.0:
	    	IWC += (wkANDcj/N)*M.log10((N*wkANDcj)/float(len(dataList)*len(cj)))
	HW += (float(len(dataList))/N)*M.log10(float(len(dataList))/N)
    print HW
    for cj in clusters:
	HC += (float(len(cj))/N)*M.log10(float(len(cj))/N)
    print HC
    print IWC
    return IWC/M.sqrt((-HW)*(-HC))	

def RI(clusters,dataset):
    TP = 0.0
    FP = 0.0
    P = 0.0
    TN = 0.0
    FN = 0.0
    N = 0.0
    
    # calculate True Positive and All Positive
    for cls in clusters:
	clsLen = len(cls)
	if clsLen > 1:
	    # calculate True Positive
	    countDict = {}
	    for data in cls:
	    	if countDict.has_key(data.cls):
		    countDict[data.cls] += 1
	    	else:
		    countDict[data.cls] = 1
	    for key,value in countDict.iteritems():
	    	if value >1:
	    	    TP += combination(value,2)
	
	    # calculate All Positive
	    P += combination(clsLen,2)
    # calculate False Positive
    FP = P - TP
    print "TP is:", TP
    print "FP is:", FP
    print "P is:", P
    
    # calculate False Negative and All Negative
    clustersLen = len(clusters)
    for i in xrange(0,clustersLen-1):
	clsiCapacity = len(clusters[i])
	classCountDict = {}
	for data in clusters[i]:
	    if classCountDict.has_key(data.cls):
		classCountDict[data.cls] += 1
	    else:
		classCountDict[data.cls] = 1
	#print classCountDict
	for j in xrange(i+1, clustersLen):
	    clsjCapacity = len(clusters[j])
	    N += clsiCapacity * clsjCapacity
	    for data in clusters[j]:
		if classCountDict.has_key(data.cls):
		    FN += classCountDict[data.cls]
	
    # calculate True Negative
    TN = N - FN
    print "TN is:", TN
    print "FN is:", FN
    print "N is:", N

    # calculate F-measure
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    return (2*precision*recall)/(precision+recall)

    # calculate RI
    #return  (TP+TN)/(TP+FP+FN+TN)


def combination(n,r):
    return M.factorial(n)/(M.factorial(r)*M.factorial(n-r)) 

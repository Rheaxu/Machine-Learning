#!/usr/bin/python

# Time: Nov. 2013
# Author: RUiyu Xu

import dbscan,evaluation
import copy
import sys

cls_idx = 2

dataset = []


with open('uniqueBumpData.txt') as data_f:
    for l in data_f.readlines():
	r = l.strip().split(',')
	for i in xrange(1, cls_idx+1):
	    r[i] = float(r[i])
	dataset.append(dbscan.data(r,cls_idx))
    data_f.close()

mins = [copy.deepcopy(min(dataset,key=lambda d:d.tuple[i])) for i in xrange(0,cls_idx)]
maxs = [copy.deepcopy(max(dataset,key=lambda d:d.tuple[i])) for i in xrange(0,cls_idx)]

f_norm = open("normData.txt",'w')
for d in dataset:
    for i in xrange(0,cls_idx):
	d.tuple[i] = float(d.tuple[i] - mins[i].tuple[i])/(maxs[i].tuple[i]-mins[i].tuple[i])
    f_norm.write(str(d.tuple))
    f_norm.write('\n')


k = int(sys.argv[1])
eps = float(sys.argv[2])

cluster = dbscan.dbscan(dataset,eps,k)
if len(cluster) == 0:
    print 'k:',k,'no. of cluster:',len(cluster)
    print

pure = evaluation.purity(cluster,len(dataset))
NMI = evaluation.NMI(cluster,dataset)
RI = evaluation.RI(cluster,dataset)

cp = [len(c) for c in cluster]

f_out = open("output.txt",'w')

for i in xrange(0,len(cluster)):
    print 'cluster:',i,'no. of pt. in cluster:', cp[i]
    for c in cluster[i]:
	s = str(i) + " " + str(c.cls)
	f_out.write(s)
	f_out.write('\n')

f_out.close()
print 'purity is:',pure
print 'NMI is:',NMI
print 'RI is:',RI
print

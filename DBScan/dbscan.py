#!/usr/bin/python

# Author: Ruiyu Xu
# Main part of dbscan
# Time : Nov. 2013

import proximity
import math

cached = False

class data:
    ''' data wrapper '''
    CORE, BORDER, NOISE = 0, 1, 2

    def __init__(self,tuple=None, target_cls_idx=None):
	if tuple is not None and target_cls_idx is not None:
	    self.cls = tuple[0]
	    self.tuple = tuple[1:]
	    self.label = data.NOISE
	    self.visited = False
	    self.belongToCluster = False

    def reset(self):
	self.label = data.NOISE
	self.visited = False


def find_neighbour(instance, dataset, radius, distance):
    ''' find all neighbour within radius '''
    global cached
    if not cached:
	proximity.build_cache(dataset, distance)
	cached = True

    pairs = proximity.cache[instance]
    neighbour = [which for which, dist in pairs if radius >= dist]

    return neighbour


def dbscan(dataset, radius, minPt, distance = proximity.euclidean):
    ''' dataset is a list of data wrapper '''
    cluster = []
    clusteredData = 0
    map(lambda d: d.reset(), dataset)
    noiseList = []

    for instance in dataset:
	if instance.visited == True: continue

	instance.visited = True
	neighbour = find_neighbour(instance, dataset, radius, distance)

#	if len(neighbour)+1 >= minPt:
#	    c = [instance]
#	    instance.belongToCluster = True
#	    N = neighbour[:]
#	    while len(N) >0:
#		pp = N.pop()
#		if pp.visited == False:
#		    pp.visited = True
#		    pp_neighbour = find_neighbour(instance,dataset,radius,distance)
#		    if len(pp_neighbour)+1 >= minPt:
#			N.extend(pp_neighbour)
#		if pp.belongToCluster == False and pp.label!=data.NOISE:
#		    pp.belongToCluster = True
#		    c.append(pp)
#	    cluster.append(c)
#	    clusteredData += len(c)
#	else:
#	    instance.label = data.NOISE
#	    noiseList.append(instance)
#    print "Total number of noise:",len(noiseList)
#    print "Total number of non-noise:",clusteredData    
#
#    # regard each noise as a cluster
#    for noisedata in noiseList:
#    	cluster.append([noisedata])
#    print "Number of clusters:",len(cluster)
			

	if len(neighbour)+1 < minPt:
	    instance.label = data.NOISE
	else:
	    c = neighbour[:] + [instance]
	    q = neighbour[:]
	    cluster.append(c)


	    while len(q) > 0:
		check_instance = q.pop()
		neighbour = find_neighbour(check_instance,dataset,radius,distance)
		
		check_instance.visited = True
		if len(neighbour)+1 < minPt:
		#if minPt > len(neighbour) + 1:
		    check_instance.label = data.BORDER
		else:
		    check_instance.label = data.CORE
		    for n in neighbour:
			if n not in c:
			    c.append(n)
			    q.append(n)

    return cluster

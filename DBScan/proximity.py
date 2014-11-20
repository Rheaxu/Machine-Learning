#!/usr/bin/python

# Author: Ruiyu Xu
# Calculate Distance
# Time: Nov. 2013

import math as M

def euclidean(a,b):
    '''L2 distance without square root'''
    dif = [(aa-bb) ** 2 for aa,bb in zip(a,b)]
    return M.sqrt(sum(dif))


cache = {}
def build_cache(dataset,distance=euclidean):
    ''' build distance cache.
	each vector will compute the distance with others distance
	from others are sorted ASC '''
    global cache
    for a in dataset:
	pairs = [(b,distance(a.tuple, b.tuple)) for b in dataset if a != b]
	cache[a] = sorted(pairs, key=lambda p: p[1])

#!/usr/bin/env python

"""
-------------------------------------------------------------------------------
About: HAAR features for Digits Dataset
Author: Ruiyu Xu
Time: Nov, 2014
-------------------------------------------------------------------------------
"""

from array import array as pyarray
import numpy as np
from mnist import load_mnist
import random as R
import math as m


img_size = 28
rows = 28
cols = 28
rec_no = 100
min_area = 130
max_area = 170
min_height = m.ceil(float(min_area)/cols)
recs = []


def sample_data():
	generate_recs()
	print "Finished generating 100 rectangles"
	train_data,train_labels,test_data,test_labels = [[],[],[],[]]
	X_list,test_X_list,y_list,test_y_list = [[],[],[],[]]
	all_train_images,all_train_labels = load_mnist('training')
	all_test_images,all_test_labels = load_mnist('testing')
	sep_train_list = [[] for i in range(10)] #train data list based on class
	for i,img in enumerate(all_train_images):
		sep_train_list[all_train_labels[i][0]].append(img)
	for i,class_img in enumerate(sep_train_list):
		class_len = len(class_img)
		per20 = int(0.2*class_len)
		train_data.extend(class_img[:per20])
		ys = [i for n in range(per20)]
		train_labels.extend(ys)
		y_list.extend(ys)
	train_data = np.array(train_data)
	train_labels = np.array(train_labels)
	X_list = generate_attr(train_data)
	print "Finished processing train data"
	for i,img in enumerate(all_test_images):
		test_data.append(img)
		test_labels.append(all_test_labels[i][0])
		test_y_list.append(all_test_labels[i][0])
	test_data = np.array(test_data)
	test_labels = np.array(test_labels)
	test_X_list = generate_attr(test_data)
	print "Finished processing test data"
	return X_list,y_list,test_X_list,test_y_list

def generate_attr(img_data):
	X = []
	for img in img_data:
		Ocornered_rec = cal_Ocornered_rec(img)
		entry = generate_ftr(Ocornered_rec)
		X.append(entry)
	return X
	
def generate_ftr(Ocornered_rec):
	X = []
	for rec in recs:
		v = cal_vertical_ftr(rec,Ocornered_rec)
		h = cal_horizontal_ftr(rec,Ocornered_rec)
		X.append(v)
		X.append(h)
	return X

def cal_vertical_ftr(rec,Ocornered_rec):
	x1,y1 = rec[0]
	x2,y2 = rec[1]
	midx = x1+((x2-x1)/2)
	black1 = cal_black((x1,y1),(midx,y2),Ocornered_rec)
	black2 = cal_black((midx,y1),(x2,y2),Ocornered_rec)
	ver_ftr = black1-black2
	return ver_ftr

def cal_horizontal_ftr(rec,Ocornered_rec):
	x1,y1 = rec[0]
	x2,y2 = rec[1]
	midy = y1+((y2-y1)/2)
	black1 = cal_black((x1,y1),(x2,midy),Ocornered_rec)
	black2 = cal_black((x1,midy),(x2,y2),Ocornered_rec)
	h_ftr = black1-black2
	return h_ftr

def cal_black(p1,p2,Ocornered_rec):
	"""
	Calculates black for a rectangle
	black(rec(p1,p2)) = black1-black2-black3+black4
	black1 = Ocornered_rec[x2][y2]
	black2 = Ocornered_rec[x1][y2]
	black3 = Ocornered_rec[x2][y1]
	black4 = Ocornered_rec[x1][y1]
	"""
	black1 = Ocornered_rec[p2[0]][p2[1]]
	black2 = Ocornered_rec[p1[0]][p2[1]]
	black3 = Ocornered_rec[p2[0]][p1[1]]
	black4 = Ocornered_rec[p1[0]][p1[1]]
	black = black1-black2-black3+black4
	return black

def generate_recs():
	"""
	A rectangle is a 2-D tuple of 2-D tuples, ((x1,y1),(x2,y2)), which
	represents the diagonal of the rectangle
	"""
	global recs
	r = 1
	count = 0
	while r <= rec_no:
		count += 1
		p1x = R.randint(0,cols-min_height)
		p2x = R.randint(p1x+min_height-1,cols-1)
		height = p2x-p1x+1
		min_width = int(m.ceil(float(min_area)/height))
		max_width = max_area/height
		p1y = R.randint(0,rows-min_width)
		p2y = R.randint(p1y+min_width-1,min(rows-1,p1y+max_width-1))
		rec = ((p1x,p1y),(p2x,p2y))
		if not rec in recs:
			recs.append(rec)
			r += 1
			area = (p2x-p1x+1)*(p2y-p1y+1)
			if area > max_area or area < min_area:
				print "rec",rec
				print "width,height",p2y-p1y+1,p2x-p1x+1
				print "covered area:",area
		
def cal_Ocornered_rec(img):
	Ocornered_rec = np.zeros((img_size,img_size))
	
	# Initialize Ocornered_rec
	Ocornered_rec[0][0] = img[0][0] 
	for j in range(1,cols):
		Ocornered_rec[0][j] = Ocornered_rec[0][j-1] + img[0][j]
	for i in range(1,rows):
		Ocornered_rec[i][0] = Ocornered_rec[i-1][0] + img[i][0]

	# Dynamically compute O-cornered rectangle
	for i in range(1,rows):
		for j in range(1,cols):
			Ocornered_rec[i][j] = Ocornered_rec[i][j-1]+	\
				Ocornered_rec[i-1][j]-Ocornered_rec[i-1][j-1]+img[i][j]
	return Ocornered_rec
	



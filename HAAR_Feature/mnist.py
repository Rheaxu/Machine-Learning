import os, struct
import numpy as np
from array import array as pyarray 
from numpy import append, array, int8, uint8, zeros

def load_mnist(dataset="training", digits=np.arange(10), path='image_data'):
	"""
	Loads MNIST files into a 3D numpy array.
	"""

	if dataset == "training":
		fname_img = os.path.join(path,'train-images-idx3-ubyte') 
		fname_lbl = os.path.join(path,'train-labels-idx1-ubyte')
	elif dataset == "testing":
		fname_img = os.path.join(path,'t10k-images-idx3-ubyte')
		fname_lbl = os.path.join(path,'t10k-labels-idx1-ubyte')
	else:
		raise ValueError("dataset must be 'testing' or 'training'")

	flbl = open(fname_lbl, 'rb')
	magic_nr, size = struct.unpack(">II", flbl.read(8))
	lbl = pyarray("b", flbl.read())
	flbl.close()

	fimg = open(fname_img, 'rb')
	magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
	img = pyarray("B", fimg.read())
	fimg.close()

	indices = [k for k in range(size) if lbl[k] in digits]
	N = len(indices)

	images = zeros((N, rows, cols), dtype=uint8)
	labels = zeros((N, 1), dtype=int8)
	for i, ind in enumerate(indices):
		images[i] = array(img[ind*rows*cols:(ind+1)*rows*cols]).reshape((rows, cols))
		labels[i] = lbl[ind]

	return images,labels

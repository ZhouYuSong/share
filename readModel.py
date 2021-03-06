import types
import numpy as np
import matplotlib.pyplot as plt
caffe_root='../'
import sys
sys.path.insert(0,caffe_root+'python')
import caffe
model_def=caffe_root+'examples/mnist/deploy.prototxt'
model_caffe=caffe_root+'examples/mnist/lenet_iter_10000.caffemodel'
net=caffe.Net(model_def,model_caffe,caffe.TEST)
print [(k, v[0].data.shape) for k, v in net.params.items()]
#print net.params['conv1'][0].data
idx=net.params['conv1'][0].data.shape
for  n_idx in range(0,idx[0]):
	for  c_idx in range(0,idx[1]):
		for  h_idx in range(0,idx[2]):
			for w_idx in range(0,idx[3]):
				if  net.params['conv1'][0].data[n_idx][c_idx][h_idx][w_idx]< 0.05:
					 net.params['conv1'][0].data[n_idx][c_idx][h_idx][w_idx]=0
 					 net.params['conv1'][0].mask[n_idx][c_idx][h_idx][w_idx]=0					
print net.params['conv1'][0].data
print net.params['conv1'][0].mask
net.save('fixed.caffemodel')



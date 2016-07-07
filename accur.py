import numpy as np
import matplotlib.pyplot as plt

import caffe
import cv2
from xml.dom.minidom import parse
import xml.dom.minidom
caffe_root='/home/song/data/caffe/'
img_root='/home/song/data/voc0712/VOCdevkit/VOC2007/'
import os
import sys


os.chdir(caffe_root)
sys.path.insert(0,caffe_root+'python')
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
caffe.set_mode_gpu()
net=caffe.Net(caffe_root+'models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt',
             caffe_root+'models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel',
              caffe.TEST)
im = caffe.io.load_image('/home/song/data/voc0712/VOCdevkit/VOC2007/JPEGImages/000002.jpg')
net.forward()

mbox=net.blobs['mbox_priorbox'].data[0][0]
idx=0
print mbox.shape
total=(mbox.shape[0])/4

domtree=xml.dom.minidom.parse("/home/song/data/voc0712/VOCdevkit/VOC2007/Annotations/000002.xml")
data=domtree.documentElement
bbox=data.getElementsByTagName('bndbox')
weight=data.getElementsByTagName('width')[0].childNodes[0].data
height=data.getElementsByTagName('height')[0].childNodes[0].nodeValue
print weight,height

throld=0.1
cnt=0
recall=0
error=0
mis=0
print total
for i in range(total):
    loc=mbox[idx:idx+4]
    idx=idx+4
    for bb in bbox:
        
        xmin=bb.getElementsByTagName('xmin')
        ymin=bb.getElementsByTagName('ymin')
        xmax=bb.getElementsByTagName('xmax')
        ymax=bb.getElementsByTagName('ymax')
        x1=float(xmin[0].childNodes[0].data)
        y1=float(ymin[0].childNodes[0].data)
        x2=float(xmax[0].childNodes[0].data)
        y2=float(ymax[0].childNodes[0].data)
        insideX1=max(x1,(loc[0]*float(weight)))
        insideY1=max(y1,(loc[1]*float(height)))
        insideX2=min(x2,(loc[2]*float(weight)))
        insideY2=min(y2,(loc[3]*float(height)))
        outsideX1=min(x1,(loc[0]*float(weight)))
        outsideY1=min(y1,(loc[1]*float(height)))
        outsideX2=max(x2,(loc[2]*float(weight)))
        outsideY2=max(y2,(loc[3]*float(height)))
        if insideX1>=insideX2 or insideY1>=insideY2:
            mis=mis+1
            continue
        inside=(insideX2-insideX1)*(insideY2-insideY1)
        
        outside=(loc[3]-loc[1])*(loc[2]-loc[0])+(x2-x1)*(y2-y1)-inside
        if inside >= outside:
            erro=error+1
        if inside/outside >0.99999999:
            recall=recall+1
        if inside/outside >0.1:
            cnt=cnt+1
            break

print cnt
print recall
print error
print mis
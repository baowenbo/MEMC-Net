import sys
import os
import time
import random
import subprocess as sp
# import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave, imshow, imresize, imsave
from skimage import color


def readFlowFile(filename):
    TAG_FLOAT = 202021.25; # check    for this when READING the file
    if filename == None:
        raise("file name not specified ")

    idx = filename.split('.')
    idx = idx[-1] #in case './xxx/xxx.flo'

    if not idx == 'flo':
        print("extension is " + idx)
        raise("wrong flo extension")

    f = open(filename,'r')

    tag = np.fromfile(f,np.float32, count=1)
    w = np.fromfile(f,np.int32, count=1)
    h = np.fromfile(f,np.int32, count=1)

    if not tag == TAG_FLOAT:
        raise ("readFlowFile(%s): wrong tag (possibly due to big-endian machine?)"%filename)

    if w < 1 or w > 99999 :
        raise ("width error")
    if h < 1 or h > 99999:
        raise ("width error")

    nBands = 2

    temp = np.fromfile(f, np.float32, count=-1)
    # temp = temp.reshape([w * 2 ,h])
    # temp = np.transpose(temp,(1,0))
    # temp = temp.reshape([h,w*2])
    temp = np.reshape(temp,[h,w*nBands])

    flow = np.zeros([h,w,nBands])
    flow[:,:,0] = temp[:,0::nBands]
    flow[:,:,1] = temp[:,1::nBands]

    f.close()
    flow = np.float32(flow)
    return flow

if __name__ == '__main__':
    flow = readFlowFile('frame_0001.flo')






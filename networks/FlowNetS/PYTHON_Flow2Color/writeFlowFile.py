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

from multiprocessing import Pool, cpu_count
from functools import partial


def writeFlowFile(flow, filename):
    TAG_STRING = "PIEH"

    if filename == None:
        raise("file name not specified ")

    idx = filename.split('.')
    idx = idx[-1] #in case './xxx/xxx.flo'

    if not idx == 'flo':
        print("extension is " + idx)
        raise("wrong flo extension")

    h = flow.shape[0]
    w = flow.shape[1]
    c = flow.shape[2]

    if not c == 2:
        raise ("wrong u,v channels")

    f = open(filename,'w')

    f.write("%s" % TAG_STRING)
    np.array(w,dtype=np.uint32).tofile(f)
    np.array(h,dtype=np.uint32).tofile(f)


    temp = np.zeros([h, w * c])
    temp[:, 0::2] = flow[:,:,0]
    temp[:, 1::2] = flow[:,:,1]
    # temp = np.transpose(temp,(1,0)) #

    temp = np.float32(temp)
    temp = temp.flatten()

    temp.tofile(f) #,format = '%f')
    f.close()




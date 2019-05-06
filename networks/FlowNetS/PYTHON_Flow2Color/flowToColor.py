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

from .computeColor import computeColor

def flowToColor(flow, maxFlow = None):
    UNKNOWN_FLOW_THRESH = 1e9
    UNKNOWN_FLOW = 1e10

    height = flow.shape[0]
    width = flow.shape[1]
    nBands = flow.shape[2]

    if not nBands == 2:
        raise('flowToColor: image must have two bands')

    u = flow[:,:, 0]
    v = flow[:,:, 1]
    
    maxu = -999 
    maxv = -999 
    
    minu = 999 
    minv = 999 
    maxrad = -1

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) >UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))
    
    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))
    
    rad = np.sqrt(u * u + v * v)
    maxrad = max(maxrad, np.max(rad))
    
    print("max flow: %.4f flow range: u = %.3f .. %.3f  v = %.3f .. %.3f" % (maxrad, minu, maxu, minv, maxv))

    if not maxFlow == None:
        if maxFlow > 0:
            maxrad = maxFlow
        elif maxFlow < 0:
            maxrad  = max(maxrad, -maxFlow) # to allow a m
            print("setting maxflow of color to " + str(maxrad))


    eps = 1e-6

    u = u/(maxrad + eps)
    v = v/(maxrad + eps)

    img = computeColor(u,v)

    IDX = np.stack([idxUnknow,idxUnknow,idxUnknow],axis = 2)

    img[IDX] = 0

    return img



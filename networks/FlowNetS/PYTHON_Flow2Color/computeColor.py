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

def makeColorwheel():
    RY = 15 
    YG = 6 
    GC = 4 
    CB = 11 
    BM = 13 
    MR = 6 

    ncols = RY + YG + GC + CB + BM + MR 

    colorwheel = np.zeros([ncols, 3])  # r     g     b

    col = 0 
    # RY
    colorwheel[0: RY, 0] = 255
    colorwheel[0: RY, 1] = np.floor(255 * np.arange(0,RY) / RY)
    col = col + RY 

    # YG
    colorwheel[col : col + YG, 0] = 255 -  np.floor(255 * np.arange(0,YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col = col + YG 

    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] =  np.floor(255 * np.arange(0, GC ) / GC)
    col = col + GC 

    # CB
    colorwheel[col : col + CB, 1] = 255 -  np.floor(255 * np.arange(0, CB ) / CB)
    colorwheel[col : col + CB, 2] = 255
    col = col + CB 

    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] =  np.floor(255 * np.arange(0, BM ) / BM)
    col = col + BM 

    # MR
    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(0, MR ) / MR)
    colorwheel[col : col + MR, 0] = 255
    
    return colorwheel

def computeColor(u,v):
    nanIdx = np.isnan(u) | np.isnan(v)

    u[nanIdx] = 0
    v[nanIdx] = 0
    colorwheel = makeColorwheel()

    ncols = colorwheel.shape[0]

    rad = np.sqrt(u *u + v*v)

    a = np.arctan2(-v,-u)/np.pi
    fk = (a+1) / 2 * (ncols - 1)
    k0 = np.int32(np.floor(fk))
    k1 = k0 + 1
    k1[k1==ncols] = 0

    f = fk - k0

    img = np.zeros([u.shape[0],u.shape[1],3])

    for i in range(colorwheel.shape[1]):
        temp = colorwheel[:,i]
        col0 = temp[k0] / 255
        col1 = temp[k1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <=1
        col[idx] = 1 - rad[idx] * ( 1-col[idx])

        idx = rad > 1
        col[idx] = col[idx] *0.75

        img[:,:,i] = np.uint8(np.floor(255*col *(1-nanIdx)).clip(0.0,255.0))
    return img




if __name__ == '__main__':
    cw = makeColorwheel()


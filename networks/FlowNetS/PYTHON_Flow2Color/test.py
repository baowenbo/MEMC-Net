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

from PYTHON_Flow2Color.readFlowFile import readFlowFile
from PYTHON_Flow2Color.writeFlowFile import writeFlowFile
from PYTHON_Flow2Color.flowToColor import flowToColor


flow = readFlowFile('frame_0001.flo')
img = flowToColor(flow)

img_ref = imread("a.png")

plt.figure(1)
plt.title("my Frame")
plt.imshow(img.astype("uint8"))
plt.show()
plt.figure(2)
plt.title("ref Frame")
plt.imshow(img_ref.astype("uint8"))
plt.show()

x = img- img_ref
print(np.mean(np.abs(x)))


writeFlowFile(flow,'frame_0001_my.flo')

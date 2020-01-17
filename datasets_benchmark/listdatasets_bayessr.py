import torch.utils.data as data
import os
import os.path
from scipy.ndimage import imread
import numpy as np
from skimage import color
import random
#from matlab_imresize.imresize import *
# import matlab_imresize
from skimage import img_as_float
from skimage.measure import compare_ssim,compare_psnr
 
def BayesSR_loader(root, im_path, video_name, task, task_param,
                     # DownResizer=None,UpResizer=None,
                     input_frame_size = ( 128,128,3)):

    # if task == 'denoise':
    #     root_input = os.path.join(root,'input',im_path)
    # else:
    root_input = os.path.join(root,video_name, 'bicubic')
    root_target =os.path.join(root,video_name, 'original')
    # path_y_  = os.path.join(root,'target_ours',im_path) 
    # frame_prefix = im_path
    
    path = []
    im  = []
    target = []
    cur_idex = int(im_path[-6:-4])
    for i in range(0,7):
        temp = os.path.join(root_input, im_path[:-6]+
                            str((cur_idex-3+i)).zfill(2)+im_path[-4:])
        path.append(temp)
    # if video_name == 'foliage':
    #     print("")

    for i in range(0,7):
        if os.path.exists(path[i]):
            im.append(imread(path[i]))
        else:
            im.append(imread(path[3]))
    
    target = imread(os.path.join(root_target,im_path ))

    if task == 'sr':
        # for i in range(0,7):
            # step 1: turn to low resolution
            #temp = img_as_float(im[i])
            # ratio = task_param[0]
            #temp = imresize(temp,1/ratio)
            #temp = convertDouble2Byte(temp)
            # temp = im[i]
            # step 2: preprocess for network
            # temp = img_as_float(temp)
            # temp = imresize(temp,ratio)
            # temp = UpResizer.imresize(temp)
            # temp = convertDouble2Byte(temp)
            # im[i] = temp
        print(compare_psnr(im[3],target))
    elif task=='denoise':
        # for i in range(0,7):
        #     temp = im[i]
            #sigma = task_param[0]
            #gaussian_noise = np.clip(np.random.normal(0.0, sigma * 255.0, input_frame_size), 0, 255).astype("uint8")
            #temp = np.clip(temp.astype("uint32") + gaussian_noise.astype("uint32"), 0, 255).astype("uint8")
            #b = np.clip((np.random.rand(input_frame_size[0],input_frame_size[1], input_frame_size[2]) * 255.0), 0, 255).astype("uint8")
            #p =  task_param[1]
            #tm = np.random.rand(input_frame_size[0], input_frame_size[1] ) <= p
            #tm = np.stack((tm, tm, tm), axis=2)
            #temp[tm] = b[tm]
            #im[i] = temp
        print(compare_psnr(im[3],target))
        pass
    elif task== 'deblock':
        for i in range(0,7):
            temp = img_as_float(im[i])
            raise("not implemented for deblock")
    
    for i in range(0,7):
        im[i] = np.transpose(im[i],(2,0,1))
    target= np.transpose(target,(2,0,1))


    return [im[i].astype("float32")/255.0 for i in range(0,7)], target.astype("float32")/255.0  

class ListDataset(data.Dataset):
    def __init__(self, root, path_list,videos_list,
                 task = 'sr', task_param = [4.0],  loader=BayesSR_loader): #transform=None, target_transform=None, co_transform=None,

        self.root = root
        self.videos_list = videos_list
        self.path_list = path_list
        # self.transform = transform
        # self.target_transform = target_transform
        # self.co_transform = co_transform
        self.loader = loader
        self.task = task
        self.task_param = task_param
        print("task is " + task," with parameter " )
        print(task_param)
        # if task == 'sr':
        #     self.DownResizer = matlab_imresize.Imresize((256,448,3),1/task_param[0])
        #     self.UpResizer = matlab_imresize.Imresize((256/task_param[0],448/task_param[0],3),task_param[0])
         # self.DownResizer
        # self.task_param += self.UpResizer
    def __getitem__(self, index):
        path = self.path_list[index]
        video = self.videos_list[index]
        # print(path)
        Xs,y,  = self.loader(self.root, path,video,  self.task, self.task_param)
        return Xs,y,path,video

    def __len__(self):
        return len(self.path_list)

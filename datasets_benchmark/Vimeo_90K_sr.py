import os.path
import random
import glob
import math
from .listdatasets import ListDataset,Vimeo_90K_loader


def make_dataset(root, list_file):
    raw_im_list = open(os.path.join(root, list_file)).read().splitlines()
    # the last line is invalid in test set.
    #print("The last sample is : " + raw_im_list[-1])
    #raw_im_list = raw_im_list[:-1]
    assert len(raw_im_list) > 0
    #random.shuffle(raw_im_list)

    # split_index = int(math.floor(len(raw_im_list)*split/100.0))
    # assert(split_index >= 0 and split_index <= len(raw_im_list))

    return  raw_im_list
    # return (raw_im_list[:split_index], raw_im_list[split_index:]) if split_index < len(raw_im_list) else (raw_im_list,[])

# use 1% of the samples to be a validation dataset
def Vimeo_90K_sr(root,task,task_param):
    #train_list = make_dataset(root,"sep_trainlist.txt")
    test_list = make_dataset(root,"sep_testlist.txt")
    # train_dataset = ListDataset(root, train_list,task= task, task_param=task_param, loader = Vimeo_90K_loader)
    test_dataset = ListDataset(root, test_list,task= task, task_param=task_param, loader = Vimeo_90K_loader)
    return  test_dataset
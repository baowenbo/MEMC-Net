import os.path
import random
import glob
import math
from .listdatasets_bayessr import ListDataset,BayesSR_loader


def make_dataset(root,videos):


    raw_im_list = []
    videos_list = []
    for video in  videos:
        print(video)
        temp = os.listdir(os.path.join(root,video,'original'))
        temp.sort()
        print(len(temp))
        raw_im_list += temp
        videos_list += [video] * len(temp)

    # the last line is invalid in test set.
    #print("The last sample is : " + raw_im_list[-1])
    #raw_im_list = raw_im_list[:-1]
    # assert len(raw_im_list) > 0
    #random.shuffle(raw_im_list)

    # split_index = int(math.floor(len(raw_im_list)*split/100.0))
    # assert(split_index >= 0 and split_index <= len(raw_im_list))

    return  raw_im_list,videos_list
    # return (raw_im_list[:split_index], raw_im_list[split_index:]) if split_index < len(raw_im_list) else (raw_im_list,[])

# use 1% of the samples to be a validation dataset
def BayesSR(root,task,task_param):
    videos = ['calendar', 'city', 'foliage', 'walk']
    #train_list = make_dataset(root,"sep_trainlist.txt")
    test_list,videos_list = make_dataset(root,videos)
    # train_dataset = ListDataset(root, train_list,task= task, task_param=task_param, loader = Vimeo_90K_loader)
    test_dataset = ListDataset(root, test_list,  videos_list,
                               task= task, task_param=task_param, loader = BayesSR_loader)
    return  test_dataset

if __name__ == '__main__':
    testset = BayesSR('/tmp4/wenbobao_data/BayesSRvideo','sr',[4.0])
    for (Xs,y,path,video) in testset:
        print(video)
        print(path)
# MEMC-Net (Motion Estimation and Motion Compensation Driven Neural Network for Video Interpolation and Enhancement)
[Project](https://sites.google.com/view/wenbobao/memc-net) **|** [Paper](http://arxiv.org/abs/1810.08768)


[Wenbo Bao](https://sites.google.com/view/wenbobao/home),
[Wei-Sheng Lai](http://graduatestudents.ucmerced.edu/wlai24/), 
Xiaoyun Zhang, 
Zhiyong Gao, 
and [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/)


### Table of Contents
1. [Introduction](#introduction)
1. [Citation](#citation)
1. [Requirements and Dependencies](#requirements-and-dependencies)
1. [Installation](#installation)
1. [Testing Pre-trained Models](#testing-pre-trained-models)
1. [Downloading Results](#downloading-results)
1. [Slow-motion Generation](#slow-motion-generation)
<!--1. [Training New Models](#training-new-models) -->

We propose the Motion Estimation and Motion Compensation (**MEMC**) Driven Neural Network for video frame interpolation as well several other video enhancement tasks.
A novel adaptive warping layer is proposed to integrate both optical flow and interpolation kernels to synthesize target frame pixels.
Our method benefits from the ME and MC model-driven architecture while avoiding the conventional hand-crafted design by training on a large amount of video data.
Extensive quantitative and qualitative evaluations demonstrate that the proposed method performs favorably against the state-of-the-art video frame interpolation and enhancement algorithms on a wide range of datasets.

### Citation
If you find the code and datasets useful in your research, please cite:

    @article{MEMC-Net,
         title={MEMC-Net: Motion Estimation and Motion Compensation Driven Neural Network for Video Interpolation and Enhancement},
         author={Bao, Wenbo and Lai, Wei-Sheng, and Zhang, Xiaoyun and Gao, Zhiyong and Yang, Ming-Hsuan},
         journal={arXiv preprint arXiv:1810.08768},
         year={2018}
    }
    

### Requirements and Dependencies
- Ubuntu (We test with Ubuntu = 16.04.5 LTS)
- Python (We test with Python = 3.5.4 in Anaconda3 = 4.1.1)
- Cuda & Cudnn (We test with Cuda = 9.0 and Cudnn = 7.0)
- PyTorch (The customized adaptive warping layer and other layers require cffi API in PyTorch = 0.2.0_4)
- GCC (Compiling PyTorch 0.2.0_4 extension files (.c/.cu) requires gcc = 4.8.5 and nvcc = 9.0 compilers)
- NVIDIA GPU (We use Titan X (Pascal) with compute = 6.1, but we support compute_50/52/60/61/70 devices, should you have devices with higher compute capability, please revise [this](https://github.com/baowenbo/MEMC-Net/blob/master/my_package/install.bash))


### Installation
Download repository:

    $ git clone https://github.com/baowenbo/MEMC-Net.git

Before building Pytorch extensions, be sure you have `pytorch == 0.2` :
    
    $ python -c "import torch; print(torch.__version__)"
    
Generate our PyTorch extensions:
    
    $ cd MEMC-Net
    $ cd my_package 
    $ ./install.sh

### Testing Pre-trained Models
Make model weights dir and Middlebury dataset dir:

    $ cd MEMC-Net
    $ mkdir model_weights
    $ mkdir MiddleBurySet
    
Download pretrained models, 

    $ cd model_weights
    $ wget http://vllab1.ucmerced.edu/~wenbobao/MEMC-Net/MEMC-Net_best.pth 
    $ wget http://vllab1.ucmerced.edu/~wenbobao/MEMC-Net/MEMC-Net_s_best.pth
    $ wget http://vllab1.ucmerced.edu/~wenbobao/MEMC-Net/MEMC-Net_star_best.pth
    
    
and Middlebury dataset:
    
    $ cd ../MiddleBurySet
    $ wget http://vision.middlebury.edu/flow/data/comp/zip/other-color-allframes.zip
    $ unzip other-color-allframes.zip
    $ wget http://vision.middlebury.edu/flow/data/comp/zip/other-gt-interp.zip
    $ unzip other-gt-interp.zip
    $ cd ..

We are good to go by:

    $ CUDA_VISIBLE_DEVICES=0 python demo_MiddleBury.py
    
Or if you would like to try MEMC-Net_s (smaller) or the MEMC-Net* model:
    
    $ CUDA_VISIBLE_DEVICES=0 python demo_MiddleBury.py  --netName MEMC_Net_s --pretrained MEMC-Net_s_best.pth
    $ CUDA_VISIBLE_DEVICES=0 python demo_MiddleBury.py  --netName MEMC_Net_star --pretrained MEMC-Net_star_best.pth
        
The interpolated results are under `MiddleBurySet/other-result-author/[random number]/`, where the `random number` is used to distinguish different runnings. 


### Downloading Results
Our DAIN model achieves the state-of-the-art performance on the UCF101, Vimeo90K, and Middlebury ([*eval*](http://vision.middlebury.edu/flow/eval/results/results-n1.php) and *other*).
Dowload our interpolated results with:
    
    $ wget http://vllab1.ucmerced.edu/~wenbobao/DAIN/UCF101_DAIN.zip
    $ wget http://vllab1.ucmerced.edu/~wenbobao/DAIN/Vimeo90K_interp_DAIN.zip
    $ wget http://vllab1.ucmerced.edu/~wenbobao/DAIN/Middlebury_eval_DAIN.zip
    $ wget http://vllab1.ucmerced.edu/~wenbobao/DAIN/Middlebury_other_DAIN.zip
    
    

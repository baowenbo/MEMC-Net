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
- PyTorch (The customized depth-aware flow projection and other layers require ATen API in PyTorch = 0.2.0_4)
- GCC (Compiling PyTorch 0.2.0_4 extension files (.c/.cu) requires gcc = 4.9.1 and nvcc = 9.0 compilers)
- NVIDIA GPU (We use Titan X (Pascal) with compute = 6.1, but we support compute_50/52/60/61/70 devices, should you have device with higher compute capability, please revise [this](https://github.com/baowenbo/MEMC-Net/blob/master/my_package/install.bash))



    

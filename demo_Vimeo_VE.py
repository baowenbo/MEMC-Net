import sys
import os
import sys
import  threading
import torch
from torch.autograd import Variable
import torch.utils.data

from torch.autograd import gradcheck

import numpy
from AverageMeter import  *
import datasets_benchmark
#import balancedsampler
#import models
import networks
from my_args import args
import time
from skimage.measure import compare_ssim,compare_psnr
from scipy.misc import imread, imsave, imshow, imresize, imsave
import math
import numpy as np

#from PYTHON_Flow2Color.flowToColor import flowToColor
#from PYTHON_Flow2Color.writeFlowFile import writeFlowFile
def test():
    args.datasetName = args.datasetName[0]
    args.datasetPath = args.datasetPath[0]
    #args.netName = 'MEMC_Net_VE'
    #args.batch_size = 1
    
    Vimeo_Other_GT = os.path.join(args.datasetPath,'target')
    Vimeo_Other_RESULT = os.path.join(args.datasetPath,'target_ours')

    if not os.path.exists(Vimeo_Other_RESULT):
        os.mkdir(Vimeo_Other_RESULT)

    torch.manual_seed(args.seed)
    assert(args.batch_size == 1)
    model = networks.__dict__[args.netName](batch=args.batch_size,channel= args.channels,width= None,height=None,
                                scale_num=1,scale_ratio=2,temporal=False,filter_size = args.filter_size ,
                                save_which = args.save_which, debug = args.debug,offset_scale=None,cuda_available=args.use_cuda, cuda_id=None,training=False)
    if args.use_cuda:
        print("Turn the model into CUDA")
        model = model.cuda()

    # torch.save(model.state_dict(), args.save_path + "/best" + ".pth")

    if not args.SAVED_MODEL==None:
        args.SAVED_MODEL ='./model_weights/'+ args.SAVED_MODEL
        print("The testing model weight is: " + args.SAVED_MODEL)
        if not  args.use_cuda:
            # pretrained_dict = torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage)
            model.load_state_dict(torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage))
        else:
            # pretrained_dict = torch.load(args.SAVED_MODEL)
            model.load_state_dict(torch.load(args.SAVED_MODEL))
        #print([k for k,v in      pretrained_dict.items()]) 
        # print([k for k,v in      pretrained_dict.items()])

    # model_dict = model.state_dict()
    # # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # and not k[:10]== 'rectifyNet'}
    # # 2. overwrite entries in the existing state dict
    # model_dict.update(pretrained_dict)
    # # 3. load the new state dict
    # model.load_state_dict(model_dict)

    test_set = datasets_benchmark.__dict__[args.datasetName](args.datasetPath, args.task, args.task_param)
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                            # sampler=balancedsampler.SequentialBalancedSampler(test_set,)
                                             num_workers=args.workers, pin_memory=True if args.use_cuda else False)
    print('{} samples found, {} train samples and {} test samples '.format(len(test_set),0 ,
                                                                           len(test_set)))


    training_losses = AverageMeter()
    auxiliary_data = []
    saved_total_loss = 10e10
    saved_total_PSNR = -1
    saved_total_loss_MB = 10e10
    MB_avgLoss, MB_avgPSNR = 1e5, 0
    ikk = 0
 
    args.uid =  str(numpy.random.randint(0, 100000))

    print("The id of this in-training network is " + str(args.uid))
    print(args)
    Vimeo_Other_RESULT = os.path.join(Vimeo_Other_RESULT, args.uid)
    #Turn into training mode
    model = model.eval()
    
    
    interp_error = AverageMeter()
    psnr_error = AverageMeter()
    ssim_error = AverageMeter()        
    tot_timer = AverageMeter()
    proc_timer = AverageMeter()
    end = time.time()
    for i, (Xs,y,path) in enumerate(val_loader):
        print("Proceeding to [" + str(i) + '/' + str(len(test_set))+ "]")
        path = path[0]
        os.makedirs(os.path.join(Vimeo_Other_RESULT, path), exist_ok=True)
        Xs = [Xs[i].cuda() if args.use_cuda else Xs[i] for i in range(0, 7)]
        y = y.cuda() if args.use_cuda else y

        Xs = [Variable(Xs[i], volatile= True) for i in range(0, 7)]
        y = Variable(y, volatile= True)

        ##DO I NEED PADDING?
        intWidth = Xs[0].size(3)
        intHeight =Xs[0].size(2)
        channel = Xs[0].size(1)
        
        if intWidth != ((intWidth >> 7) << 7):
            intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
            intPaddingLeft =int(( intWidth_pad - intWidth)/2)
            intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
        else:
            intWidth_pad = intWidth
            intPaddingLeft = 32
            intPaddingRight= 32

        if intHeight != ((intHeight >> 7) << 7):
            intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
            intPaddingTop = int((intHeight_pad - intHeight) / 2)
            intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
        else:
            intHeight_pad = intHeight
            intPaddingTop = 32
            intPaddingBottom = 32

        pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

        Xs = [pader(x) for x in Xs]
         
        proc_end = time.time()  
        if not args.debug:
            y_= model(Xs)#,offset_,filter_
        else:
            y_, offset_,filter_ = model(Xs)
        proc_timer.update(time.time() -proc_end)
        tot_timer.update(time.time() - end)
        end  = time.time()            
         
        y_ = y_.data.cpu().numpy()
        y_ = np.transpose(255.0 * y_.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
        
        arguments_strOut = os.path.join(Vimeo_Other_RESULT, path, 'im4.png')

        gt_path = os.path.join(Vimeo_Other_GT, path, 'im4.png')
        imsave(arguments_strOut, np.round(y_).astype(numpy.uint8))
        
        rec_rgb =  imread(arguments_strOut)
        gt_rgb = imread(gt_path)

        diff_rgb = 128.0 + rec_rgb - gt_rgb
        avg_interp_error_abs = np.mean(np.abs(diff_rgb - 128.0))
        
        interp_error.update(avg_interp_error_abs, args.batch_size)
        
        mse = numpy.mean((diff_rgb - 128.0) ** 2)
        if mse == 0:
            return 100.0
        PIXEL_MAX = 255.0
        psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        psnr_error.update(psnr, args.batch_size)
        ssim = compare_ssim(rec_rgb, gt_rgb,multichannel=True)
        ssim_error.update(ssim, args.batch_size)
        
        print("interpolation error / PSNR : " + str(round(avg_interp_error_abs,4)) + " ,\t  psnr " + str(round(psnr,4))+ ",\t ssim " + str(round(ssim,5)))
        print("Per Image Processing Total time (I/O + processing) : " + str(tot_timer.avg))
        print("Per Image Processing Total time (processing) : " + str(proc_timer.avg))
        
        metrics = "The average interpolation error / PSNR for all images are : " + \
            str(round(interp_error.avg,4)) + ",\t  psnr " + str(round(psnr_error.avg,4)) + ",\t  ssim " + str(round(ssim_error.avg,4)) 
        print(metrics)
    
    fl = open(os.path.join(Vimeo_Other_RESULT, "metrics.txt"), 'w')
    fl.write(metrics)
    fl.write("\n")
    fl.close()
           

if __name__ == '__main__':
    sys.setrecursionlimit(100000)# 0xC00000FD exception for the recursive detach of gradients.
    threading.stack_size(200000000)# 0xC00000FD exception for the recursive detach of gradients.
    thread = threading.Thread(target=test)
    thread.start()
    thread.join()

    exit(0)

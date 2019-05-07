import torch
import os
from torch.autograd import Variable
import math

import random
import numpy as np
import numpy
import networks
from my_args import  args

from scipy.misc import imread, imsave
from AverageMeter import  *

torch.backends.cudnn.benchmark = True # to speed up the


DO_MiddleBurryOther = True
MB_Other_DATA = "./MiddleBurySet/other-data/"
MB_Other_RESULT = "./MiddleBurySet/other-result-author/"
MB_Other_GT = "./MiddleBurySet/other-gt-interp/"
if not os.path.exists(MB_Other_RESULT):
    os.mkdir(MB_Other_RESULT)



model = networks.__dict__[args.netName](
                                channel=args.channels,
                                filter_size = args.filter_size ,
                                training=False)

if args.use_cuda:
    model = model.cuda()

if isinstance(args.SAVED_MODEL, str):
    args.SAVED_MODEL = './model_weights/' + args.SAVED_MODEL
    print("The testing model weight is: " + args.SAVED_MODEL)
    if not args.use_cuda:
        pretrained_dict = torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage)
        # model.load_state_dict(torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage))
    else:
        pretrained_dict = torch.load(args.SAVED_MODEL)
        # model.load_state_dict(torch.load(args.SAVED_MODEL))

    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
else:
    print("running speed test")


model = model.eval() # deploy mode



use_cuda=args.use_cuda
save_which=args.save_which
dtype = args.dtype
unique_id =str(random.randint(0, 100000))
print("The unique id for current testing is: " + str(unique_id))

interp_error = AverageMeter()
if DO_MiddleBurryOther:
        subdir = os.listdir(MB_Other_DATA)
        gen_dir = os.path.join(MB_Other_RESULT, unique_id)
        os.mkdir(gen_dir)

        for dir in subdir:
            # prepare the image save path
            print(dir)
            os.mkdir(os.path.join(gen_dir, dir))
            arguments_strFirst = os.path.join(MB_Other_DATA, dir, "frame10.png")
            arguments_strSecond = os.path.join(MB_Other_DATA, dir, "frame11.png")
            arguments_strOut = os.path.join(gen_dir, dir, "frame10i11.png")
            gt_path = os.path.join(MB_Other_GT, dir, "frame10i11.png")

            X0 =  torch.from_numpy( np.transpose(imread(arguments_strFirst) , (2,0,1)).astype("float32")/ 255.0).type(dtype)
            X1 =  torch.from_numpy( np.transpose(imread(arguments_strSecond) , (2,0,1)).astype("float32")/ 255.0).type(dtype)


            y_ = torch.FloatTensor()

            assert (X0.size(1) == X1.size(1))
            assert (X0.size(2) == X1.size(2))

            intWidth = X0.size(2)
            intHeight = X0.size(1)
            channel = X0.size(0)
            if not channel == 3:
                continue

            assert ( intWidth <= 1280)  # while our approach works with larger images, we do not recommend it unless you are aware of the implications
            assert ( intHeight <= 720)  # while our approach works with larger images, we do not recommend it unless you are aware of the implications

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

            X0 = Variable(torch.unsqueeze(X0,0),volatile = True)
            X1 = Variable(torch.unsqueeze(X1,0),volatile = True)
            X0 = pader(X0)
            X1 = pader(X1)

            if use_cuda:
                X0 = X0.cuda()
                X1 = X1.cuda()
            y_s,offset,filter,occlusion = model(torch.stack((X0, X1),dim = 0))
            y_ = y_s[save_which]

            if use_cuda:
                X0 = X0.data.cpu().numpy()
                y_ = y_.data.cpu().numpy()
                offset = [offset_i.data.cpu().numpy() for offset_i in offset]
                filter = [filter_i.data.cpu().numpy() for filter_i in filter]
                occlusion = [occlusion_i.data.cpu().numpy() for occlusion_i in occlusion]
                X1 = X1.data.cpu().numpy()
            else:
                X0 = X0.data.numpy()
                y_ = y_.data.numpy()
                offset = [offset_i.data.numpy() for offset_i in offset]
                filter = [filter_i.data.numpy() for filter_i in filter]
                occlusion = [occlusion_i.data.numpy() for occlusion_i in occlusion]
                X1 = X1.data.numpy()



            X0 = np.transpose(255.0 * X0.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
            y_ = np.transpose(255.0 * y_.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
            offset = [np.transpose(offset_i[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for offset_i in offset]
            filter = [np.transpose(
                filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
                (1, 2, 0)) for filter_i in filter]
            occlusion = [np.transpose(
                occlusion_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
                (1, 2, 0)) for occlusion_i in occlusion]
            X1 = np.transpose(255.0 * X1.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))


            imsave(arguments_strOut, np.round(y_).astype(numpy.uint8))

            rec_rgb =  imread(arguments_strOut)
            gt_rgb = imread(gt_path)

            diff_rgb = 128.0 + rec_rgb - gt_rgb
            avg_interp_error_abs = np.mean(np.abs(diff_rgb - 128.0))

            interp_error.update(avg_interp_error_abs, 1)

            mse = numpy.mean((diff_rgb - 128.0) ** 2)
            PIXEL_MAX = 255.0
            psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

            print("interpolation error / PSNR : " + str(round(avg_interp_error_abs,4)) + " / " + str(round(psnr,4)))
            metrics = "The average interpolation error / PSNR for all images are : " + str(round(interp_error.avg, 4))
            print(metrics)

            diff_rgb = diff_rgb.astype("uint8")

            imsave(os.path.join(gen_dir, dir, "frame10i11_diff" + str('{:.4f}'.format(avg_interp_error_abs)) + ".png"),
                   diff_rgb)

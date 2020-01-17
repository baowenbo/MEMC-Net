# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import torch.nn as nn
# from torch.nn import  init
import torch.nn.init as weight_init
import torchvision
from PIL import Image, ImageOps
from itertools import  product
import torch.nn.functional as F
import numpy as np
import sys
import threading

from my_package.modules.InterpolationModule import InterpolationModule
from my_package.modules.FilterInterpolationModule import  FilterInterpolationModule
from my_package.modules.FlowProjectionModule import  FlowProjectionModule #,FlowFillholeModule

import networks.FlowNetS as FlowNetS
from Stack import Stack

#import spynet
import networks.ResNet as ResNet

from networks.EDSR.EDSR import EDSR

class MEMC_Net_VE(torch.nn.Module):
    def __init__(self,
                 batch=128, channel = 3,
                 width= 100, height = 100,
                 scale_num = 3, scale_ratio = 2,
                 temporal = False,
                 filter_size = 4,
                 offset_scale = 1,                              
                 save_which = 0,
                 debug = False,
                 cuda_available=False, cuda_id = 0,
                 training=True):

        # base class initialization
        super(MEMC_Net_VE, self).__init__()

        # class parameters
        self.scale_num = scale_num
        self.scale_ratio = scale_ratio
        # self.temporal = temporal

        self.filter_size = filter_size
        self.cuda_available = cuda_available
        self.cuda_id = cuda_id
        self.offset_scale = offset_scale
        self.training = training

        # assert width == height

        self.w = []
        self.h = []
        self.ScaleNets_offset = []
        self.ScaleNets_filter = []
        self.ScaleNets_occlusion = []

        self._grid_param = None


        i = 0
        self.initScaleNets_filter,self.initScaleNets_filter1 = \
            self.get_MonoNet5(channel if i == 0 else channel + filter_size * filter_size, filter_size * filter_size, "filter")
        # self.initScaleNets_occlusion,self.initScaleNets_occlusion1=\
        #     self.get_MonoNet5(channel if i == 0 else channel+2, 1 , "occlusion")

        if self.scale_num > 1:
            i = 1


        # self.rectifyNet = self.get_RectifyNet2(3 * 7, 3)
        self.randinput = None
        self.rectifyNet = EDSR(3*7 + 64 * 7 + 2 * 6 + 16 * 6, n_resblocks= 10, n_feats= 128)

        # initialize model weights.
        self._initialize_weights()

        #self.flowmethod =flowmethod
        #if flowmethod == 0 :
        if self.training:
            self.flownets = FlowNetS.__dict__['flownets']("models/flownets_pytorch.pth")
        else:
            self.flownets = FlowNetS.__dict__['flownets']()
        self.div_flow = 20
        #elif flowmethod == 1:
        #    self.flownets = spynet.Network()
        #    self.div_flow = 1 # No scaling is used in SPynet

        # extract contextual information
        self.ctxNet = ResNet.__dict__['resnet18_conv1'](pretrained=True)
        self.ctx_ch = 64  # conv1 output has 64 channels.

        # self.ip1 = InterpolationModule()
        # self.ip2 = InterpolationModule()
        self.save_which = save_which
        self.debug = debug

        # self.FilterInterpolate_ModuleList = [FilterInterpolationModule() for i in range(0,6)]
        # self.FilterInterpolate_ctx_ModuleList = [FilterInterpolationModule() for i in range(0,6)]
        return

    def _initialize_weights(self):
        count = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # print(m)
                count+=1
                # print(count)
                # weight_init.xavier_uniform(m.weight.data)
                weight_init.kaiming_uniform(m.weight.data, a = 0, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            # else:
            #     print(m)


    def forward(self, inputs,y=None):
        # if self.randinput is None:
        #     self.randinput = Variable(torch.zeros(inputs[0].size(0),3*7 + 64 * 7 + 2 * 6 + 16 * 6,
        #                                           inputs[0].size(2),inputs[0].size(3)))
        #     self.randinput = self.randinput.cuda()
        # cur_output_rectified = inputs[3] + self.rectifyNet(self.randinput)
        #
        # # cur_output_rectified = []
        # # flow = []
        # # filter = []
        # # print("dd")
        # return [ cur_output_rectified,cur_output_rectified, cur_output_rectified,cur_output_rectified,
        #          cur_output_rectified, cur_output_rectified,cur_output_rectified]

        """
        The input may compose 3 or 7 frames which should be consistent
        with the temporal settings.

        Parameters
        ----------
        input: shape (3, batch, 3, width, height)
        -----------
        """
        losses = []
        # offsets= []
        # filters = []
        # occlusions = []


        '''
            STEP 1: sequeeze the input 
        '''
        # assert input.size(0) == 7
        # inputs = torch.squeeze(input,dim=0)

        batch = inputs[0].size(0)

        '''
            STEP 2: initialize the auxiliary input either from temporal or scale predecessor
        '''

        '''
            STEP 3: iteratively execuate the Multiscale Network 
        '''
        # from the coarser scale to the most

        '''
            STEP 3.1: prepare current scale inputs
        '''

        '''
            STEP 3.2: concatenating the inputs.
        '''
        '''
            STEP 3.3: perform the estimation by the Three subpath Network 
        '''
        cur_inputs =[]
        cur_output= []
        cur_ctx_output= []
        for i in range(0,7):
            if not i==3:
                cur_inputs.append(torch.cat((inputs[3], inputs[i]),dim=1))


        cur_input = torch.cat(cur_inputs,dim=0)
        flow = self.forward_flownets(self.flownets,cur_input)
        # print("dd")
        # return flow

        temp = self.forward_singlePath(self.initScaleNets_filter,cur_input, 'filter')
        filter =self.forward_singlePath(self.initScaleNets_filter1,temp,name=None)

        # return filter
        # temp = self.forward_singlePath(self.initScaleNets_occlusion,cur_input,'occlusion')
        # occ= self.forward_singlePath(self.initScaleNets_occlusion1,temp,name=None)

        '''
            STEP 3.4: perform the frame interpolation process 
        '''
        for i in range(0,7):
            if i < 3:
                cur_output.append(self.FilterInterpolate(#self.FilterInterpolate_ModuleList[i],
                                        inputs[i],
                                        flow[i*batch:(i+1) * batch],
                                         filter[i*batch:(i+1) * batch],
                                         self.debug
                                                 # occ[i*batch:(i+1) *batch]
                                         ))
                cur_ctx_output.append(self.FilterInterpolate_ctx(#self.FilterInterpolate_ctx_ModuleList[i],
                    ctx0=self.ctxNet(inputs[i]),
                    offset=flow[i*batch:(i+1) * batch],
                    filter=filter[i*batch:(i+1) * batch]
                ))
            elif i >3:
                cur_output.append(self.FilterInterpolate(#self.FilterInterpolate_ModuleList[i-1],
                                    inputs[i],
                                     flow[ (i-1) * batch:(i ) * batch],
                                     filter[(i -1) * batch:(i) * batch],
                                     self.debug
                                     # occ[i*batch:(i+1) *batch]
                                     ))
                cur_ctx_output.append(self.FilterInterpolate_ctx(
#                    self.FilterInterpolate_ctx_ModuleList[i-1],
                    ctx0=self.ctxNet(inputs[i]),
                     offset=flow[ (i-1) * batch:(i ) * batch],
                     filter=filter[(i -1) * batch:(i) * batch]
                ))
            # else:
            else:
                cur_output.append(inputs[3])
                cur_ctx_output.append(self.ctxNet(inputs[3]))
        # return  cur_output
        cat_cur_output = torch.cat((cur_output),dim =1)
        cat_cur_ctx_output = torch.cat((cur_ctx_output),dim=1)
        # for i in range(0,7):
            # cur_ctx_output.append(self.ctxNet(inputs[i]))
            # cur_ctx_output.append(self.ctxNet(inputs[i]))


        cat_flow_filter = torch.cat( (flow[0:batch],        flow[batch:2*batch],    flow[2*batch:3*batch],
                                      flow[3*batch:4*batch], flow[4*batch:5*batch], flow[5*batch:6*batch],
                                      filter[0:batch], filter[batch:2 * batch], filter[2 * batch:3 * batch],
                                      filter[3 * batch:4 * batch], filter[4 * batch:5 * batch], filter[5 * batch:6 * batch],
                                      ) ,dim=1)
        # cat_cur_ctx_output = torch.cat(cur_ctx_output,dim=1)
        # if self.save_which == 1:
        # temp_rectify_input = self.fillHole(cur_output,cur_input_0, cur_input_2,hole_value=0.0)
        rectify_input = torch.cat((cat_cur_ctx_output,cat_flow_filter,cat_cur_output),dim =1)
        #I need to detach the rectify input so that the gradients won't be back propagated.
        # use a residual connection here
        # cur_output_rectified = inputs[3] + self.forward_singlePath(self.rectifyNet, rectify_input, name=None)
        cur_output_rectified = inputs[3] + self.rectifyNet(rectify_input)
        # else:
        #     cur_output_rectified = cur_output
        # if self.debug:
        #     print('max' + str(torch.max(cur_output_rectified).data[0]))
        #     print("min" + str(torch.min(cur_output_rectified).data[0]))

        '''
            STEP 3.5: for training phase, we collect the variables to be penalized.
        '''
        if self.training == True:
            for i in range(0,7):
                if not i == 3:
                    losses += [cur_output[i] - y]
                else:
                    losses += [cur_output_rectified - y]

        '''
            STEP 3.6: prepare inputs for the next finer scale
        '''
        # print("D")
        '''
            STEP 4: return the results
        '''
        if self.training == True:
            return losses #, offsets,filters,occlusions
        else:
            if not self.debug:
                return cur_output_rectified
            else:
                return cur_output_rectified,flow,filter

    def forward_flownets(self, model, input):
        temp = model(input)  # this is a single direction motion results, but not a bidirectional one
        temp = self.div_flow * temp # single direction to bidirection should haven it.
        temp = nn.Upsample(scale_factor=4, mode='bilinear')(temp)  # nearest interpolation won't be better i think
        return temp
    # @staticmethod
    def forward_singlePath(self, modulelist, input, name):
        stack = Stack()
        # if self.temporal:
        #     stack_c = Stack()
        #     stack_n = Stack()

        k = 0
        temp = []
        # if self.temporal:
        #     temp_c = []
        #     temp_n = []
        for layers in modulelist:  # self.initScaleNets_offset:
            # TODO: we need to store the sequential results so as to add a skip connection into the whole model.
            # print(type(layers).__name__)
            # print(k)
            # use the pop-pull logic, looks like a stack.
            if k == 0:
                temp = layers(input)
                # if self.temporal:
                #     temp_c = layers(input_c)
                #     temp_n = layers(input_n)
            else:
                # met a pooling layer, take its input
                if isinstance(layers, nn.AvgPool2d) or isinstance(layers,nn.MaxPool2d):
                    stack.push(temp)
                    # if self.temporal:
                    #     stack_c.push(temp_c)
                    #     stack_n.push(temp_n)

                temp = layers(temp)
                # if self.temporal:
                #     temp_c = layers(temp_c)
                #     temp_n = layers(temp_n)

                # met a unpooling layer, take its output
                if isinstance(layers, nn.Upsample):
                    if name == 'offset':
                        temp = torch.cat((temp,stack.pop()),dim=1)  # short cut here, but optical flow should concat instead of add
                    else:
                        temp += stack.pop()  # short cut here, but optical flow should concat instead of add
                    # if self.temporal:
                    #     temp_c += stack_c.pop()
                    #     temp_n += stack_n.pop()

            k += 1
        # if self.temporal == False:
        return temp
        # else:
            # return temp, temp_c, temp_n
    
    '''keep this function'''
    def get_MonoNet5(self, channel_in, channel_out, name):

        '''
        Generally, the MonoNet is aimed to provide a basic module for generating either offset, or filter, or occlusion.

        :param channel_in: number of channels that composed of multiple useful information like reference frame, previous coarser-scale result
        :param channel_out: number of output the offset or filter or occlusion
        :param name: to distinguish between offset, filter and occlusion, since they should use different activations in the last network layer

        :return: output the network model
        '''
        model = []

        # block1
        model += self.conv_relu(channel_in * 2, 32, (3, 3), (1, 1))
        model += self.conv_relu(32, 32, (3, 3), (1, 1))
        model += self.conv_relu_maxpool(32, 32, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.5
        # block2
        model += self.conv_relu(32, 64, (3, 3), (1, 1))
        model += self.conv_relu_maxpool(64, 64, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.4
        # block3
        model += self.conv_relu(64, 128, (3, 3), (1, 1))
        model += self.conv_relu_maxpool(128, 128, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.3
        # block4
        model += self.conv_relu(128, 256, (3, 3), (1, 1))
        model += self.conv_relu_maxpool(256, 256, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.2
        # block5
        model += self.conv_relu(256, 512, (3, 3), (1, 1))
        model += self.conv_relu_maxpool(512, 512, (3, 3), (1, 1), (2, 2))

        # intermediate block5_5
        model += self.conv_relu(512, 512, (3, 3), (1, 1))
        model += self.conv_relu(512, 512, (3, 3), (1, 1))

        # block 6
        model += self.conv_relu_unpool(512, 512, (3, 3), (1, 1), 2)  # THE OUTPUT No.1 UP
        model += self.conv_relu(512+512 if name == "offset" else 512, 256, (3, 3), (1, 1))
        # block 7
        model += self.conv_relu_unpool(256, 256, (3, 3), (1, 1), 2)  # THE OUTPUT No.2 UP
        model += self.conv_relu(256+256 if name == "offset" else 256, 128, (3, 3), (1, 1))
        # block 8
        model += self.conv_relu_unpool(128, 128, (3, 3), (1, 1), 2)  # THE OUTPUT No.3 UP
        model += self.conv_relu(128+128 if name == "offset" else 128, 64, (3, 3), (1, 1))

        # block 9
        model += self.conv_relu_unpool(64, 64, (3, 3), (1, 1), 2)  # THE OUTPUT No.4 UP
        model += self.conv_relu(64+64 if name == "offset" else 64, 32, (3, 3), (1, 1))

        # block 10
        model += self.conv_relu_unpool(32,  32, (3, 3), (1, 1), 2)  # THE OUTPUT No.5 UP
        model += self.conv_relu(32+32 if name == "offset" else 32, 16, (3, 3), (1, 1))

        # output our final purpose
        branch1 = []
        branch2 = []
        branch1 += self.conv_relu_conv(16, channel_out,  (3, 3), (1, 1))
        # branch2 += self.conv_relu_conv(16, channel_out,  (3, 3), (1, 1))
        #
        if name == "offset":
            branch1 += self.sigmoid_activation() # limit to 0~1
            # branch2 += self.sigmoid_activation()
            # pass
        elif name == "filter":
            # TODO: Sigmoid is not used in Simon's implementation, which mean we dont acutally need to explicityly use a filter with weights summed to exactly 1.0.
            # since i use a non-normalized distance weighted , then the learned filter is also non-normalized.
            # We only have to make it a positive value.
            # model += self.softmax_activation()
            # model += self.relu_activation()
            # model = self.binary_activation() # we shouldn't have used the relu because each participated pixel should have a weight larget than zeros
            pass

        elif name == "occlusion":
            # we need to get a binary occlusion map for both reference frames
            # model += self.binary_activation()
            # model  += self.softmax_activation()
            pass # we leave all the three branched no special layer.
        return  (nn.ModuleList(model), nn.ModuleList(branch1)) #, nn.ModuleList(branch2))

    def get_RectifyNet2(self, channel_in, channel_out):
        model = []
        #model += self.conv_relu_conv(channel_in, channel_out, (7, 7), (3, 3))
        model += self.conv_relu(channel_in, 64, (3,3), (1,1))
        model += self.conv_relu(64, 64, (3,3), (1,1))
        model += self.conv_relu(64, 64, (3,3), (1,1))
        model += self.conv_relu(64, 64, (3,3), (1,1))
        model += self.conv_relu(64, 64, (3,3), (1,1))
        model += self.conv_relu(64, 64, (3,3), (1,1))
        model += self.conv_relu(64, 64, (3,3), (1,1))
        model += nn.Sequential(*[nn.Conv2d(64,channel_out,(3,3),1, (1,1))])
        return nn.ModuleList(model)

    def get_RectifyNet(self, channel_in, channel_out):
        model = []
        model += self.conv_relu_conv(channel_in, channel_out, (7, 7), (3, 3))
        return nn.ModuleList(model)


    # @staticmethod
    # def Interpolate(ip0,ip1, ref0,ref2,offset,filter,occlusion):
    #     ref0_offset = ip0(ref0, offset[:, :2, ...])
    #     ref2_offset = ip1(ref2, offset[:, 2:, ...])
    #     return  ref0_offset/2.0 + ref2_offset/2.0
    @staticmethod
    def FlowProject(input):
        # print(input.requires_grad)
        output = FlowProjectionModule(input.requires_grad)(input)

        # if output.requires_grad == True:
        #     output = FlowFillholeModule()(input,hole_value = -10000.0)
        return output

    @staticmethod
    def fillHole(input,ref0,ref2, hole_value = 0.0):
        index = input == hole_value
        output = input.clone()
        output[index] = (ref0[index] + ref2[index]) /2.0

        return output
    @staticmethod
    # def FilterInterpolate_ctx(self, module, ctx0, offset,filter):
    def FilterInterpolate_ctx(ctx0, offset,filter):
        ctx0_offset = FilterInterpolationModule()(ctx0,offset,filter)
        # ctx0_offset = module(ctx0,offset,filter)
        # ctx2_offset = FilterInterpolationModule()(ctx2,offset[1],filter[1])

        return ctx0_offset.detach()

    @staticmethod
    # def FilterInterpolate(self, module, ref0, offset, filter,debug):
    def FilterInterpolate(ref0, offset, filter,debug):
        output = FilterInterpolationModule()(ref0, offset,filter)
        # output = module(ref0, offset,filter)
        if debug:
            print('filter max: ' + str(torch.max(filter).data[0]))
            print('filter min: ' + str(torch.min(filter).data[0]))

            print('max' + str(torch.max(output).data[0]))
            print("min" + str(torch.min(output).data[0]))
        # occlusion0, occlusion2 = torch.split(occlusion, 1, dim=1)
        # print((occlusion0[0,0,1,1] + occlusion2[0,0,1,1]))
        # output = (occlusion0 * ref0_offset + occlusion2 * ref2_offset) / (occlusion0 + occlusion2)
        # automatically broadcasting the occlusion to the three channels of and image.
        return output
        # return ref0_offset/2.0 + ref2_offset/2.0
 
    @staticmethod
    def Interpolate(ref0,ref2,offset,filter,occlusion):
        ref2_offset = InterpolationModule()(ref2, offset[:, 2:, ...])
        ref0_offset = InterpolationModule()(ref0, offset[:, :2, ...])

        occlusion0, occlusion2 = torch.split(occlusion,1, dim=1)

        # output = (occlusion0 * ref0_offset + occlusion2 * ref2_offset)/(occlusion0 + occlusion2)
        output = occlusion0 * ref0_offset + occlusion2 * ref2_offset
        # automatically broadcasting the occlusion to the three channels of and image.
        return  output
        #return ref0_offset/2.0 + ref2_offset/2.0
 
    '''keep this function'''
    @staticmethod
    def conv_relu_conv(input_filter, output_filter, kernel_size,
                        padding):

        # we actually don't need to use so much layer in the last stages.
        layers = nn.Sequential(
            nn.Conv2d(input_filter, input_filter, kernel_size, 1, padding),
            nn.ReLU(inplace=False),
            nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),
            # nn.ReLU(inplace=False),
            # nn.Conv2d(output_filter, output_filter, kernel_size, 1, padding),
            # nn.ReLU(inplace=False),
            # nn.Conv2d(output_filter, output_filter, kernel_size, 1, padding),
        )
        return layers

    '''keep this function'''

    @staticmethod
    def sigmoid_activation():
        layers = nn.Sequential(
            # No need to use a Sigmoid2d, since we just focus on one
            nn.Sigmoid()
        )
        return layers
 

    '''keep this function'''
    @staticmethod
    def conv_relu(input_filter, output_filter, kernel_size,
                        padding):
        layers = nn.Sequential(*[
            nn.Conv2d(input_filter,output_filter,kernel_size,1, padding),

            nn.ReLU(inplace=False)
        ])
        return layers

    '''keep this function'''
    @staticmethod
    def conv_relu_maxpool(input_filter, output_filter, kernel_size,
                            padding,kernel_size_pooling):

        layers = nn.Sequential(*[
            nn.Conv2d(input_filter,output_filter,kernel_size,1, padding),

            nn.ReLU(inplace=False),

            nn.BatchNorm2d(output_filter),

            nn.MaxPool2d(kernel_size_pooling)
        ])
        return layers

    '''keep this function'''
    @staticmethod
    def conv_relu_unpool(input_filter, output_filter, kernel_size,
                            padding,unpooling_factor):

        layers = nn.Sequential(*[
            nn.Conv2d(input_filter,output_filter,kernel_size,1, padding),

            nn.ReLU(inplace=False),

            nn.BatchNorm2d(output_filter),

            nn.Upsample(scale_factor=unpooling_factor, mode='bilinear')
            # nn.UpsamplingBilinear2d(unpooling_size,scale_factor=unpooling_size[0])
        ])
        return layers
 
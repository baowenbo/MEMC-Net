# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.init as weight_init

from my_package.modules.FilterInterpolationModule import  FilterInterpolationModule
from my_package.modules.FlowProjectionModule import  FlowProjectionModule

from Stack import Stack
import networks.SPyNet as SPyNet

class MEMC_Net_s(torch.nn.Module):
    def __init__(self,
                 channel = 3,
                 filter_size = 4,
                 training=True):

        # base class initialization
        super(MEMC_Net_s, self).__init__()

        # class parameters
        self.filter_size = filter_size
        self.training = training
        i = 0
        self.initScaleNets_filter,self.initScaleNets_filter1,self.initScaleNets_filter2 = \
            self.get_MonoNet5(channel if i == 0 else channel + filter_size * filter_size, filter_size * filter_size, "filter")

        self.rectifyNet = self.get_RectifyNet2(3 + 4 + 32 + 2, 3)

        # initialize model weights.
        self._initialize_weights()


        self.flownets = SPyNet.Network(self.training)
        self.div_flow = 1 # No scaling is used in SPynet

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


    def forward(self, input):

        """
        Parameters
        ----------
        input: shape (3, batch, 3, width, height)
        -----------
        """
        losses = []
        offsets= []
        filters = []
        occlusions = []


        '''
            STEP 1: sequeeze the input 
        '''
        if self.training == True:
            assert input.size(0) == 3
            input_0,input_1,input_2 = torch.squeeze(input,dim=0)
        else:
            assert input.size(0) ==2
            input_0,input_2 = torch.squeeze(input,dim=0)

        #prepare the input data of current scale
        cur_input_0 = input_0
        if self.training == True:
            cur_input_1 = input_1
        cur_input_2 = input_2

        '''
            STEP 3.2: concatenating the inputs.
        '''
        cur_offset_input = torch.cat((cur_input_0, cur_input_2), dim=1)
        cur_filter_input = cur_offset_input # torch.cat((cur_input_0, cur_input_2), dim=1)

        '''
            STEP 3.3: perform the estimation by the Three subpath Network 
        '''
        cur_offset_output = [
                self.FlowProject(  self.forward_flownets(self.flownets,cur_offset_input,flowmethod=1)),
                self.FlowProject(  self.forward_flownets(self.flownets, torch.cat((cur_offset_input[:,3:,...],
                                                       cur_offset_input[:,0:3,...]),dim = 1),1))
        ]

        temp = self.forward_singlePath(self.initScaleNets_filter,cur_filter_input, 'filter')
        cur_filter_output =[ self.forward_singlePath(self.initScaleNets_filter1,temp,name=None),
                             self.forward_singlePath(self.initScaleNets_filter2,temp,name=None)]

        cur_occlusion_output = [
            cur_filter_output[0][:, :1, :, :],
            cur_filter_output[0][:, :1, :, :]
        ]
        '''
            STEP 3.4: perform the frame interpolation process 
        '''
        cur_output = self.FilterInterpolate(cur_input_0, cur_input_2,cur_offset_output,cur_filter_output,self.filter_size**2)

        rectify_input = torch.cat((cur_output,cur_offset_output[0],cur_offset_output[1],
                                   cur_filter_output[0],cur_filter_output[1],
                                    cur_occlusion_output[0], cur_occlusion_output[1]),dim =1)
        cur_output_rectified = cur_output + self.forward_singlePath(self.rectifyNet, rectify_input, name=None)

        '''
            STEP 3.5: for training phase, we collect the variables to be penalized.
        '''
        if self.training == True:
            losses +=[cur_output - cur_input_1]
            losses += [cur_output_rectified - cur_input_1]
            offsets +=[cur_offset_output]
            filters += [cur_filter_output]
            occlusions += [cur_occlusion_output]

        '''
            STEP 4: return the results
        '''
        if self.training == True:
            # if in the training phase, we output the losses to be minimized.
            # return losses, loss_occlusion
            return losses, offsets,filters,occlusions
        else:
            # if in test phase, we directly return the interpolated frame
            cur_outputs = [cur_output,cur_output_rectified]
            return cur_outputs,cur_offset_output,cur_filter_output,cur_occlusion_output

    def forward_flownets(self, model, input, flowmethod = 1):
        if flowmethod == 0:  # spynet don't need upsample
            temp = model(input)  # this is a single direction motion results, but not a bidirectional one
            temp = self.div_flow * temp / 2.0  # single direction to bidirection should haven it.
            temp = nn.Upsample(scale_factor=4, mode='bilinear')(temp)  # nearest interpolation won't be better i think
        elif flowmethod == 1:
            temp = model(input[:,:3,:,:],input[:,3:,:,:])
            temp = temp / 2.0
            # print(temp.size())
        return temp
    # @staticmethod
    def forward_singlePath(self, modulelist, input, name):
        stack = Stack()


        k = 0
        temp = []

        for layers in modulelist:  # self.initScaleNets_offset:
            # TODO: we need to store the sequential results so as to add a skip connection into the whole model.
            # print(type(layers).__name__)
            # print(k)
            # if k == 27:
            #     print(k)
            #     pass
            # use the pop-pull logic, looks like a stack.
            if k == 0:
                temp = layers(input)
            else:
                # met a pooling layer, take its input
                if isinstance(layers, nn.AvgPool2d) or isinstance(layers,nn.MaxPool2d):
                    stack.push(temp)

                temp = layers(temp)

                # met a unpooling layer, take its output
                if isinstance(layers, nn.Upsample):
                    if name == 'offset':
                        temp = torch.cat((temp,stack.pop()),dim=1)  # short cut here, but optical flow should concat instead of add
                    else:
                        temp += stack.pop()  # short cut here, but optical flow should concat instead of add

            k += 1
        return temp

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
        model += self.conv_relu(channel_in * 2, 16, (3, 3), (1, 1))
        model += self.conv_relu_maxpool(16, 32, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.5
        # block2
        model += self.conv_relu_maxpool(32, 64, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.4
        # block3
        model += self.conv_relu_maxpool(64, 128, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.3
        # block4
        model += self.conv_relu_maxpool(128, 256, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.2
        # block5
        model += self.conv_relu_maxpool(256, 512, (3, 3), (1, 1), (2, 2))

        # intermediate block5_5
        model += self.conv_relu(512, 512, (3, 3), (1, 1))

        # block 6
        model += self.conv_relu_unpool(512, 256, (3, 3), (1, 1), 2)  # THE OUTPUT No.1 UP
        # block 7
        model += self.conv_relu_unpool(256, 128, (3, 3), (1, 1), 2)  # THE OUTPUT No.2 UP
        # block 8
        model += self.conv_relu_unpool(128, 64, (3, 3), (1, 1), 2)  # THE OUTPUT No.3 UP

        # block 9
        model += self.conv_relu_unpool(64, 32, (3, 3), (1, 1), 2)  # THE OUTPUT No.4 UP

        # block 10
        model += self.conv_relu_unpool(32,  16, (3, 3), (1, 1), 2)  # THE OUTPUT No.5 UP

        # output our final purpose
        branch1 = []
        branch2 = []
        branch1 += self.conv_relu_conv(16, channel_out,  (3, 3), (1, 1))
        branch2 += self.conv_relu_conv(16, channel_out,  (3, 3), (1, 1))

        return  (nn.ModuleList(model), nn.ModuleList(branch1), nn.ModuleList(branch2))

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

    @staticmethod
    def FlowProject(input):
        output = FlowProjectionModule(input.requires_grad)(input)

        return output
    @staticmethod
    def FilterInterpolate(ref0, ref2, offset, filter,filter_size2):
        ref0_offset = FilterInterpolationModule()(ref0, offset[0],filter[0])
        ref2_offset = FilterInterpolationModule()(ref2, offset[1],filter[1])

        return ref0_offset/2.0 + ref2_offset/2.0



    '''keep this function'''
    @staticmethod
    def conv_relu_conv(input_filter, output_filter, kernel_size,
                        padding):

        # we actually don't need to use so much layer in the last stages.
        layers = nn.Sequential(
            nn.Conv2d(input_filter, input_filter, kernel_size, 1, padding),
            nn.ReLU(inplace=False),
            nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),
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

            nn.Upsample(scale_factor=unpooling_factor, mode='bilinear'),

            nn.Conv2d(input_filter,output_filter,kernel_size,1, padding),

            nn.ReLU(inplace=False),

            nn.BatchNorm2d(output_filter),
        ])
        return layers

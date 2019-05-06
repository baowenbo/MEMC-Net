# this is for wrapping the customized layer
import torch
from torch.autograd import Function
import my_package._ext.my_lib as my_lib

class FilterInterpolationLayer(Function):
    def __init__(self):
        super(FilterInterpolationLayer,self).__init__()

    def forward(self, input1,input2,input3):

        # assert(input1.is_contiguous())
        # assert(input2.is_contiguous())
        self.input1 = input1.contiguous() # need to use in the backward process, so we need to cache it
        self.input2 = input2.contiguous() # TODO: Note that this is simply a shallow copy?
        self.input3 = input3.contiguous()

        # if input1.is_cuda:
        #     self.device = torch.cuda.current_device()
        # else:
        #     self.device = -1

        # output =  torch.zeros(input1.size())


        if input1.is_cuda :
            # output = output.cuda()
            output = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
            my_lib.FilterInterpolationLayer_gpu_forward(input1, input2, input3, output)
        else:
            # output = torch.cuda.FloatTensor(input1.data.size())
            my_lib.FilterInterpolationLayer_cpu_forward(input1, input2, input3, output)

        # the function returns the output to its caller
        return output

    #TODO: if there are multiple outputs of this function, then the order should be well considered?
    def backward(self, gradoutput):
        # print("Backward of Filter Interpolation Layer")
        # gradinput1 = input1.new().zero_()
        # gradinput2 = input2.new().zero_()
        # gradinput1 = torch.zeros(self.input1.size())
        # gradinput2 = torch.zeros(self.input2.size())
        # gradinput3 = torch.zeros(self.input3.size())

        gradinput1 = torch.cuda.FloatTensor().resize_(self.input1.size()).zero_()
        gradinput2 = torch.cuda.FloatTensor().resize_(self.input2.size()).zero_()
        gradinput3 = torch.cuda.FloatTensor().resize_(self.input3.size()).zero_()
        if self.input1.is_cuda:
            # print("CUDA backward")
            # gradinput1 = gradinput1.cuda(self.device)
            # gradinput2 = gradinput2.cuda(self.device)
            # gradinput3 = gradinput3.cuda(self.device)

            err = my_lib.FilterInterpolationLayer_gpu_backward(self.input1,self.input2,self.input3, gradoutput, gradinput1, gradinput2, gradinput3)
            if err != 0 :
                print(err)

        else:
            # print("CPU backward")
            # print(gradoutput)
            err = my_lib.FilterInterpolationLayer_cpu_backward(self.input1, self.input2,self.input3, gradoutput, gradinput1, gradinput2, gradinput3)
            # print(err)
            if err != 0 :
                print(err)
            # print(gradinput1)
            # print(gradinput2)

        # print(gradinput1)

        return gradinput1, gradinput2,gradinput3

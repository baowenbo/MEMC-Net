from torch.nn import Module
import torch
from torch.autograd import Variable
from torch.autograd import gradcheck
from my_package.functions.FilterInterpolationLayer import FilterInterpolationLayer

class FilterInterpolationModule(Module):
    def __init__(self):
        super(FilterInterpolationModule, self).__init__()
        self.f = FilterInterpolationLayer()

    def forward(self, input1, input2, input3):
        return self.f(input1, input2, input3)

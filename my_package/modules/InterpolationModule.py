# modules/InterpolationLayer.py
from torch.nn import Module
from my_package.functions.InterpolationLayer import InterpolationLayer

class InterpolationModule(Module):
    def __init__(self):
        super(InterpolationModule, self).__init__()
        self.f = InterpolationLayer()

    def forward(self, input1, input2):
        return self.f(input1, input2)



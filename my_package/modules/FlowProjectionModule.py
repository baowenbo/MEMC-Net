# modules/FlowProjectionModule.py
from torch.nn import Module
from my_package.functions.FlowProjectionLayer import FlowProjectionLayer #, FlowFillholeLayer

class FlowProjectionModule(Module):
    def __init__(self, requires_grad = True):
        super(FlowProjectionModule, self).__init__()

        self.f = FlowProjectionLayer(requires_grad)

    def forward(self, input1 ):
        return self.f(input1)


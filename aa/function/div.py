from .function import Function
import numpy as np
from ..variable import Variable


def div(x0, x1):
    return Div()(x0, x1)

def rdiv(x0, x1):
    return Div()(x1, x0)

class Div(Function):
    def forward(self, x0, x1):
        return x0/x1

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy/x1 , -gy*x0/x1**2

Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv

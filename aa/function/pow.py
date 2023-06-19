from ..core.function import Function
import numpy as np
from ..core.variable import Variable


def pow(x, c):
    return Pow(c)(x)

class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        return x**self.c

    def backward(self, gy):
        x, = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


Variable.__pow__ = pow
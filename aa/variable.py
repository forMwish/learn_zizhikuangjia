import numpy as np

class Variable:
    def __init__(self, x):
        self.data = x
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        func = self.creator
        while func:
            x = func.input
            y = func.output
            x.grad = func.backward(y.grad)
            func = x.creator
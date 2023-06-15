import numpy as np

class Variable:
    def __init__(self, x):
        if x is not None:
            if np.isscalar(x):
                x = np.array(x)
            if not isinstance(x, np.ndarray):
                raise TypeError(f"{type(x)} is not support")
        
        self.data = x
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad == None:
            self.grad = np.ones_like(self.data)
        func = self.creator
        while func:
            x = func.input
            y = func.output
            x.grad = func.backward(y.grad)
            func = x.creator
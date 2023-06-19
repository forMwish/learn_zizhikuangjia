import weakref
import numpy as np
from ..variable import Variable
from ..config import Config


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    elif np.isscalar(obj): # 针对传入浮点和定点值
        return Variable(np.array(obj))
    elif isinstance(obj, np.ndarray):
        return Variable(obj)
    else:
        raise TypeError (f" input type is {type(obj)}")

class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(y) for y in ys]
        
        if Config.enable_backprop:
            self.generation = max([input.generation for input in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]


        return outputs if len(outputs)>1 else outputs[0]

    def forward(self, xs):
        return NotImplementedError()

    def backward(self, gys):
        return NotImplementedError()
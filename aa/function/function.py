from ..variable import Variable

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(y) for y in ys]
        
        self.generation = max([input.generation for input in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs


        return outputs if len(outputs)>1 else outputs[0]

    def forward(self, xs):
        return NotImplementedError()

    def backward(self, gys):
        return NotImplementedError()
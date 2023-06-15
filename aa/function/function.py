from ..variable import Variable

class Function:
    def __call__(self, inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(y) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs

        return outputs

    def forward(self, xs):
        return NotImplementedError()

    def backward(self, gys):
        return NotImplementedError()
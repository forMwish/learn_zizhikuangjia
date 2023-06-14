from ..variable import Variable

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output

        return output

    def forward(self, x):
        return NotImplementedError()

    def backward(self, gy):
        return NotImplementedError()
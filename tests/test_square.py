import unittest
import numpy as np
from aa import Variable, square, numerical_diff


class SquareTest(unittest.TestCase):
    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)

        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)

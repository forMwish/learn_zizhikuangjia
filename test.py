from aa import *
import numpy as np


A = Square()
B = Exp()
C = Square()

x = Variable(0.5)

y0 = A(x)
y1 = B(y0)
y2 = C(y1)

y2.backward()

print(x.grad)

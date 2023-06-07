from aa import *

A = Square()
B = Exp()
x = Variable(np.array(0.5))

y0 = A(x)
y1 = B(y0)
y2 = A(y1)

print(y2.data)

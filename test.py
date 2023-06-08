from aa import *

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))

y0 = A(x)
y1 = B(y0)
y2 = C(y1)

y2.grad = np.array(1.0)
y2.backward()

print(x.grad)

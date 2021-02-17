import SimPy.Regression as Reg
import numpy as np
import matplotlib.pyplot as plt


# generate x values (observations)
x = np.random.randn(200)
# generate y values (assuming y = x^2 -2 + error)
y = pow(x, 2) - 4 * pow(x, 1) - 2 + np.random.randn(200)

polyReg = Reg.SingleVarPolyRegression(x, y, degree=2)
polyReg.plot_fit(x_range=[-3, 3])

print(polyReg.get_coeffs())

print(polyReg.get_predicted_y(0))
print(polyReg.get_predicted_y(1))
print(polyReg.get_predicted_y(-1))

print(polyReg.get_derivative(0))
print(polyReg.get_derivative(1))

print(polyReg.get_roots())

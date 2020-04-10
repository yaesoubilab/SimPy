import SimPy.RegressionClasses as Reg
import numpy as np
import matplotlib.pyplot as plt


# generate x values (observations)
x = np.random.randn(200)
# generate y values (assuming y = x^2 -2 + error)
y = pow(x,3) - 4 * pow(x, 2) + x - 2 + np.random.randn(200)

polyReg = Reg.PolyRegression(x, y, degree=3)

print(polyReg.get_coeffs())
print(polyReg.get_predicted_y(0))
print(polyReg.get_predicted_y(1))
print(polyReg.get_predicted_y(-1))

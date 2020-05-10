import SimPy.RegressionClasses as Reg

x = [1296, 680,529,471,444,426, 412, 407, 401, 238]
y = [1, 18, 29, 36, 42, 48, 57, 60, 67, 207]

reg = Reg.PowerRegression(x, y, if_c0_zero=True)
print(reg.get_coeffs())

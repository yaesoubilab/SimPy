import SimPy.Regression as Reg

x = [1296, 680, 529, 471, 444, 426, 412, 407, 401, 238]
y = [1, 18, 29, 36, 42, 48, 57, 60, 67, 207]

reg = Reg.PowerRegression(x, y, if_c0_zero=True)
reg_with_p0 = Reg.PowerRegression(x, y, if_c0_zero=True, p0=(1, 0))
reg.plot_fit()
reg_with_p0.plot_fit()
print(reg.get_coeffs())
print('\ncoefficients of model without initial value:', reg.get_coeffs())
print('coefficients of model with initial value', reg_with_p0.get_coeffs())

import numpy as np
import SimPy.Regression as Reg
from SimPy.Regression import ExpRegression as exp

# parameters of an exponential function
par1 = (3, 2, 0.5)  # 3 + 2*exp(0.5*x)
par2 = (2, 0.5)     # 2*exp(0.5*x)

# x values between
x = np.linspace(0, 4, 50)

# y values
y1 = exp.exp_func(x, *par1)
y2 = exp.exp_func_c0_zero(x, *par2)

# adding noice to y values
y1n = y1 + 0.2*np.random.normal(size=len(x))
y2n = y2 + 0.2*np.random.normal(size=len(x))

# fitting an exponential function
exp_reg_1 = Reg.ExpRegression(x, y1n, if_c0_zero=False)
exp_reg_2 = Reg.ExpRegression(x, y2n, if_c0_zero=True)

# display the fit
exp_reg_1.plot_fit()
exp_reg_2.plot_fit()

# print the estimated coefficients
print('\ncoefficients of y1 model :', exp_reg_1.get_coeffs(), 'v.s', par1)
print('prediction vs actual y1 at x=2 :', exp_reg_1.get_predicted_y(2), 'v.s', exp.exp_func(2, *par1))

print('\ncoefficients of y2 model :', exp_reg_2.get_coeffs(), 'v.s', par2)
print('prediction vs actual y2 at x=2: ', exp_reg_2.get_predicted_y(2), 'v.s', exp.exp_func_c0_zero(2, *par2))

# testing on with p0
x = [112.25, 728.185, 879.07, 937.03, 963.485, 982.22, 996.4, 1001.005,	1006.87, 1170.445]
y = [1.07, 17.68, 28.53, 35.675, 41.59, 48.05, 57.47, 59.605, 66.565, 206.935]

exp_reg_3 = Reg .ExpRegression(x, y, if_c0_zero=True)
exp_reg_3_with_p0 = Reg .ExpRegression(x, y, if_c0_zero=True, p0=[0.5, 0.005])
exp_reg_3.plot_fit()
exp_reg_3_with_p0.plot_fit()
print('\ncoefficients of y3 model (without initial value)', exp_reg_3.get_coeffs())
print('coefficients of y3 model (with initial value)', exp_reg_3_with_p0.get_coeffs())

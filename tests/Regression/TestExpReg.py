import numpy as np
import SimPy.RegressionClasses as Reg
from SimPy.RegressionClasses import ExpRegression as exp

par1 = (3, 2, 0.5)  # 3 + 2*exp(0.5*x)
par2 = (2, 0.5)     # 2*exp(0.5*x)

x = np.linspace(0, 4, 50)

y1 = exp.exp_func(x, *par1)
y2 = exp.exp_func_zero_at_limit(x, *par2)

y1n = y1 + 0.2*np.random.normal(size=len(x))
y2n = y2 + 0.2*np.random.normal(size=len(x))

exp_reg_1 = Reg.ExpRegression(x, y1n, if_zero_at_limit=False)
exp_reg_2 = Reg.ExpRegression(x, y2n, if_zero_at_limit=True)

print(exp_reg_1.get_coeffs(), 'v.s', par1)
print(exp_reg_1.get_predicted_y(2), 'v.s', exp.exp_func(2, *par1))

print(exp_reg_2.get_coeffs(), 'v.s', par2)
print(exp_reg_2.get_predicted_y(2), 'v.s', exp.exp_func_zero_at_limit(2, *par2))

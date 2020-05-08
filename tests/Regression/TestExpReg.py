import numpy as np
import SimPy.RegressionClasses as Reg
from SimPy.RegressionClasses import ExpRegression as exp

x = np.linspace(0, 4, 50)
y = exp.exp_func(x, 2.5, 1.3, 0.5)
yn = y + 0.2*np.random.normal(size=len(x))

exp_reg = Reg.ExpRegression(x, yn)
print(exp_reg.para)

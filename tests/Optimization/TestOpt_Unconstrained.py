import numpy as np

import SimPy.Optimization as Opt
from tests.Optimization import ToyModels

# create an object for the stochastic approximation method
mySimOpt = Opt.StochasticApproximation(
    sim_model=ToyModels.Xto2(err_sigma=10),
    derivative_step=Opt.StepSize_Df(c0=5),
    step_size=Opt.StepSize_GeneralizedHarmonic(a0=5, b=100))

# find the minimum
mySimOpt.minimize(max_itr=5000, n_last_itrs_to_ave=200, x0=np.array([20, 5]))

# plot x and objective function values
mySimOpt.plot_f_itr(f_star=0)
mySimOpt.plot_x_irs(x_stars=[-1, 0])
mySimOpt.plot_Df_irs()
mySimOpt.plot_step_move()
mySimOpt.plot_step_Df()


print('Optimal x: ', mySimOpt.xStar)
print('Optimal f: ', mySimOpt.fStar)


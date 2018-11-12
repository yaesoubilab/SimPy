import SimPy.Optimization as Opt
from tests.Optimization import ToyModels
import numpy as np

# create an object for the stochastic approximation method
mySimOpt = Opt.StochasticApproximation(
    sim_model=ToyModels.Xto2Constrained(err_sigma=1, penalty=10000),
    derivative_step=Opt.StepSize_Df(c0=5),
    step_size=Opt.StepSize_GeneralizedHarmonic(a0=5, b=10))

# find the minimum
mySimOpt.minimize(max_itr=5000, n_last_itrs_to_ave=200, x0=np.array([20, 5]))

# plot x and objective function values
mySimOpt.plot_f_itr(f_star=0)
mySimOpt.plot_x_irs(x_star=[0, 1])

print(mySimOpt.xStar, mySimOpt.fStar)

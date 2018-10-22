import SimPy.Optimization as Opt
from tests.Optimization import ToyModels
import numpy as np

# create an object for the stochastic approximation method
mySimOpt = Opt.StochasticApproximation(
    sim_model=ToyModels.Xto2(err_sigma=10),
    derivative_step=Opt.StepSize_e(e=1),
    step_size=Opt.StepSize_a(a=100),
penalty=Opt.Penalty(p=10))

# find the minimum
mySimOpt.minimize(max_itr=5000, nLastItrsToAve=200, x0=np.array([30, 5]),penalty_or_not=1)

# plot x and objective function values
mySimOpt.plot_f_itr(f_star=0)
mySimOpt.plot_x_irs(x_star=[0, 0])

print(mySimOpt.xStar,mySimOpt.fStar)

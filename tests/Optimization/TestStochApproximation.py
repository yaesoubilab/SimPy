import SimPy.Optimization as Opt
from tests.Optimization import ToyModels

# create an object for the stochastic approximation method
mySimOpt = Opt.StochasticApproximation(
    sim_model=ToyModels.Xto2(err_sigma=100),
    step_size=Opt.StepSize(a=1),
    derivative_step=1)

# find the minimum
mySimOpt.minimize(max_itr=1000, x0=5)

# plot x and objective function values
mySimOpt.plot_xs(xStar=0)
mySimOpt.plot_fs(yStar=0)

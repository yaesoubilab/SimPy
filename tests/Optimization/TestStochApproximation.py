import SimPy.Optimization as Opt
from tests.Optimization import ToyModels

mySimOpt = Opt.StochasticApproximation(
    sim_model=ToyModels.Xto2(err_sigma=10),
    step_size=Opt.StepSize(a=1),
    derivative_step=0.1)

mySimOpt.minimize(max_itr=1000, x0=5)
mySimOpt.plot_xs()
mySimOpt.plot_fs()

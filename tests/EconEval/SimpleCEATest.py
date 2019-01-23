from SimPy import EconEvalClasses as EV
import numpy as np


S0 = EV.Strategy(name='Base', cost_obs=[100], effect_obs=[1])
S1 = EV.Strategy(name='A1', cost_obs=[800], effect_obs=[0.5])
S2 = EV.Strategy(name='A2', cost_obs=[2000], effect_obs=[10])
S3 = EV.Strategy(name='A3', cost_obs=[500], effect_obs=[6])
S4 = EV.Strategy(name='A4', cost_obs=[-100], effect_obs=[2])

cea = EV.CEA_paired(strategies=[S0, S1, S2, S3, S4], health_measure=EV.HealthMeasure.UTILITY)

print('On frontier')
for s in cea.get_strategies_on_frontier():
    print(s.name)

print('Not on frontier')
for s in cea.get_strategies_not_on_frontier():
    print(s.name)

cea.show_CE_plane('CE plane', 'E[Effect]', 'E[Cost]',
                  show_names=True, figure_size=6)
cea.build_CE_table(cost_digits=0, interval=EV.Interval.NO_INTERVAL)
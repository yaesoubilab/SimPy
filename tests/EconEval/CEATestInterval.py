from SimPy import EconEvalClasses as EV
import numpy as np

np.random.seed(seed=1)

S0 = EV.Strategy(name='Base',
                 cost_obs=np.random.normal(loc=100, scale=10, size=100),
                 effect_obs=np.random.normal(loc=1, scale=.2, size=100))
S1 = EV.Strategy(name='A1',
                 cost_obs=np.random.normal(loc=800, scale=50, size=100),
                 effect_obs=np.random.normal(loc=0.5, scale=0.1, size=100))
S2 = EV.Strategy(name='A2',
                 cost_obs=np.random.normal(loc=2000, scale=200, size=100),
                 effect_obs=np.random.normal(loc=10, scale=1, size=100))
S3 = EV.Strategy(name='A3',
                 cost_obs=np.random.normal(loc=500, scale=50, size=100),
                 effect_obs=np.random.normal(loc=6, scale=1, size=100))
S4 = EV.Strategy(name='A4',
                 cost_obs=np.random.normal(loc=-100, scale=10, size=100),
                 effect_obs=np.random.normal(loc=2, scale=0.1, size=100))

cea = EV.CEA_paired(strategies=[S0, S1, S2, S3, S4], health_measure=EV.HealthMeasure.UTILITY)

print('On frontier')
for s in cea.get_strategies_on_frontier():
    print(s.name)

print('Not on frontier')
for s in cea.get_strategies_not_on_frontier():
    print(s.name)

cea.show_CE_plane('CE plane', 'E[Effect]', 'E[Cost]',
                  show_names=True, show_clouds=False, figure_size=6)
cea.build_CE_table(cost_digits=1, interval=EV.Interval.PREDICTION)
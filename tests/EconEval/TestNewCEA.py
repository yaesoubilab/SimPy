from SimPy import EconEval as EV


S0 = EV.Strategy(name='Base', cost_obs=[100], effect_obs=[1])
S1 = EV.Strategy(name='A1', cost_obs=[800], effect_obs=[0.5])
S2 = EV.Strategy(name='A2', cost_obs=[2000], effect_obs=[10])
S3 = EV.Strategy(name='A3', cost_obs=[500], effect_obs=[7])
S4 = EV.Strategy(name='A4', cost_obs=[-100], effect_obs=[2])
S5 = EV.Strategy(name='A5', cost_obs=[200], effect_obs=[2])
S6 = EV.Strategy(name='A6', cost_obs=[1000], effect_obs=[7.2])
S7 = EV.Strategy(name='A7', cost_obs=[1100], effect_obs=[7.3])

cea = EV.CEA(strategies=[S0, S1, S2, S3, S4, S5, S6, S7], if_paired=False, health_measure='u')

# show the ce plane:
cea.show_CE_plane()

# build table
cea.build_CE_table(interval_type='n')

print('On frontier')
frontier_strategies = cea.get_strategies_on_frontier()
for i, s in enumerate(frontier_strategies):
    print(s.name)
    if i>0:
        print('incCost:', s.incCost.get_mean())
        print('incEffect:', s.incEffect.get_mean())
        print('ICER:', s.icer.get_ICER())
#
# print('Not on frontier')
# for s in cea.get_strategies_not_on_frontier():
#     print(s.name)
#
# cea.show_CE_plane('CE plane', 'E[Effect]', 'E[Cost]',
#                   show_names=True, figure_size=6)
# cea.build_CE_table(cost_digits=0, interval_type='n')
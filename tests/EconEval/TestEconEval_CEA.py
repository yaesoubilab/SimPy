import numpy as np

from SimPy import EconEval as ce

np.random.seed(573)
# create the centers of strategies
s_center = np.random.normal(0, 5000, (10, 2))

s0 = ce.Strategy("s1", s_center[0, 0]+np.random.normal(0, 200, 10), s_center[0, 1]+np.random.normal(0, 200, 10))
s1 = ce.Strategy("s2", s_center[1, 0]+np.random.normal(0, 200, 10), s_center[1, 1]+np.random.normal(0, 100, 10))
s2 = ce.Strategy("s3", s_center[2, 0]+np.random.normal(0, 200, 10), s_center[2, 1]+np.random.normal(0, 200, 10))
s3 = ce.Strategy("s4", s_center[3, 0]+np.random.normal(0, 200, 10), s_center[3, 1]+np.random.normal(0, 200, 10))
s4 = ce.Strategy("s5", s_center[4, 0]+np.random.normal(0, 200, 10), s_center[4, 1]+np.random.normal(0, 200, 10))
s5 = ce.Strategy("s6", s_center[5, 0]+np.random.normal(0, 200, 10), s_center[5, 1]+np.random.normal(0, 200, 10))
s6 = ce.Strategy("s7", s_center[6, 0]+np.random.normal(0, 200, 10), s_center[6, 1]+np.random.normal(0, 200, 10))
s7 = ce.Strategy("s8", s_center[7, 0]+np.random.normal(0, 200, 10), s_center[7, 1]+np.random.normal(0, 200, 10))
s8 = ce.Strategy("s9", s_center[8, 0]+np.random.normal(0, 200, 10), s_center[8, 1]+np.random.normal(0, 200, 10))
s9 = ce.Strategy("s10", s_center[9, 0]+np.random.normal(0, 200, 10), s_center[9, 1]+np.random.normal(0, 200, 10))

# create a CEA object -- unpaired
myCEA = ce.CEA([s0, s1, s2, s3, s4, s5, s6, s7, s8, s9], if_paired=False)

# plot with label and sample cloud
myCEA.plot_CE_plane('CE plane with unpaired observations and showing labels',
                    x_label='E[Effect]', y_label='E[Cost]', show_legend=True, add_clouds=True, fig_size=(6, 6))

# table
print('')
myCEA.build_CE_table(interval_type='c',
                     cost_digits=0, effect_digits=0, icer_digits=1,
                     file_name='Table-Indp.csv')

# plot with label and sample cloud
myCEA.plot_CE_plane('CE plane with paired observations and showing labels',
                    x_label='E[Effect]', y_label='E[Cost]', show_legend=True, add_clouds=True, fig_size=(6, 6))

dict = myCEA.get_dCost_dEffect_cer(interval_type='c', alpha=0.05,
                                   cost_digits=0, effect_digits=0, icer_digits=1)
print('dCost, dEffect, CER')
for s in dict:
    print(s, dict[s])

# create a CEA object -- paired
myCEA2 = ce.CEA([s0, s1, s2, s3, s4, s5, s6, s7, s8, s9], if_paired=True)


# frontier results
print('')
print('Strategies on the frontier:')
frontier = myCEA2.get_strategies_on_frontier()
for s in frontier:
    print('name:', s.name)
    print('cost', s.cost.get_mean())
    print('dCost', s.dCost.get_mean())
    if s.incCost is None:
        print('incCost', None)
    else:
        print('incCost', s.incCost.get_mean())






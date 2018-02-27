from scr import EconEvalClasses as ce
import numpy as np


np.random.seed(573)
s_center = np.random.normal(0, 5, (10, 2))

s0 = ce.Strategy("s1", s_center[0, 0]+np.random.normal(0, 0.5, 10), s_center[0, 1]+np.random.normal(0, 0.5, 10))
s1 = ce.Strategy("s2", s_center[1, 0]+np.random.normal(0, 0.5, 10), s_center[1, 1]+np.random.normal(0, 0.5, 10))
s2 = ce.Strategy("s3", s_center[2, 0]+np.random.normal(0, 0.5, 10), s_center[2, 1]+np.random.normal(0, 0.5, 10))
s3 = ce.Strategy("s4", s_center[3, 0]+np.random.normal(0, 0.5, 10), s_center[3, 1]+np.random.normal(0, 0.5, 10))
s4 = ce.Strategy("s5", s_center[4, 0]+np.random.normal(0, 0.5, 10), s_center[4, 1]+np.random.normal(0, 0.5, 10))
s5 = ce.Strategy("s6", s_center[5, 0]+np.random.normal(0, 0.5, 10), s_center[5, 1]+np.random.normal(0, 0.5, 10))
s6 = ce.Strategy("s7", s_center[6, 0]+np.random.normal(0, 0.5, 10), s_center[6, 1]+np.random.normal(0, 0.5, 10))
s7 = ce.Strategy("s8", s_center[7, 0]+np.random.normal(0, 0.5, 10), s_center[7, 1]+np.random.normal(0, 0.5, 10))
s8 = ce.Strategy("s9", s_center[8, 0]+np.random.normal(0, 0.5, 10), s_center[8, 1]+np.random.normal(0, 0.5, 10))
s9 = ce.Strategy("s10", s_center[9, 0]+np.random.normal(0, 0.5, 10), s_center[9, 1]+np.random.normal(0, 0.5, 10))

# create a CEA object -- unpaired
myCEA = ce.CEA([s0, s1, s2, s3, s4, s5, s6, s7, s8, s9], if_paired=True)

# plot with label and sample cloud
myCEA.show_CE_plane('CE plane with unpaired observations and showing labels',
                    'E[Effect]', 'E[Cost]', show_names=True, show_clouds=True, figure_size=6)

# plot with sample cloud and legend
myCEA.show_CE_plane('CE Plane with unpaired observations and showing legend',
                    'E[Effect]', 'E[Cost]', show_legend=True, show_clouds=True, figure_size=5)

# plot with no label and sample cloud
myCEA.show_CE_plane('CE Plane with unpaired observations and no clouds',
                    'E[Effect]','E[Cost]', show_clouds=False, show_names=True)

# table
print('')
# return none and write result into csv
print(myCEA.build_CE_table(ce.CETableInterval.PREDICTION))


# frontier results
print('')
print('Strategies on the frontier:')
frontier = myCEA.get_frontier()
for s in frontier:
    print(s.name)


# create a CEA object -- paired
myCEA2 = ce.CEA([s0, s1, s2, s3, s4, s5, s6, s7, s8, s9], if_paired=True)

# plot with label and sample cloud
myCEA2.show_CE_plane('CE plane with paired observations and showing labels',
                     'E[Effect]', 'E[Cost]', show_names=True, show_clouds=True, figure_size=6)

# plot with sample cloud and legend
myCEA2.show_CE_plane('CE Plane with paired observations and showing legend',
                     'E[Effect]', 'E[Cost]', show_legend=True, show_clouds=True, figure_size=5)

# plot with no label and sample cloud
myCEA2.show_CE_plane('CE Plane with paired observations and no clouds',
                     'E[Effect]','E[Cost]', show_clouds=False, show_names=True)

# frontier results
print('')
print('Strategies on the frontier:')
frontier = myCEA2.get_frontier()
for s in frontier:
    print(s.name)







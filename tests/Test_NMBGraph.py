from scr import EconEvalClasses as ce
import numpy as np


np.random.seed(573)
s_center = np.random.normal(0, 5, (10, 2))

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

# create a CEA object and get the frontier stategies
myCEA = ce.CEA([s0, s1, s2, s3, s4, s5, s6, s7, s8, s9], if_paired=False)
nmb_paired = ce.CBA(myCEA.get_strategies_on_frontier(), if_paired=True) # list of frontier strategies as input
nmb_indp = ce.CBA(myCEA.get_strategies_on_frontier(), if_paired=False) # list of frontier strategies as input

# Try NMB_Lines figure
nmb_paired.graph_deltaNMB_lines(1,10,"deltaNMB lines","wtp values","NMB values",interval=ce.Interval.CONFIDENCE,
                             show_legend=True, figure_size=8)

nmb_indp.graph_deltaNMB_lines(1,10,"deltaNMB lines","wtp values","NMB values",interval=ce.Interval.PREDICTION,
                             show_legend=True, figure_size=8)



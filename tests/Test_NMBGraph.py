from scr import EconEvalClasses as ce
import numpy as np


np.random.seed(573)
s_center = np.array([[10000, 0.2],[20000, 0.3],[50000, 0.35]])


s0 = ce.Strategy("s1", s_center[0, 0]+np.random.normal(0, 2000, 10), s_center[0, 1]+np.random.normal(0, 0.01, 10))
s1 = ce.Strategy("s1", s_center[1, 0]+np.random.normal(0, 2000, 10), s_center[1, 1]+np.random.normal(0, 0.01, 10))
s2 = ce.Strategy("s2", s_center[2, 0]+np.random.normal(0, 2000, 10), s_center[2, 1]+np.random.normal(0, 0.01, 10))


nmb_paired = ce.CBA([s0, s1, s2], if_paired=True) # list of frontier strategies as input
nmb_indp = ce.CBA([s0, s1, s2], if_paired=False) # list of frontier strategies as input

# Try NMB_Lines figure - paired CI
nmb_paired.graph_deltaNMB_lines(0,50000,"deltaNMB lines for paired CI","wtp values","NMB values",
                                interval=ce.Interval.CONFIDENCE,
                                show_legend=True, figure_size=8)

# Try NMB_Lines figure - paired PI
nmb_paired.graph_deltaNMB_lines(0,50000,"deltaNMB lines for paired PI","wtp values","NMB values",
                                interval=ce.Interval.PREDICTION,
                                show_legend=True, figure_size=8)

# Try NMB_Lines figure - indp CI
nmb_indp.graph_deltaNMB_lines(0,50000,"deltaNMB lines for indp CI","wtp values","NMB values",
                              interval=ce.Interval.CONFIDENCE,
                              show_legend=True, figure_size=8)

# Try NMB_Lines figure - indp PI
nmb_indp.graph_deltaNMB_lines(0,50000,"deltaNMB lines for indp PI","wtp values","NMB values",
                              interval=ce.Interval.PREDICTION,
                              show_legend=True, figure_size=8)



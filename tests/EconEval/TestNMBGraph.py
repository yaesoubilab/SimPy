from SimPy import EconEval as ce
import numpy as np

N = 100

np.random.seed(573)
s_center = np.array([[10000, 0.2], [20000, 0.7], [50000, 1.2]])


s0 = ce.Strategy("s0", s_center[0, 0]+np.random.normal(0, 1000, N),
                 s_center[0, 1]+np.random.normal(0, 0.01, N),
                 color='red')
s1 = ce.Strategy("s1", s_center[1, 0]+np.random.normal(0, 1000, N),
                 s_center[1, 1]+np.random.normal(0, 0.01, N),
                 color='blue')
s2 = ce.Strategy("s2", s_center[2, 0]+np.random.normal(0, 1000, N),
                 s_center[2, 1]+np.random.normal(0, 0.05, N),
                 color='green')

cea = ce.CEA([s0, s1, s2], if_paired=True)
cea.show_CE_plane()

nmb_paired = ce.CBA([s0, s1, s2],
                    wtp_range=[0, 100000],
                    if_paired=True)
nmb_indp = ce.CBA([s0, s1, s2],
                  wtp_range=[0, 100000],
                  if_paired=False)  # list of frontier strategies as input

# Try NMB_Lines figure - paired CI
nmb_paired.graph_incremental_nmbs(
    title="deltaNMB lines for paired CI",
    x_label="wtp values",
    y_label="NMB values",
    y_axis_multiplier=0.1,
    interval_type='c',
    show_legend=True,
    figure_size=(6, 5))

# Try NMB_Lines figure - paired PI
nmb_paired.graph_incremental_nmbs(
    "deltaNMB lines for paired PI", "wtp values", "NMB values",
    interval_type='p',
    show_legend=True,
    figure_size=(6, 5))

# Try NMB_Lines figure - indp CI
nmb_indp.graph_incremental_nmbs(
    "deltaNMB lines for indp CI", "wtp values", "NMB values",
    interval_type='c',
    show_legend=True,
    figure_size=(6, 5))

# Try NMB_Lines figure - indp PI
nmb_indp.graph_incremental_nmbs(
    "deltaNMB lines for indp PI", "wtp values", "NMB values",
    interval_type='p',
    show_legend=True,
    figure_size=(6, 5))



import numpy as np
import scr.FigureSupport as cls

obs = np.random.normal(4, 3, 1000)
cls.graph_histogram(
    data=obs,
    title='Histogram',
    x_label='Values',
    y_label='Counts',
    x_range=[-5, 20],
    y_range=[0, 140],
    legend='Number of patients')

obs_sets = [
    np.random.normal(4, 3, 1000),
    np.random.normal(8, 3, 1000)
]

cls.graph_histograms(
    data_sets=obs_sets,
    title='Two histograms',
    x_label='Values',
    y_label='Counts',
    legend=['H 1', 'H 2'],
    bin_width=0.5,
    x_range=[-10, 20],
    y_range=[0, 100],
    transparency=0.6
)
import numpy as np
import SimPy.Plots.FigSupport as cls

obs = np.random.normal(4, 3, 1000)

cls.plot_histogram(
    data=obs,
    title='Histogram',
    x_label='Values',
    y_label='Counts',
    color='g',
    bin_width=1,
    x_range=[-5, 20],
    y_range=[0, 140],
    legend='Number of patients')

obs_sets = [
    np.random.normal(4, 3, 1000),
    np.random.normal(8, 3, 1000)
]

cls.plot_histograms(
    data_sets=obs_sets,
    title='Two histograms',
    x_label='Values',
    y_label='Counts',
    legends=['H 1', 'H 2'],
    bin_width=1,
    x_range=[-10, 20],
    #y_range=[0, 100],
    color_codes=['blue', 'green'],
    transparency=0.6
)
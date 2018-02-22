import numpy as np
import scr.FigureSupport as cls

obs = np.random.normal(4, 3, 1000)
mygraph = cls.graph_histogram(
    observations=obs,
    title='Histogram',
    x_label='Values',
    y_label='Counts',
    x_range=[-5,20],
    legend='Number of patients')

import numpy as np
import scr.HistogramFunctions as cls


obs = np.random.normal(10, 2, 1000)
cls.graph_histogram(
    observations=obs,
    title='Histogram',
    x_label='Values',
    y_label='Counts',
    legend='Number of patients')

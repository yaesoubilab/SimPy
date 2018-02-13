import numpy as np
import scr.HistogramFunctions as cls


obs = np.random.normal(10, 2, 1000)
cls.graph_histogram(obs, 'Histogram', 'Values', 'Counts', cls.OutType.SHOW, 'Number of patients')

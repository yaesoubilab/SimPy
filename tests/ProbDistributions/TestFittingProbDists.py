import numpy as np
import SimPy.RandomVariantGenerators as RVGs
from tests.ProbDistributions.RVGtests import get_samples
import SimPy.Plots.ProbDist as Plot

# simulate some data
np.random.seed(1)

# fitting a beta distribution
dist = RVGs.Beta(a=2, b=3, loc=1, scale=2)
data = np.array(get_samples(dist, np.random))
# method of moment
dictMMResults = RVGs.Beta.get_fit_mm(
    mean=np.mean(data),
    st_dev=np.std(data),
    minimum=1,
    maximum=3)
# maximum likelihood
dictMLResults = RVGs.Beta.get_fit_ml(data=data, minimum=1, maximum=3)

print("Fitting Beta with a=2, b=3, loc=1, scale=2:")
print("  MM:", dictMMResults)
print("  ML:", dictMLResults)

# plot the fitted distributions
Plot.plot_beta_fit(data=data, fit_results=dictMMResults, title='Method of Moment', x_label='Data')
Plot.plot_beta_fit(data=data, fit_results=dictMLResults, title='Maximum Likelihood', x_label='Data')


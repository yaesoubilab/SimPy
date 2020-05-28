import numpy as np
import SimPy.RandomVariantGenerators as RVGs
from tests.ProbDistributions.RVGtests import get_samples
import SimPy.Plots.ProbDist as Plot


def test_fitting_beta():
    dist = RVGs.Beta(a=2, b=3, loc=1, scale=2)
    data = np.array(get_samples(dist, np.random))
    # method of moment
    dict_mm_results = RVGs.Beta.fit_mm(
        mean=np.mean(data), st_dev=np.std(data), minimum=1, maximum=3)
    # maximum likelihood
    dict_ml_results = RVGs.Beta.fit_ml(data=data, minimum=1, maximum=3)

    print("Fitting Beta with a=2, b=3, loc=1, scale=2:")
    print("  MM:", dict_mm_results)
    print("  ML:", dict_ml_results)

    # plot the fitted distributions
    Plot.plot_beta_fit(data=data, fit_results=dict_mm_results, title='Method of Moment', x_label='Data')
    Plot.plot_beta_fit(data=data, fit_results=dict_ml_results, title='Maximum Likelihood', x_label='Data')


def test_fitting_beta_binomial():

    dist = RVGs.BetaBinomial(n=20, a=2, b=3, loc=1)
    data = np.array(get_samples(dist, np.random))
    # method of moment
    dict_mm_results = RVGs.BetaBinomial.fit_mm(
        mean=np.mean(data), st_dev=np.std(data), n=20, fixed_location=1)
    # maximum likelihood
    dict_ml_results = RVGs.BetaBinomial.get_fit_ml(data=data, fixed_location=1)

    print("Fitting Beta-Binomial with n=100, a=2, b=3, loc=1, scale=2:")
    print("  MM:", dict_mm_results)
    print("  ML:", dict_ml_results)

    # plot the fitted distributions
    Plot.plot_beta_binomial_fit(data=data, fit_results=dict_mm_results, title='Method of Moment', x_label='Data')
    Plot.plot_beta_binomial_fit(data=data, fit_results=dict_ml_results, title='Maximum Likelihood', x_label='Data')

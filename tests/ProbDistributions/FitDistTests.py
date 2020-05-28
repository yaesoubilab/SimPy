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
    Plot.plot_beta_fit(data=data, fit_results=dict_mm_results, title='Method of Moment')
    Plot.plot_beta_fit(data=data, fit_results=dict_ml_results, title='Maximum Likelihood')


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
    Plot.plot_beta_binomial_fit(data=data, fit_results=dict_mm_results, title='Method of Moment')
    Plot.plot_beta_binomial_fit(data=data, fit_results=dict_ml_results, title='Maximum Likelihood')


def test_fitting_binomial():
    dist = RVGs.Binomial(n=100, p=0.3, loc=1)
    data = np.array(get_samples(dist, np.random))
    dict_mm_results = RVGs.Binomial.fit_mm(
        mean=np.mean(data), st_dev=np.std(data), fixed_location=1)

    print("Fitting Binomial with n=100, p=0.3, loc=1:")
    print("  MM:", dict_mm_results)


def test_fitting_empirical():
    dist = RVGs.Empirical(probabilities=[0.1, 0.2, 0.7])
    data = np.array(get_samples(dist, np.random))
    dict_mm_results = RVGs.Empirical.fit_mm(
        data=data, bin_size=1)

    print("Fitting empirical with p=[0.1, 0.2, 0.7]")
    print("  MM:", dict_mm_results)


def test_fitting_exponential():
    dist = RVGs.Exponential(scale=0.5, loc=2)
    data = np.array(get_samples(dist, np.random))
    dict_mm_results = RVGs.Exponential.fit_mm(
        mean=np.average(data), fixed_location=2)

    print("Fitting Exponential with scale=0.5, loc=2")
    print("  MM:", dict_mm_results)


def test_fitting_gamma():
    dist = RVGs.Gamma(a=10, scale=1, loc=2)
    data = np.array(get_samples(dist, np.random))
    dict_mm_results = RVGs.Gamma.fit_mm(
        mean=np.average(data), st_dev=np.std(data), fixed_location=2)

    print("Fitting Gamma with a=10, scale=1, loc=2")
    print("  MM:", dict_mm_results)


def test_fitting_gamma_poisson():
    dist = RVGs.GammaPoisson(a=2, gamma_scale=4, scale=2, loc=1)
    data = np.array(get_samples(dist, np.random))
    dict_mm_results = RVGs.GammaPoisson.fit_mm(
        mean=np.average(data), st_dev=np.std(data), fixed_scale=2, fixed_location=1)

    print("Fitting Gamma Poisson with a=2, gamma_scale=4, scale=2, loc=1")
    print("  MM:", dict_mm_results)


def test_fitting_poisson():
    dist = RVGs.Poisson(mu=100, loc=10)
    data = np.array(get_samples(dist, np.random))
    dict_mm_results = RVGs.Poisson.fit_mm(mean=np.average(data), fixed_location=10)

    print("Fitting Poisson with mean=100 and loc = 10")
    print("  MM:", dict_mm_results)

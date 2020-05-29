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

    # plot the fitted distributions
    Plot.plot_binomial_fit(data=data, fit_results=dict_mm_results, title='Method of Moment')


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

    # plot the fitted distributions
    Plot.plot_exponential_fit(data=data, fit_results=dict_mm_results, title='Method of Moment')


def test_fitting_gamma():
    dist = RVGs.Gamma(a=10, scale=1, loc=2)
    data = np.array(get_samples(dist, np.random))
    dict_mm_results = RVGs.Gamma.fit_mm(
        mean=np.average(data), st_dev=np.std(data), fixed_location=2)

    print("Fitting Gamma with a=10, scale=1, loc=2")
    print("  MM:", dict_mm_results)

    # plot the fitted distributions
    Plot.plot_gamma_fit(data=data, fit_results=dict_mm_results, title='Method of Moment')


def test_fitting_gamma_poisson():
    dist = RVGs.GammaPoisson(a=2, gamma_scale=4, loc=2)
    data = np.array(get_samples(dist, np.random))
    dict_mm_results = RVGs.GammaPoisson.fit_mm(
        mean=np.average(data), st_dev=np.std(data), fixed_location=2)

    print("Fitting Gamma Poisson with a=2, gamma_scale=4, loc=2")
    print("  MM:", dict_mm_results)

    # plot the fitted distributions
    Plot.plot_gamma_poisson_fit(data=data, fit_results=dict_mm_results, title='Method of Moment')


def test_fitting_geometric():
    dist = RVGs.Geometric(p=0.3, loc=1)
    data = np.array(get_samples(dist, np.random))
    dict_mm_results = RVGs.Geometric.fit_mm(
        mean=np.average(data), fixed_location=1)

    print("Fitting Geometric with p=0.3, loc=1")
    print("  MM:", dict_mm_results)


def test_fitting_lognormal():
    dist = RVGs.LogNormal(mu=0.2, sigma=0.1, loc=1)
    data = np.array(get_samples(dist, np.random))
    dict_mm_results = RVGs.LogNormal.fit_mm(
        mean=np.average(data), st_dev=np.std(data), fixed_location=1)

    print("Fitting LogNormal with mu=0.2, sigma=0.1, loc=1")
    print("  MM:", dict_mm_results)


def test_fitting_negbinomial():
    dist = RVGs.NegativeBinomial(n=10, p=0.2, loc=1)
    data = np.array(get_samples(dist, np.random))
    dict_mm_results = RVGs.NegativeBinomial.fit_mm(
        mean=np.average(data), st_dev=np.std(data), fixed_location=1)

    print("Fitting NegBinomial with n=10, p=0.2, loc=1")
    print("  MM:", dict_mm_results)


def test_fitting_poisson():
    dist = RVGs.Poisson(mu=100, loc=10)
    data = np.array(get_samples(dist, np.random))
    dict_mm_results = RVGs.Poisson.fit_mm(mean=np.average(data), fixed_location=10)

    print("Fitting Poisson with mean=100 and loc = 10")
    print("  MM:", dict_mm_results)


def test_fitting_uniform():
    dist = RVGs.Uniform(scale=10, loc=1)
    data = np.array(get_samples(dist, np.random))
    dict_mm_results = RVGs.Uniform.fit_mm(mean=np.average(data), st_dev=np.std(data))

    print("Fitting uniform with scale=10, loc=1")
    print("  MM:", dict_mm_results)


def test_fitting_uniform_discrete():
    dist = RVGs.UniformDiscrete(l=10, u=18)
    data = np.array(get_samples(dist, np.random))
    dict_mm_results = RVGs.UniformDiscrete.fit_mm(mean=np.average(data), st_dev=np.std(data))

    print("Fitting uniform discrete with l=10, u=18")
    print("  MM:", dict_mm_results)


def test_fitting_weibull():
    dist = RVGs.Weibull(a=5, scale=2, loc=1)
    data = np.array(get_samples(dist, np.random))
    dict_mm_results = RVGs.Weibull.fit_mm(
        mean=np.average(data), st_dev=np.std(data), fixed_location=1)

    print("Fitting Weibull with a=5, scale=2, loc=1")
    print("  MM:", dict_mm_results)

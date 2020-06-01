import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats as scs
from scipy.optimize import fmin_slsqp
import SimPy.RandomVariantGenerators as RVGs

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------------------------------
# This module contains procedures to fit probability distributions to the data using maximum likelihood approaches.
# Functions to fit probability distributions of non-negative random variables (e.g. exponential and gamma) take an
# fixed_location as an argument (with a default value set to 0). Specifying this argument allows for
# fitting a shifted distribution to the data.
# -------------------------------------------------------------------------------------------------

COLOR_CONTINUOUS_FIT = 'r'
COLOR_DISCRETE_FIT = 'r'
MIN_PROP = 0.005
MAX_PROB = 0.995
MAX_PROB_DISCRETE = 0.999999


def AIC(k, log_likelihood):
    """ :returns Akaike information criterion"""
    return 2 * k - 2 * log_likelihood


def find_bins(data, bin_width):
    # find bins
    if bin_width is None:
        bins = 'auto'
    else:
        bins = np.arange(min(data), max(data) + bin_width, bin_width)
    return bins




# GammaPoisson
def fit_gamma_poisson(data, x_label, fixed_location=0, fixed_scale=1, figure_size=5, bin_width=None):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param figure_size: int, specify the figure size
    :param bin_width: bin width
    :returns: dictionary with keys "a", "scale" and "AIC"
    """
    data = 1 * (data - fixed_location) / fixed_scale

    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(figure_size+1, figure_size))
    ax.hist(data, density=True, bins=find_bins(data, bin_width), edgecolor='black', alpha=0.5, label='Frequency')

    # Maximum-Likelihood Algorithm
    # ref: https://en.wikipedia.org/wiki/Negative_binomial_distribution#Gamma%E2%80%93Poisson_mixture
    n = len(data)

    # define density function of gamma-poisson
    def gamma_poisson(r,p,k):
        part1 = 1.0*sp.special.gamma(r+k)/(sp.special.gamma(r) * sp.special.factorial(k))
        part2 = (p**k)*((1-p)**r)
        return part1*part2

    # define log_likelihood function: sum of log(density) for each data point
    def log_lik(theta):
        r, p = theta[0], theta[1]
        result = 0
        for i in range(n):
            result += np.log(gamma_poisson(r,p,data[i]))
        return result

    # define negative log-likelihood, the target function to minimize
    def neg_loglik(theta):
        return -log_lik(theta)

    # estimate the parameters by minimize negative log-likelihood
    # initialize parameters
    theta0 = [2, 0.5]
    # call Scipy optimizer to minimize the target function
    # with bounds for p [0,1] and r [0,10]
    paras, value, iter, imode, smode = fmin_slsqp(neg_loglik, theta0, bounds=[(0.0, 10.0), (0,1)],
                              disp=False, full_output=True)

    # get the Maximum-Likelihood estimators
    a = paras[0]
    scale = paras[1]/(1.0-paras[1])

    # plot the estimated distribution
    # calculate PMF for each data point using newly estimated parameters
    x_values = np.arange(0, np.max(data), step=1)
    pmf = np.zeros(len(x_values))
    for i in x_values:
        pmf[int(i)] = gamma_poisson(paras[0], paras[1], i)

    pmf = np.append([0], pmf[:-1])

    # plot PMF
    ax.step(x_values, pmf, color=COLOR_CONTINUOUS_FIT, lw=2, label='GammaPoisson')

    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0%}'))
    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()

    # calculate AIC
    aic = AIC(
        k=2,
        log_likelihood=log_lik(paras)
    )

    # report results in the form of a dictionary
    return {"a": a, "gamma_scale": scale, "loc": fixed_location, "scale": fixed_scale, "AIC": aic}



# NegativeBinomial
def fit_negative_binomial(data, x_label, fixed_location=0, figure_size=5, bin_width=None):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param fixed_location: fixed location
    :param figure_size: int, specify the figure size
    :param bin_width: bin width
    :returns: dictionary with keys "n", "p" and "AIC"
    """
    # n is the number of successes, p is the probability of a single success.

    data = data-fixed_location

    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(figure_size+1, figure_size))
    ax.hist(data, normed=1, bins=find_bins(data, bin_width), # bins=np.max(data)+1, range=[-0.5, np.max(data)+0.5],
            edgecolor='black', alpha=0.5, label='Frequency')

    # Maximum-Likelihood Algorithm
    M = np.max(data)  # bound
    # define log_likelihood for negative-binomial, sum(log(pmf))
    def log_lik(theta):
        n, p = theta[0], theta[1]
        result = 0
        for i in range(len(data)):
            result += scs.nbinom.logpmf(data[i], n, p)
        return result

    # define negative log-likelihood, the target function to minimize
    def neg_loglik(theta):
        return -log_lik(theta)

    # estimate the parameters by minimize negative log-likelihood
    # initialize parameters
    theta0 = [2, 0.5]
    # call Scipy optimizer to minimize the target function
    # with bounds for p [0,1] and n [0,M]
    paras, value, iter, imode, smode = fmin_slsqp(neg_loglik, theta0, bounds=[(0.0, M), (0,1)],
                              disp=False, full_output=True)

    # plot the estimated distribution
    # calculate PMF for each data point using newly estimated parameters
    x_values = np.arange(0, np.max(data), step=1)
    rv = scs.nbinom(paras[0],paras[1])

    y_plot = rv.pmf(x_values)
    y_plot = np.append([0], y_plot[:-1])

    # plot PMF
    ax.step(x_values, y_plot, color=COLOR_CONTINUOUS_FIT, lw=2, label='NegativeBinomial')

    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0%}'))
    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()

    # calculate AIC
    aic = AIC(
        k=2,
        log_likelihood=log_lik(paras)
    )

    # report results in the form of a dictionary
    return {"n": paras[0], "p": paras[1], "loc": fixed_location, "AIC": aic}





# Triangular
def fit_triang(data, x_label, fixed_location=0, figure_size=5, bin_width=None):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param fixed_location: fixed location
    :param figure_size: int, specify the figure size
    :param bin_width: bin width
    :returns: dictionary with keys "c", "loc", "scale", and "AIC"
    """
    # The triangular distribution can be represented with an up-sloping line from
    # loc to (loc + c*scale) and then downsloping for (loc + c*scale) to (loc+scale).

    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(figure_size+1, figure_size))
    ax.hist(data, normed=1, bins=find_bins(data, bin_width), edgecolor='black', alpha=0.5, label='Frequency')

    # estimate the parameters
    c, loc, scale = scs.triang.fit(data, floc=fixed_location)

    # plot the estimated distribution
    x_values = np.linspace(scs.triang.ppf(MIN_PROP, c, loc, scale), scs.triang.ppf(MAX_PROB, c, loc, scale), 200)
    rv = scs.triang(c, loc, scale)
    ax.plot(x_values, rv.pdf(x_values), color=COLOR_CONTINUOUS_FIT, lw=2, label='Triangular')

    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0%}'))
    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()

    # calculate AIC
    aic = AIC(
        k=2,
        log_likelihood=np.sum(scs.triang.logpdf(data, c, loc, scale))
    )

    # report results in the form of a dictionary
    return {"c": c, "loc": loc, "scale": scale, "AIC": aic}


# Uniform
def fit_uniform(data, x_label, figure_size=5, bin_width=None):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param figure_size: int, specify the figure size
    :param bin_width: bin width
    :returns: dictionary with keys "loc", "scale", and "AIC"
    """
    # This distribution is constant between loc and loc + scale.

    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(figure_size+1, figure_size))
    ax.hist(data, normed=1, bins=find_bins(data, bin_width), edgecolor='black', alpha=0.5, label='Frequency')

    # estimate the parameters
    loc, scale = scs.uniform.fit(data)

    # plot the estimated distribution
    x_values = np.linspace(scs.uniform.ppf(MIN_PROP, loc, scale), scs.uniform.ppf(MAX_PROB, loc, scale), 200)
    rv = scs.uniform(loc, scale)
    ax.plot(x_values, rv.pdf(x_values), color=COLOR_CONTINUOUS_FIT, lw=2, label='Uniform')

    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0%}'))
    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()

    # calculate AIC
    aic = AIC(
        k=2,
        log_likelihood=np.sum(scs.uniform.logpdf(data, loc, scale))
    )

    # report results in the form of a dictionary
    return {"loc": loc, "scale": scale, "AIC": aic}


# UniformDiscrete
def fit_uniformDiscrete(data, x_label, figure_size=5, bin_width=None):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param figure_size: int, specify the figure size
    :param bin_width: bin width
    :returns: dictionary with keys "low", "high", and "AIC"
    """
    # This distribution is constant between low and high.

    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(figure_size+1, figure_size))
    ax.hist(data, normed=1, bins=find_bins(data, bin_width), edgecolor='black', alpha=0.5, label='Frequency')

    # estimate the parameters
    # as likelihood = 1/(high-low)^n, so the smaller the range, the higher the likelihood
    # the MLE is
    low = np.min(data)
    high = np.max(data)

    # plot the estimated distribution
    x_values = np.arange(low, high, step=1)
    rv = scs.randint(low, high)
    ax.step(x_values, rv.pmf(x_values), color=COLOR_CONTINUOUS_FIT, lw=2, label='UniformDiscrete')

    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0%}'))
    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()

    # calculate AIC
    aic = AIC(
        k=2,
        log_likelihood=np.sum(scs.randint.logpmf(data, low, high))
    )

    # report results in the form of a dictionary
    return {"l": low, "r": high, "AIC": aic}


# Weibull
def fit_weibull(data, x_label, fixed_location=0, figure_size=5, bin_width=None):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param fixed_location: fixed location
    :param figure_size: int, specify the figure size
    :param bin_width: bin width
    :returns: dictionary with keys "c", "loc", "scale", and "AIC"
    """

    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(figure_size+1, figure_size))
    ax.hist(data, normed=1, bins=find_bins(data, bin_width), edgecolor='black', alpha=0.5, label='Frequency')

    # estimate the parameters of weibull
    # location is fixed at 0
    c, loc, scale = scs.weibull_min.fit(data, floc=fixed_location)

    # plot the fitted Weibull distribution
    x_values = np.linspace(scs.weibull_min.ppf(MIN_PROP, c, loc, scale), scs.weibull_min.ppf(MAX_PROB, c, loc, scale), 100)
    rv = scs.weibull_min(c, loc, scale)
    ax.plot(x_values, rv.pdf(x_values), color=COLOR_CONTINUOUS_FIT, lw=2, label='Weibull')

    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0%}'))
    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()

    # calculate AIC
    aic = AIC(
        k=2,
        log_likelihood=np.sum(scs.weibull_min.logpdf(data, c, loc, scale))
    )

    # report results in the form of a dictionary
    return {"c": c, "loc": loc, "scale": scale, "AIC": aic}

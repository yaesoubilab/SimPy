import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy

warnings.filterwarnings("ignore")


# Exponential
def get_expon_params(mean, fixed_location=0):
    """
    :param mean: sample mean
    :param fixed_location: minimum of the exponential distribution, set to 0 by default
    :return: dictionary with keys "loc" and "scale"
    """
    mean = mean - fixed_location
    scale = mean

    return {"loc": fixed_location, "scale": scale}


# BetaBinomial
# 3 parameters in total, won't be able to estimate if we only know mean and st_dev
def get_beta_binomial_params(mean, st_dev, n, fixed_location=0, fixed_scale=1):
    """
    # ref: https://en.wikipedia.org/wiki/Beta-binomial_distribution
    :param mean: sample mean of an observation set
    :param st_dev: sample standard deviation of an observation set
    :param n: the number of trials in the Binomial distribution
    :param fixed_location: location, 0 by default
    :param fixed_scale: scale, 1 by default
    :return: dictionary with keys "a", "b", "n", "loc", and "scale"
    """
    mean = 1.0*(mean - fixed_location)/fixed_scale
    variance = (st_dev/fixed_scale)**2.0
    m2 = variance + mean**2  # second moment

    a1 = n*mean - m2
    a2 = n*(m2/mean - mean - 1) + mean
    a = a1/a2
    b1 = (n-mean)*(n-m2/mean)
    b2 = a2
    b = b1/b2

    return {"a": a, "b": b, "n": n, "loc": fixed_location, "scale": fixed_scale}


# Binomial
def get_binomial_params(mean, st_dev, fixed_location=0):
    """
    :param mean: sample mean
    :param st_dev: sample standard deviation
    :param fixed_location: fixed location, 0 by default
    :return: dictionary with keys "p", "n" and "loc"
    """
    mean = mean-fixed_location
    p = 1.0 - (st_dev**2)/mean
    n = mean/p

    return {"p": p, "n": n, "loc": fixed_location}


# Empirical
def get_empirical_params(data, bin_size=1):
    """
    :param data: (numpy.array) observations
    :param bin_size: float, the width of histogram's bins
    :returns: dictionary keys of "bins" and "freq"
    """
    result = plt.hist(data, bins=np.arange(np.min(data), np.max(data) + bin_size, bin_size))

    bins = result[1] # bins are in the form of [a,b)
    freq = result[0]*1.0/len(data)

    return {"bins": bins, "freq": freq}


# Gamma
def get_gamma_params(mean, st_dev, fixed_location=0):
    """
    :param mean: sample mean of an observation set
    :param st_dev: sample standard deviation of an observation set
    :param fixed_location: location, 0 by default
    :return: dictionary with keys "a", "loc" and "scale"
    """
    mean = mean - fixed_location

    shape = (mean / st_dev) ** 2
    scale = (st_dev ** 2) / mean

    # report results in the form of a dictionary
    return {"a": shape, "loc": fixed_location, "scale": scale}


# GammaPoisson
# ref: http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Gammapoisson.pdf
# scale = 1/beta
def get_gamma_poisson_params(mean, st_dev, fixed_location=0, fixed_scale=1):
    """
    :param mean: sample mean
    :param st_dev: sample standard deviation
    :param n: the number of trials in the Binomial distribution
    :param fixed_location: location, 0 by default
    :param fixed_scale: scale, 1 by default
    :return: dictionary with keys "a", "gamma_scale", "loc" and "scale"
    """
    mean = 1.0*(mean - fixed_location)/fixed_scale
    variance = (st_dev/fixed_scale)**2.0

    gamma_scale = mean**2.0/(variance - mean)
    a = (variance-mean)*1.0/mean

    return {"a": a, "gamma_scale": gamma_scale, "loc": fixed_location, "scale": fixed_scale}


# Geometric
def get_geometric_params(mean, fixed_location=0):
    """
    :param mean: sample mean
    :param fixed_location: location, 0 by default
    :return: dictionary with keys "p", "loc"
    """
    mean = mean - fixed_location
    p = 1.0/mean

    return {"p": p, "loc": fixed_location}


# LogNormal
def get_lognormal_params(mean, st_dev, fixed_location=0):
    """
    :param mean: sample mean of an observation set
    :param st_dev: sample standard deviation of an observation set
    :param fixed_location: location, 0 by default
    :return: dictionary with keys "n", "p" and "loc"
    """

    mean = mean-fixed_location

    mu = np.log(
        mean**2 / np.sqrt(st_dev**2 + mean**2)
    )
    sigma = np.sqrt(
        np.log(1 + st_dev**2 / mean**2)
    )

    return {"s": sigma, "loc": fixed_location, "scale": np.exp(mu)}


# NegativeBinomial
def get_negative_binomial_params(mean, st_dev, fixed_location=0):
    """
    :param mean: sample mean of an observation set
    :param st_dev: sample standard deviation of an observation set
    :param fixed_location: location, 0 by default
    :returns: dictionary with keys "n", "p" and "loc"
    """
    # in Scipy, n is the number of successes, p is the probability of a single success.
    # in Wiki, r is the number of failure, p is success probability
    # to match the moments, define r = n is the number of successes, 1-p is the failure probability
    mean = mean - fixed_location

    p = mean/st_dev**2.0
    n = mean*p/(1-p)

    return {"n": n, "p": p, "loc": fixed_location}


# Normal
def get_normal_params(mean, st_dev):
    """
    :param mean: sample mean of an observation set
    :param st_dev: sample standard deviation of an observation set
    :return: dictionary with keys "loc" and "scale"
    """

    return {"loc": mean, "scale": st_dev}


# Poisson
def get_poisson_params(mean, fixed_location=0):
    """
    :param mean: sample mean of an observation set
    :param fixed_location: location, 0 by default
    :returns: dictionary with keys "mu" and "loc"
    """

    mu = mean - fixed_location

    return {"mu": mu, "loc": fixed_location}


# Uniform
def get_uniform_params(mean, st_dev):
    """
    :param mean: sample mean of an observation set
    :param st_dev: sample standard deviation of an observation set
    :return: dictionary with keys "loc" and "scale"
    """

    b = 0.5*(2*mean + np.sqrt(12)*st_dev)
    a = 2.0*mean - b

    loc = a
    scale = b-a

    return {"loc": loc, "scale": scale}


# UniformDiscrete
# ref: https://en.wikipedia.org/wiki/Discrete_uniform_distribution
def get_uniform_discrete_params(mean, st_dev):
    """
    :param mean: sample mean of an observation set
    :param st_dev: sample standard deviation of an observation set
    :return: dictionary with keys "l" and "r"
    """
    variance = st_dev**2
    b = (np.sqrt(12.0*variance + 1) + 2.0*mean-1)*0.5
    a = (-np.sqrt(12.0*variance + 1) + 2.0*mean+1)*0.5

    return {"l": a, "r": b}


# Weibull
# ref: https://stats.stackexchange.com/questions/159452/how-can-i-recreate-a-weibull-distribution-given-mean-and-standard-deviation-and
def get_weibull_params(mean, st_dev, fixed_location=0):
    """
    :param mean: sample mean of an observation set
    :param st_dev: sample standard deviation of an observation set
    :param fixed_location: location, 0 by default
    :returns: dictionary with keys "c", "loc" and "scale"
    """
    mean = mean - fixed_location

    c = (st_dev*1.0/mean)**(-1.086)
    scale = mean/scipy.special.gamma(1 + 1.0/c)

    return {"c": c, "loc": fixed_location, "scale": scale}




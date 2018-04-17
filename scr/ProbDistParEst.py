import numpy as np

import warnings
warnings.filterwarnings("ignore")


# 1 Exponential
def get_expon_params(mean, fixed_location=0):
    """
    :param mean: sample mean of an observation set
    :param fixed_location: specify location, 0 by default
    :return: scale and location
    """
    mean = mean - fixed_location
    scale = mean

    return {"loc": fixed_location, "scale": scale}


# 2 Beta
def get_beta_params(mean, st_dev, min=0, max=1):
    """
    :param mean: sample mean of an observation set
    :param st_dev: sample standard deviation of an observation set
    :return: alpha and beta of the gamma distribution with mean and standard deviation matching the
    provided sample mean and sample st_dev
    """
    # shift the distribution by loc and scale
    mean = mean-min
    st_dev = st_dev/(max-min)*1.0

    aPlusB = mean*(1-mean)/st_dev**2 - 1
    a = mean*aPlusB

    return {"a": a, "b": aPlusB - a, "loc": min, "scale": max-min}


# 3 BetaBinomial
# ref: https://en.wikipedia.org/wiki/Beta-binomial_distribution
def get_beta_binomial_paras(mean, st_dev, n, fixed_location=0, fixed_scale=1):
    """
    :param mean: sample mean of an observation set
    :param st_dev: sample standard deviation of an observation set
    :param n: the number of trials in the Binomial distribution
    :param fixed_location: specify location, 0 by default
    :param fixed_scale: specify scale, 1 by default
    :return: dictionary with keys "a", "b", "n" and fixed parameters
    """
    mean = 1.0*(mean - fixed_location)/fixed_scale
    variance = (st_dev/fixed_scale)**2.0
    m2 = variance + mean**2 # second moment

    a1 = n*mean - m2
    a2 = n*(m2/mean - mean - 1) + mean
    a = a1/a2
    b1 = (n-mean)*(n-m2/mean)
    b2 = a2
    b = b1/b2

    return {"a": a, "b": b, "n": n, "loc": fixed_location, "scale": fixed_scale}


# 4 Binomial
def get_binomial_parameters(mean, st_dev, fixed_location=0):
    """
    :param mean: sample mean of an observation set
    :param st_dev: sample standard deviation of an observation set
    :param fixed_location: specify location, 0 by default
    :return: success probability p and location
    """
    mean = mean-fixed_location
    p = 1.0 - (st_dev**2)/mean

    return {"p": p, "loc": fixed_location}


# 5 Empirical (I guess for this, we just need to return the frequency of each observation)
def get_empirical_parameters(data, x_label):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :returns: frequency of unique observations
    """
    unique, counts = np.unique(data, return_counts=True)
    freq = counts*1.0/len(data)

    return unique,freq


# 6 Gamma
def get_gamma_parameters(mean, st_dev, fixed_location=0):
    """
    :param mean: sample mean of an observation set
    :param st_dev: sample standard deviation of an observation set
    :param fixed_location: specify location, 0 by default
    :return: shape and scale of the gamma distribution with mean and standard deviation matching the
    provided sample mean and sample st_dev
    """
    mean = mean - fixed_location

    shape = (mean / st_dev) ** 2
    scale = (st_dev ** 2) / mean

    # report results in the form of a dictionary
    return {"a": shape, "loc": fixed_location, "scale": scale}


# 7 GammaPoisson
# ref: http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Gammapoisson.pdf
# scale = 1/beta
def get_gamma_poisson_paras(mean, st_dev, fixed_location=0, fixed_scale=1):
    """
    :param mean: sample mean of an observation set
    :param st_dev: sample standard deviation of an observation set
    :param n: the number of trials in the Binomial distribution
    :param fixed_location: specify location, 0 by default
    :param fixed_scale: specify scale, 1 by default
    :return: dictionary with keys "a", "gamma_scale" and fixed parameters
    """
    mean = 1.0*(mean - fixed_location)/fixed_scale
    variance = (st_dev/fixed_scale)**2.0

    gamma_scale = mean**2.0/(variance - mean)
    a = (variance-mean)*1.0/mean

    return {"a": a, "gamma_scale": gamma_scale, "loc": fixed_location, "scale": fixed_scale}


# 8 Geometric
def get_geomertic_paras(mean, fixed_location=0):
    """
    :param mean: sample mean of an observation set
    :param fixed_location:
    :return: probability p and location
    """
    mean = mean - fixed_location
    p = 1.0/mean

    return {"p": p, "loc": fixed_location}


# 9 JohnsonSb
# need percentiles of data
# ref: https://digitalcommons.wayne.edu/cgi/viewcontent.cgi?article=1326&context=jmasm


# 10 JohnsonSu
# need percentiles of data
# ref: https://digitalcommons.wayne.edu/cgi/viewcontent.cgi?article=1326&context=jmasm


# 11 LogNormal
def get_log_normal_parameters(mean, st_dev, fixed_location=0):
    """
    :param mean: sample mean of an observation set
    :param st_dev: sample standard deviation of an observation set
    :return: mu and sigma the lognormal distribution with mean and standard deviation matching the
    provided sample mean and sample st_dev
    """

    mean = mean-fixed_location

    mu = np.log(
        mean**2 / np.sqrt(st_dev**2 + mean**2)
    )
    sigma = np.sqrt(
        np.log(1 + st_dev**2 / mean**2)
    )

    return {"s": sigma, "loc": fixed_location, "scale": np.exp(mu)}


# 12 NegativeBinomial
def get_negative_binomial_paras(mean, st_dev, fixed_location=0):
    """
    :param mean: sample mean of an observation set
    :param st_dev: sample standard deviation of an observation set
    :returns: dictionary with keys "n", "p"
    """
    # in Scipy, n is the number of successes, p is the probability of a single success.
    # in Wiki, r is the number of failure, p is success probability
    # to match the moments, define r = n is the number of successes, 1-p is the failure probability
    mean = mean - fixed_location

    p = mean/st_dev**2.0
    n = mean*p/(1-p)

    return {"n": n, "p": p, "loc": fixed_location}


# 13 Normal
def get_normal_param(mean, st_dev):
    """
    :param mean: sample mean of an observation set
    :param st_dev: sample standard deviation of an observation set
    :param fixed_location: specify location, 0 by default
    :return: location and scale of the normal distribution
    """

    return {"loc": mean, "scale": st_dev}


# 14 Triangular


# 15 Uniform


# 16 UniformDiscrete


# 17 Weibull


# 18 Poisson


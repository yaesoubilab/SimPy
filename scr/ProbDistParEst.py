import numpy as np
import scipy.stats as scs
import scipy as sp

import warnings
warnings.filterwarnings("ignore")

COLOR_CONTINUOUS_FIT = 'r'
COLOR_DISCRETE_FIT = 'r'



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
# too many parameters ??

# 10 JohnsonSu


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


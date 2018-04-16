import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
import scipy as sp
from scipy.optimize import fmin_slsqp

import warnings
warnings.filterwarnings("ignore")

COLOR_CONTINUOUS_FIT = 'r'
COLOR_DISCRETE_FIT = 'r'









# 1 Exponential


# 2 Beta
def get_beta_parameters(mean, st_dev, min=0, max=1):
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


# 5 Empirical (I guess for this, we just need to return the frequency of each observation)
def fit_empirical(data, x_label):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :returns: frequency of unique observations
    """
    unique, counts = np.unique(data, return_counts=True)
    freq = counts*1.0/len(data)

    # plot histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(data, normed=1, bins=np.max(data)+1, range=[-0.5, np.max(data)+0.5],
            edgecolor='black', alpha=0.5, label='Frequency')

    # plot with fitted parameter
    x_plot = unique
    ax.step(unique, freq, COLOR_DISCRETE_FIT, ms=8, label='Empirical')

    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()

    return unique,freq


# 6 Gamma
def get_gamma_parameters(mean, st_dev, fixed_location=0):
    """
    :param mean: sample mean of an observation set
    :param st_dev: sample standard deviation of an observation set
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


# 9 JohnsonSb


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


# 14 Triangular


# 15 Uniform


# 16 UniformDiscrete


# 17 Weibull


# 18 Poisson


import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats as scs
from scipy.optimize import fmin_slsqp
import SimPy.RandomVariateGenerators as RVGs

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
    ax.hist(data, density=True, bins=find_bins(data, bin_width), # bins=np.max(data)+1, range=[-0.5, np.max(data)+0.5],
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



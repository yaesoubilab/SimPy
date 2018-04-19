import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
import scipy as sp
from scipy.optimize import fmin_slsqp

import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------------------------------
# This module contains procedures to fit probability distributions to the data using maximum likelihood approaches
# Functions to fit probability distributions of non-negative random variables (e.g. exponential and gamma) take an
# fixed_location as an argument (with a default value set to 0). Specifying this argument allows for
# fitting a shifted distribution to the data.
# -------------------------------------------------------------------------------------------------

COLOR_CONTINUOUS_FIT = 'r'
COLOR_DISCRETE_FIT = 'r'


def AIC(k, log_likelihood):
    """ :returns Akaike information criterion"""
    return 2 * k - 2 * log_likelihood


# 1 Exponential
def fit_exp(data, x_label, fixed_location=0, figure_size=5):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param fixed_location: specify location, 0 by default
    :param figure_size: int, specify the figure size
    :returns: dictionary with keys "loc", "scale", and "AIC"
    """

    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(figure_size+1, figure_size))
    ax.hist(data, normed=1, bins='auto', edgecolor='black', alpha=0.5, label='Frequency')

    # estimate the parameters of exponential
    loc, scale = scs.expon.fit(data, floc=fixed_location)

    # plot the estimated exponential distribution
    x_values = np.linspace(scs.expon.ppf(0.0001, loc, scale), scs.expon.ppf(0.9999, loc, scale), 200)
    rv = scs.expon(loc, scale)
    ax.plot(x_values, rv.pdf(x_values), color=COLOR_CONTINUOUS_FIT, lw=2, label='Exponential')

    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()

    # calculate AIC
    aic = AIC(
        k=1,
        log_likelihood=np.sum(scs.expon.logpdf(data, loc, scale))
    )

    # report results in the form of a dictionary
    return {"loc": loc, "scale": scale, "AIC": aic}


# 2 Beta
def fit_beta(data, x_label, minimum=None, maximum=None, figure_size=5):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param minimum: minimum of data (calculated from data if not provided)
    :param maximum: maximum of data (calculated from data if not provided)
    :param figure_size: int, specify the figure size
    :returns: dictionary with keys "a", "b", "loc", "scale", and "AIC"
    """
    # transform data into [0,1]
    if minimum==None:
        L = np.min(data)
    else:
        L = minimum

    if maximum==None:
        U = np.max(data)
    else:
        U = maximum

    data = (data-L)/(U-L)

    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(figure_size+1, figure_size))
    ax.hist(data, normed=1, bins='auto', edgecolor='black', alpha=0.5, label='Frequency')

    # estimate the parameters
    a, b, loc, scale = scs.beta.fit(data, floc=0)

    # plot the estimated distribution
    x_values = np.linspace(scs.beta.ppf(0.0001, a, b, loc, scale),
                           scs.beta.ppf(0.9999, a, b, loc, scale), 200)
    rv = scs.beta(a, b, loc, scale)
    ax.plot(x_values, rv.pdf(x_values), color=COLOR_CONTINUOUS_FIT, lw=2, label='Beta')

    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()

    # calculate AIC
    aic = AIC(
        k=3,
        log_likelihood=np.sum(scs.beta.logpdf(data, a, b, loc, scale))
    )

    # report results in the form of a dictionary
    return {"a": a, "b": b, "loc": L, "scale": U-L, "AIC": aic}


# 3 BetaBinomial
def fit_beta_binomial(data, x_label, fixed_location=0, fixed_scale=1, figure_size=5):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param fixed_location: fixed location
    :param fixed_scale: fixed scale
    :param figure_size: int, specify the figure size
    :returns: dictionary with keys "a", "b", "n" and "AIC"
    """

    data = 1.0*(data - fixed_location)/fixed_scale

    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(figure_size+1, figure_size))
    ax.hist(data, normed=1, bins='auto', edgecolor='black', alpha=0.5, label='Frequency')

    # define log_likelihood
    # ref: http://www.channelgrubb.com/blog/2015/2/27/beta-binomial-in-python
    def BetaBinom(a, b, n, k): # log(pmf) of beta binomial
        part_1 = sp.misc.comb(n, k)
        part_2 = sp.special.betaln(k + a, n - k + b)
        part_3 = sp.special.betaln(a, b)
        result = (np.log(part_1) + part_2) - part_3
        return result

    def loglik(theta):
        a, b, n = theta[0], theta[1], theta[2]
        n = int(np.round(n, 0))
        result = 0
        for i in range(len(data)):
            result += BetaBinom(a, b, n, data[i])
        return result

    def neg_loglik(theta):
        return -loglik(theta)

    # estimate the parameters by minimize -loglik
    theta0 = [1, 1, np.max(data)]
    paras, value, iter, imode, smode = fmin_slsqp(neg_loglik, theta0,
                            bounds=[(0.0, 10.0), (0.0, 10.0), (0, np.max(data)+100)],
                            disp=False, full_output=True)

    # plot the estimated distribution
    # get PMF
    x_values = np.arange(0, paras[2], step=1)
    pmf = np.zeros(len(x_values))
    for i in x_values:
        pmf[int(i)] = np.exp(BetaBinom(paras[0], paras[1], paras[2], i))
    # plot
    ax.step(x_values, pmf, color=COLOR_CONTINUOUS_FIT, lw=2, label='BetaBinomial')

    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()

    # calculate AIC
    aic = AIC(
        k=3,
        log_likelihood=loglik([paras[0], paras[1], paras[2]])
    )

    # report results in the form of a dictionary
    return {"a": paras[0], "b": paras[1], "n": paras[2], "loc": fixed_location, "scale": fixed_scale, "AIC": aic}


# 4 Binomial
def fit_binomial(data, x_label, fixed_location=0, figure_size=5):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param fixed_location: fixed location
    :param figure_size: int, specify the figure size
    :returns: dictionary with keys "p", "n" and "AIC"
    """

    # fit Binomial distribution: the MLE of p is x/n
    # if we have N data point with Xi~Bin(n,p), then sum(Xi)~Bin(n*N,p), p_hat = sum(xi)/(n*N)
    # # https://onlinecourses.science.psu.edu/stat504/node/28

    data = data - fixed_location

    mean = np.mean(data)
    st_dev = np.std(data)
    p = 1.0 - (st_dev**2)/mean
    n = mean/p

    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(figure_size+1, figure_size))
    ax.hist(data, normed=1, bins=np.max(data)+1, range=[-0.5, np.max(data)+0.5],
            edgecolor='black', alpha=0.5, label='Frequency')

    # plot with fitted parameter
    x_plot = np.arange(0, n, step=1)
    ax.step(x_plot, scs.binom.pmf(x_plot, n, p), COLOR_DISCRETE_FIT, ms=8, label='Binomial')

    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()

    # calculate AIC
    aic = AIC(
        k=1,
        log_likelihood=np.sum(scs.binom.logpmf(data, n, p))
    )

    # report results in the form of a dictionary
    return {"p": p, "n": n, "loc": fixed_location, "AIC": aic}


# 5 Empirical (I guess for this, we just need to return the frequency of each observation)
def fit_empirical(data, x_label, figure_size=5, bin_size=1):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param figure_size: int, specify the figure size
    :param bin_size: float, the width of histogram's bins
    :returns: dictionary keys of "bins" and "freq"
    """
    result = plt.hist(data, bins=np.arange(np.min(data), np.max(data) + bin_size, bin_size))

    bins = result[1] # bins are in the form of [a,b)
    freq = result[0]*1.0/len(data)

    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(figure_size+1, figure_size))
    ax.hist(data, normed=1, bins=np.arange(np.min(data), np.max(data) + bin_size, bin_size),
            edgecolor='black', alpha=0.5, label='Frequency')

    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()

    return {"bins": bins, "freq": freq}

# 6 Gamma
def fit_gamma(data, x_label, fixed_location=0, figure_size=5):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param fixed_location: fixed location
    :param figure_size: int, specify the figure size
    :returns: dictionary with keys "a", "loc", "scale", and "AIC"
    """

    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(figure_size+1, figure_size))
    ax.hist(data, normed=1, bins='auto', edgecolor='black', alpha=0.5, label='Frequency')

    # estimate the parameters of gamma
    # alpha = a, beta = 1/scale
    a, loc, scale = scs.gamma.fit(data,floc=fixed_location)

    # plot the estimated gamma distribution
    x_values = np.linspace(scs.gamma.ppf(0.0001, a, loc, scale), scs.gamma.ppf(0.9999, a, loc, scale), 200)
    rv = scs.gamma(a, loc, scale)
    ax.plot(x_values, rv.pdf(x_values), color=COLOR_CONTINUOUS_FIT, lw=2, label='Gamma')

    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()

    # calculate AIC
    aic = AIC(
        k=2,
        log_likelihood=np.sum(scs.gamma.logpdf(data, a, loc, scale))
    )

    # report results in the form of a dictionary
    return {"a": a, "loc": loc, "scale": scale, "AIC": aic}


# 7 GammaPoisson
def fit_gamma_poisson(data, x_label, fixed_location=0, fixed_scale=1, figure_size=5):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param figure_size: int, specify the figure size
    :returns: dictionary with keys "a", "scale" and "AIC"
    """
    data = 1 * (data - fixed_location) / fixed_scale

    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(figure_size+1, figure_size))
    ax.hist(data, normed=1, bins='auto', edgecolor='black', alpha=0.5, label='Frequency')

    # define log_likelihood
    # ref: https://en.wikipedia.org/wiki/Negative_binomial_distribution#Gamma%E2%80%93Poisson_mixture
    n=len(data)
    def gamma_poisson(r,p,k):
        part1 = 1.0*sp.special.gamma(r+k)/(sp.special.gamma(r) * sp.misc.factorial(k))
        part2 = (p**k)*((1-p)**r)
        return part1*part2

    def log_lik(theta):
        r, p = theta[0], theta[1]
        result = 0
        for i in range(n):
            result += np.log(gamma_poisson(r,p,data[i]))
        return result

    def neg_loglik(theta):
        return -log_lik(theta)

    # estimate the parameters by minimize -loglik
    # alpha=a, beta=1/scale
    theta0 = [2, 0.5]
    paras, value, iter, imode, smode = fmin_slsqp(neg_loglik, theta0, bounds=[(0.0, 10.0), (0,1)],
                              disp=False, full_output=True)

    a = paras[0]
    scale = paras[1]/(1.0-paras[1])

    # plot the estimated distribution
    # get PMF
    x_values = np.arange(0, np.max(data), step=1)
    pmf = np.zeros(len(x_values))
    for i in x_values:
        pmf[int(i)] = gamma_poisson(paras[0], paras[1], i)
    ax.step(x_values, pmf, color=COLOR_CONTINUOUS_FIT, lw=2, label='GammaPoisson')

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
    return {"a": a, "gamma_scale": scale, "AIC": aic, "loc": fixed_location, "scale": fixed_scale}


# 8 Geometric
def fit_geometric(data, x_label, fixed_location=0, figure_size=5):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param fixed_location: fixed location
    :param figure_size: int, specify the figure size
    :returns: dictionary with keys "p" and "AIC"
    """

    # https://www.projectrhea.org/rhea/index.php/MLE_Examples:_Exponential_and_Geometric_Distributions_Old_Kiwi
    data = data-fixed_location
    p = len(data)*1.0/np.sum(data)

    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(figure_size+1, figure_size))
    ax.hist(data, normed=1, bins=np.max(data)+1, range=[0.5, np.max(data)+0.5],
            edgecolor='black', alpha=0.5, label='Frequency')

    # plot poisson-deviation with fitted parameter
    x_plot = np.arange(scs.geom.ppf(0.0001, p), scs.geom.ppf(0.9999, p))
    ax.step(x_plot, scs.geom.pmf(x_plot, p), COLOR_DISCRETE_FIT, ms=8, label='Geometric')

    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()

    # calculate AIC
    aic = AIC(
        k=1,
        log_likelihood=np.sum(scs.geom.logpmf(data, p))
    )

    # report results in the form of a dictionary
    return {"p": p, "loc": fixed_location, "AIC": aic}


# 9 JohnsonSb
def fit_johnsonSb(data, x_label, fixed_location=0, figure_size=5):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param figure_size: int, specify the figure size
    :returns: dictionary with keys "a", "b", "loc", "scale", and "AIC"
    """

    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(figure_size+1, figure_size))
    ax.hist(data, normed=1, bins='auto', edgecolor='black', alpha=0.5, label='Frequency')

    # estimate the parameters
    a, b, loc, scale = scs.johnsonsb.fit(data, floc=fixed_location)

    # plot the estimated JohnsonSb distribution
    x_values = np.linspace(scs.johnsonsb.ppf(0.01, a, b, loc, scale),
                           scs.johnsonsb.ppf(0.99, a, b, loc, scale), 100)
    rv = scs.johnsonsb(a, b, loc, scale)
    ax.plot(x_values, rv.pdf(x_values), color=COLOR_CONTINUOUS_FIT, lw=2, label='JohnsonSb')

    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()

    # calculate AIC
    aic = AIC(
        k=3,
        log_likelihood=np.sum(scs.johnsonsb.logpdf(data, a, b, loc, scale))
    )

    # report results in the form of a dictionary
    return {"a": a, "b": b, "loc": loc, "scale": scale, "AIC": aic}


# 10 JohnsonSu
def fit_johnsonSu(data, x_label, fixed_location=0, figure_size=5):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param figure_size: int, specify the figure size
    :returns: dictionary with keys "a", "b", "loc", "scale", and "AIC"
    """

    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(figure_size+1, figure_size))
    ax.hist(data, normed=1, bins='auto', edgecolor='black', alpha=0.5, label='Frequency')

    # estimate the parameters
    a, b, loc, scale = scs.johnsonsu.fit(data, floc=fixed_location)

    # plot the estimated JohnsonSu distribution
    x_values = np.linspace(scs.johnsonsu.ppf(0.01, a, b, loc, scale),
                           scs.johnsonsu.ppf(0.99, a, b, loc, scale), 100)
    rv = scs.johnsonsu(a, b, loc, scale)
    ax.plot(x_values, rv.pdf(x_values), color=COLOR_CONTINUOUS_FIT, lw=2, label='JohnsonSu')

    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()

    # calculate AIC
    aic = AIC(
        k=3,
        log_likelihood=np.sum(scs.johnsonsu.logpdf(data, a, b, loc, scale))
    )

    # report results in the form of a dictionary
    return {"a": a, "b": b, "loc": loc, "scale": scale, "AIC": aic}


# 11 LogNormal
def fit_lognorm(data, x_label, fixed_location=0, figure_size=5):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param figure_size: int, specify the figure size
    :returns: dictionary with keys "s", "loc", "scale", and "AIC", s = sigma and scale = exp(mu)
    """

    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(figure_size+1, figure_size))
    ax.hist(data, normed=1, bins='auto', edgecolor='black', alpha=0.5, label='Frequency')

    # estimate the parameters
    s, loc, scale = scs.lognorm.fit(data, floc=fixed_location)

    # plot the estimated distribution
    x_values = np.linspace(scs.lognorm.ppf(0.0001, s, loc, scale), scs.lognorm.ppf(0.9999, s, loc, scale), 200)
    rv = scs.lognorm(s, loc, scale)
    ax.plot(x_values, rv.pdf(x_values), color=COLOR_CONTINUOUS_FIT, lw=2, label='LogNormal')

    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()

    # calculate AIC
    aic = AIC(
        k=2,
        log_likelihood=np.sum(scs.lognorm.logpdf(data, s, loc, scale))
    )

    # report results in the form of a dictionary
    return {"s": s, "loc": loc, "scale": scale, "AIC": aic}


# 12 NegativeBinomial
def fit_negative_binomial(data, x_label, fixed_location=0, figure_size=5):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param fixed_location: fixed location
    :param figure_size: int, specify the figure size
    :returns: dictionary with keys "n", "p" and "AIC"
    """
    # n is the number of successes, p is the probability of a single success.

    data = data-fixed_location

    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(figure_size+1, figure_size))
    ax.hist(data, normed=1, bins='auto', edgecolor='black', alpha=0.5, label='Frequency')

    M=np.max(data)

    # define log_likelihood
    def log_lik(theta):
        n, p = theta[0], theta[1]
        result = 0
        for i in range(len(data)):
            result += scs.nbinom.logpmf(data[i], n, p)
        return result

    def neg_loglik(theta):
        return -log_lik(theta)

    # estimate the parameters by minimize -loglik
    theta0 = [2, 0.5]
    paras, value, iter, imode, smode = fmin_slsqp(neg_loglik, theta0, bounds=[(0.0, M), (0,1)],
                              disp=False, full_output=True)

    # plot the estimated distribution
    # get PMF
    x_values = np.arange(0, np.max(data), step=1)
    rv = scs.nbinom(paras[0],paras[1])
    ax.step(x_values, rv.pmf(x_values), color=COLOR_CONTINUOUS_FIT, lw=2, label='NegativeBinomial')

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


# 13 Normal
def fit_normal(data, x_label, figure_size=5):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param figure_size: int, specify the figure size
    :returns: dictionary with keys "loc", "scale", and "AIC"
    """

    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(figure_size+1, figure_size))
    ax.hist(data, normed=1, bins='auto', edgecolor='black', alpha=0.5, label='Frequency')

    # estimate the parameters
    loc, scale = scs.norm.fit(data)

    # plot the estimated distribution
    x_values = np.linspace(scs.norm.ppf(0.0001, loc, scale), scs.norm.ppf(0.9999, loc, scale), 200)
    rv = scs.norm(loc, scale)
    ax.plot(x_values, rv.pdf(x_values), color=COLOR_CONTINUOUS_FIT, lw=2, label='Normal')

    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()

    # calculate AIC
    aic = AIC(
        k=2,
        log_likelihood=np.sum(scs.norm.logpdf(data, loc, scale))
    )

    # report results in the form of a dictionary
    return {"loc": loc, "scale": scale, "AIC": aic}


# 18 Poisson
def fit_poisson(data, x_label, fixed_location=0, figure_size=5):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param figure_size: int, specify the figure size
    :returns: dictionary with keys "lambda" and "AIC"
    """

    # fit poisson distribution: the MLE of lambda is the sample mean
    # https://en.wikipedia.org/wiki/Poisson_distribution#Maximum_likelihood
    data = data-fixed_location
    mu = data.mean()

    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(figure_size+1, figure_size))
    ax.hist(data, normed=1, bins=np.max(data)+1, range=[-0.5, np.max(data)+0.5],
            edgecolor='black', alpha=0.5, label='Frequency')

    # plot poisson-deviation with fitted parameter
    x_plot = np.arange(scs.poisson.ppf(0.0001, mu), scs.poisson.ppf(0.9999, mu))
    ax.step(x_plot, scs.poisson.pmf(x_plot, mu), COLOR_DISCRETE_FIT, ms=8, label='Poisson')

    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()

    # calculate AIC
    aic = AIC(
        k=1,
        log_likelihood=np.sum(scs.poisson.logpmf(data, mu))
    )

    # report results in the form of a dictionary
    return {"mu": mu, "AIC": aic, "loc": fixed_location}


# 14 Triangular
def fit_triang(data, x_label, fixed_location=0, figure_size=5):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param fixed_location: fixed location
    :param figure_size: int, specify the figure size
    :returns: dictionary with keys "c", "loc", "scale", and "AIC"
    """
    # The triangular distribution can be represented with an up-sloping line from
    # loc to (loc + c*scale) and then downsloping for (loc + c*scale) to (loc+scale).

    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(figure_size+1, figure_size))
    ax.hist(data, normed=1, bins='auto', edgecolor='black', alpha=0.5, label='Frequency')

    # estimate the parameters
    c, loc, scale = scs.triang.fit(data, floc=fixed_location)

    # plot the estimated distribution
    x_values = np.linspace(scs.triang.ppf(0.0001, c, loc, scale), scs.triang.ppf(0.9999, c, loc, scale), 200)
    rv = scs.triang(c, loc, scale)
    ax.plot(x_values, rv.pdf(x_values), color=COLOR_CONTINUOUS_FIT, lw=2, label='Triangular')

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


# 15 Uniform
def fit_uniform(data, x_label, figure_size=5):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param figure_size: int, specify the figure size
    :returns: dictionary with keys "loc", "scale", and "AIC"
    """
    # This distribution is constant between loc and loc + scale.

    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(figure_size+1, figure_size))
    ax.hist(data, normed=1, bins='auto', edgecolor='black', alpha=0.5, label='Frequency')

    # estimate the parameters
    loc, scale = scs.uniform.fit(data)

    # plot the estimated distribution
    x_values = np.linspace(scs.uniform.ppf(0.0001, loc, scale), scs.uniform.ppf(0.9999, loc, scale), 200)
    rv = scs.uniform(loc, scale)
    ax.plot(x_values, rv.pdf(x_values), color=COLOR_CONTINUOUS_FIT, lw=2, label='Uniform')

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


# 16 UniformDiscrete
def fit_uniformDiscrete(data, x_label, figure_size=5):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param figure_size: int, specify the figure size
    :returns: dictionary with keys "low", "high", and "AIC"
    """
    # This distribution is constant between low and high.

    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(figure_size+1, figure_size))
    ax.hist(data, normed=1, bins='auto', edgecolor='black', alpha=0.5, label='Frequency')

    # estimate the parameters
    # as likelihood = 1/(high-low)^n, so the smaller the range, the higher the likelihood
    # the MLE is
    low = np.min(data)
    high = np.max(data)

    # plot the estimated distribution
    x_values = np.arange(low, high, step=1)
    rv = scs.randint(low, high)
    ax.step(x_values, rv.pmf(x_values), color=COLOR_CONTINUOUS_FIT, lw=2, label='UniformDiscrete')

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


# 17 Weibull
def fit_weibull(data, x_label, fixed_location=0, figure_size=5):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param fixed_location: fixed location
    :param figure_size: int, specify the figure size
    :returns: dictionary with keys "c", "loc", "scale", and "AIC"
    """

    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(figure_size+1, figure_size))
    ax.hist(data, normed=1, bins='auto', edgecolor='black', alpha=0.5, label='Frequency')

    # estimate the parameters of weibull
    # location is fixed at 0
    c, loc, scale = scs.weibull_min.fit(data, floc=fixed_location)

    # plot the fitted Weibull distribution
    x_values = np.linspace(scs.weibull_min.ppf(0.001, c, loc, scale), scs.weibull_min.ppf(0.999, c, loc, scale), 100)
    rv = scs.weibull_min(c, loc, scale)
    ax.plot(x_values, rv.pdf(x_values), color=COLOR_CONTINUOUS_FIT, lw=2, label='Weibull')

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

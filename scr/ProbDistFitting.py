import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
import scipy as sp

COLOR_CONTINUOUS_FIT = 'r'
COLOR_DISCRETE_FIT = 'r'

# In most functions with Location parameter, floc=0 is applied
# (fix location at 0), since if not, estimated parameters are not unique
# for example Exponential distribution only has one parameter lambda, k=1

def AIC(k, log_likelihood):
    """ :returns Akaike information criterion"""
    return 2 * k - 2 * log_likelihood

# 1 Exponential
def fit_exp(data, x_label):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :returns: dictionary with keys "loc", "scale", and "AIC"
    """

    # plot histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(data, normed=1, bins='auto', edgecolor='black', alpha=0.5, label='Frequency')

    # estimate the parameters of exponential
    loc, scale = scs.expon.fit(data, floc=0)

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
        k=1, # lambda = 1/scale
        log_likelihood=np.sum(scs.expon.logpdf(data, loc, scale))
    )

    # report results in the form of a dictionary
    return {"loc": loc, "scale": scale, "AIC": aic}

# 2 Beta
def fit_beta(data, x_label, min=None, max=None):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param min: minimum of data, given or calculated from data
    :param max: maximum of data, given or calculated from data
    :returns: dictionary with keys "a", "b", "loc", "scale", and "AIC"
    """
    # transform data into [0,1]
    if min==None:
        L = np.min(data)
    else:
        L = min

    if max==None:
        U = np.max(data)
    else:
        U = max

    data = (data-L)/(U-L)

    # plot histogram
    fig, ax = plt.subplots(1, 1)
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
    return {"a": a, "b": b, "loc": loc, "scale": scale, "AIC": aic}

# 3 BetaBinomial
def fit_beta(data, x_label, min=None, max=None):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param min: minimum of data, given or calculated from data
    :param max: maximum of data, given or calculated from data
    :returns: dictionary with keys "a", "b", "loc", "scale", and "AIC"
    """

    # plot histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(data, normed=1, bins='auto', edgecolor='black', alpha=0.5, label='Frequency')

    # define log_likelihood
    # ref: http://www.channelgrubb.com/blog/2015/2/27/beta-binomial-in-python
    def BetaBinom(a, b, n, k):
        part_1 = sp.misc.comb(n, k)
        part_2 = sp.special.betaln(k + a, n - k + b)
        part_3 = sp.special.betaln(a, b)
        result = (np.log(part_1) + part_2) - part_3
        return result

    def loglik(x):
        a, b, n = x[0], x[1],x[2]
        result = 0
        for i in range(len(data)):
            result += BetaBinom(a, b, n, data[i])
        return result

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
    return {"a": a, "b": b, "loc": loc, "scale": scale, "AIC": aic}

# 4 Binomial
# 5 Empirical (I guess for this, we just need to return the frequency of each observation)


# 6 Gamma
def fit_gamma(data, x_label):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :returns: dictionary with keys "a", "loc", "scale", and "AIC"
    """

    # plot histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(data, normed=1, bins='auto', edgecolor='black', alpha=0.5, label='Frequency')

    # estimate the parameters of gamma
    a, loc, scale = scs.gamma.fit(data)

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
        k=3,
        log_likelihood=np.sum(scs.gamma.logpdf(data, a, loc, scale))
    )

    # report results in the form of a dictionary
    return {"a": a, "loc": loc, "scale": scale, "AIC": aic}


# 7 GammaPoisson
# 8 Geometric

# 9 JohnsonSb
def fit_johnsonSb(data, x_label):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :returns: dictionary with keys "a", "b", "loc", "scale", and "AIC"
    """

    # plot histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(data, normed=1, bins='auto', edgecolor='black', alpha=0.5, label='Frequency')

    # estimate the parameters
    a, b, loc, scale = scs.johnsonsb.fit(data, floc=0)

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
        k=3, # loc is fixed at 0, so 3 parameters
        log_likelihood=np.sum(scs.johnsonsb.logpdf(data, a, b, loc, scale))
    )

    # report results in the form of a dictionary
    return {"a": a, "b": b, "loc": loc, "scale": scale, "AIC": aic}

# 10 JohnsonSu
def fit_johnsonSu(data, x_label):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :returns: dictionary with keys "a", "b", "loc", "scale", and "AIC"
    """

    # plot histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(data, normed=1, bins='auto', edgecolor='black', alpha=0.5, label='Frequency')

    # estimate the parameters
    a, b, loc, scale = scs.johnsonsu.fit(data, floc=0)

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
        k=3, # loc is fixed at 0, so 3 parameters
        log_likelihood=np.sum(scs.johnsonsu.logpdf(data, a, b, loc, scale))
    )

    # report results in the form of a dictionary
    return {"a": a, "b": b, "loc": loc, "scale": scale, "AIC": aic}



# 11 LogNormal
# 12 NegativeBinomial
# 13 Normal
# 14 Triangular
# 15 Uniform
# 16 UniformDiscrete

# 17 Weibull
def fit_weibull(data, x_label):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :returns: dictionary with keys "a", "loc", "scale", and "AIC"
    """

    # plot histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(data, normed=1, bins='auto', edgecolor='black', alpha=0.5, label='Frequency')

    # estimate the parameters of weibull
    # location is fixed at 0
    a, loc, scale = scs.weibull_min.fit(data, floc=0)

    # plot the estimated gamma distribution
    x_values = np.linspace(scs.weibull_min.ppf(0.01, a, loc, scale), scs.weibull_min.ppf(0.99, a, loc, scale), 100)
    rv = scs.weibull_min(a, loc, scale)
    ax.plot(x_values, rv.pdf(x_values), color=COLOR_CONTINUOUS_FIT, lw=2, label='Weibull')

    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()

    # calculate AIC
    aic = AIC(
        k=2, # as location is fixed, only 2 parameters estimated
        log_likelihood=np.sum(scs.weibull_min.logpdf(data, a, loc, scale))
    )

    # report results in the form of a dictionary
    return {"a": a, "loc": loc, "scale": scale, "AIC": aic}

# 18 Poisson
def fit_poisson(data, x_label):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :returns: dictionary with keys "lambda" and "AIC"
    """

    # fit poisson distribution: the MLE of lambda is the sample mean
    # https://en.wikipedia.org/wiki/Poisson_distribution#Maximum_likelihood
    lamb = data.mean()

    # plot histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(data, normed=1, bins=np.max(data)+1, range=[-0.5, np.max(data)+0.5],
            edgecolor='black', alpha=0.5, label='Frequency')

    # plot poisson-deviation with fitted parameter
    x_plot = np.arange(scs.poisson.ppf(0.0001, lamb), scs.poisson.ppf(0.9999, lamb))
    ax.step(x_plot, scs.poisson.pmf(x_plot, lamb), COLOR_DISCRETE_FIT, ms=8, label='Poisson')

    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()

    # calculate AIC
    aic = AIC(
        k=1,
        log_likelihood=np.sum(scs.poisson.logpmf(data, lamb))
    )

    # report results in the form of a dictionary
    return {"lambda": lamb, "AIC": aic}

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
import scipy as sp
import pyjags
from scipy.optimize import fmin_slsqp

COLOR_CONTINUOUS_FIT = 'r'
COLOR_DISCRETE_FIT = 'r'

# In most functions with Location parameter, floc=0 is applied
# (fix location at 0), since if not, estimated parameters are not unique
# for example Exponential distribution only has one parameter lambda, k=1

def AIC(k, log_likelihood):
    """ :returns Akaike information criterion"""
    return 2 * k - 2 * log_likelihood

# JAGS Monte-Carlo simulation for parameter estimation
# with uniform prior, equivalent to MLE
def MCMC_JAGS(num_samples, var_names, model, data):
    """
    :param num_samples: number of steps in sample path, real steps = num_samples*4
    :param var: list of parameter names
    :param model: the JAGS code for model defining
    :return: return MCMC sample path and Maximum Likelihood Estimation of parameters
    """
    mc = pyjags.Model(model, data=dict(y=data, N=len(data), n=100))
    samples = mc.sample(num_samples, vars=var_names)

    for varname in var_names:
        print(varname, np.mean(samples[varname]))

    return samples



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
def fit_betaBinomial(data, x_label, n=None):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param n: the number of trials in the Binomial distribution
    :returns: dictionary with keys "a", "b", "n" and "AIC"
    """
    if n==None:
        n = np.max(data)

    # plot histogram
    fig, ax = plt.subplots(1, 1)
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
        a, b = theta[0], theta[1]
        result = 0
        for i in range(len(data)):
            result += BetaBinom(a, b, n, data[i])
        return result

    def neg_loglik(theta):
        return -loglik(theta)

    # estimate the parameters by minimize -loglik
    theta0 = [1, 1]
    paras, value, iter, imode, smode = fmin_slsqp(neg_loglik, theta0, bounds=[(0.0, 10.0)] * len(theta0),
                              disp=False, full_output=True)

    ##########################
    # JAGS Model
    # jags_model = '''
    # model {
    #     for (i in 1:N) {
    #         p[i] ~ dbeta(a,b)
    #         y[i] ~ dbin(p[i], n)
    #     }
    #     a ~ dunif(0, 10)
    #     b ~ dunif(0, 10)
    # }
    # '''
    # # 1000 samples
    # sample_path = MCMC_JAGS(250, ['a', 'b'], jags_model, data)
    # a, b = np.mean(sample_path['a']), np.mean(sample_path['b'])
    ##########################

    # plot the estimated distribution
    # get PMF
    x_values = np.arange(0, n, step=1)
    pmf = np.zeros(len(x_values))
    for i in x_values:
        pmf[i] = np.exp(BetaBinom(paras[0], paras[1], n, i))
    # plot
    ax.step(x_values, pmf, color=COLOR_CONTINUOUS_FIT, lw=2, label='BetaBinomial')

    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()

    # calculate AIC
    aic = AIC(
        k=3,
        log_likelihood=loglik([paras[0], paras[1], n])
    )

    # report results in the form of a dictionary
    return {"a": paras[0], "b": paras[1], "n": n, "AIC": aic}

# 4 Binomial
def fit_binomial(data, x_label, n=None):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :param n: the number of trials in the Binomial distribution, set by experiment design
    :returns: dictionary with keys "p" and "AIC"
    """
    if n==None:
        n = np.max(data)

    # fit Binomial distribution: the MLE of p is x/n
    # if we have N data point with Xi~Bin(n,p), then sum(Xi)~Bin(n*N,p), p_hat = sum(xi)/(n*N)
    # # https://onlinecourses.science.psu.edu/stat504/node/28
    p=np.sum(data)*1.0/(len(data)*n)

    # plot histogram
    fig, ax = plt.subplots(1, 1)
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
    return {"p": p, "AIC": aic}


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
def fit_lognorm(data, x_label):
    """
    :param data: (numpy.array) observations
    :param x_label: label to show on the x-axis of the histogram
    :returns: dictionary with keys "s", "loc", "scale", and "AIC", s = sigma and scale = exp(mu)
    """

    # plot histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(data, normed=1, bins='auto', edgecolor='black', alpha=0.5, label='Frequency')

    # estimate the parameters
    s, loc, scale = scs.lognorm.fit(data, floc=0)

    # plot the estimated gamma distribution
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

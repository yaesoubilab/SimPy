import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs

COLOR_CONTINUOUS_FIT = 'r'
COLOR_DISCRETE_FIT = 'r'


def AIC(k, log_likelihood):
    """ :returns Akaike information criterion"""
    return 2 * k - 2 * log_likelihood


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
    gamma_dist = scs.gamma(a, loc, scale)
    ax.plot(x_values, gamma_dist.pdf(x_values), color=COLOR_CONTINUOUS_FIT, lw=2, label='Gamma')

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
    ax.plot(x_plot, scs.poisson.pmf(x_plot, lamb), COLOR_DISCRETE_FIT+'o', ms=8, label='Poisson')
    #ax.vlines(x_plot, 0, scs.poisson.pmf(x_plot, lamb), colors='k', lw=6, alpha=0.5)

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

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
from scipy.misc import factorial
from scipy.optimize import curve_fit

# simulate some data
np.random.seed(1)
dat_gamma = np.random.gamma(10,2,1000)
# get poisson deviated random numbers
dat_poisson = np.random.poisson(3, 1000)

def fit_poisson(data):
    # fit poisson distribution: the MLE of lambda is the sample mean
    # https://en.wikipedia.org/wiki/Poisson_distribution#Maximum_likelihood
    lamb = data.mean()

    # plot histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(data, normed=1, bins=11, range=[-0.5, 10.5],
            edgecolor='black', alpha=0.5, label='Frequency')

    # plot poisson-deviation with fitted parameter
    x_plot = np.arange(scs.poisson.ppf(0.01, lamb), scs.poisson.ppf(0.99, lamb))
    ax.plot(x_plot, scs.poisson.pmf(x_plot, lamb), 'bo', ms=8, label='poisson pmf')
    ax.vlines(x_plot, 0, scs.poisson.pmf(x_plot, lamb), colors='r', lw=5, alpha=0.5)

    # ax.set_xlabel("data")
    ax.set_ylabel("Frequency")

    # calculate AIC
    logLik = np.sum(scs.poisson.logpmf(data, lamb))
    k = 1
    aic = 2 * k - 2 * logLik

    print("Lambda:")
    print('Poisson', lamb)
    print("AIC")
    print(aic)

    ax.legend()
    plt.show()

def fit_gamma(data):
    # plot histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(data, normed=1, bins=15, edgecolor='black', alpha=0.5, label='Frequency')

    # fit gamma and plot
    a, loc, scale = scs.gamma.fit(data)

    x_plot = np.linspace(scs.gamma.ppf(0.01, a, loc, scale), scs.gamma.ppf(0.99, a, loc, scale), 100)
    rv4 = scs.gamma(a, loc, scale)
    ax.plot(x_plot, rv4.pdf(x_plot), color='orange', lw=2, label='Gamma')

    # ax.set_xlabel("data")
    ax.set_ylabel("Frequency")

    # calculate AIC
    logLik = np.sum(scs.gamma.logpdf(data, a, loc, scale))
    k = 3
    aic = 2 * k - 2 * logLik

    print("a, loc, scale:")
    print('Gamma', a, loc, scale)
    print("AIC")
    print(aic)

    ax.legend()
    plt.show()


# for simulated data from Gamma distribution,
# AIC for gamma is smaller
fit_gamma(dat_gamma)
fit_poisson(dat_gamma) # Inf since continuous

# for simulated data from Poisson distribution,
# AIC for Poisson is smaller
fit_poisson(dat_poisson)
fit_gamma(dat_poisson)
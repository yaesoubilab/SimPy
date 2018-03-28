import scr.ProbDistFitting as Fit
import numpy as np
import scr.RandomVariantGenerators as RVGs
from tests.RVGtests import get_samples
import scipy.stats as scs

# simulate some data
np.random.seed(1)

# In most functions with Location parameter, floc=0 is applied
# (fix location at 0), since if not, estimated parameters are not unique
# for example Exponential distribution only has one parameter lambda, k=1

# 1 fitting a exponential distribution
dat_exp = np.random.exponential(5, 1000)            # generate data
dictResults=Fit.fit_exp(dat_exp, 'Data')        # fit
print("Fitting Exponential:", dictResults)

# 2 fitting a beta distribution
dat_beta = 5 * np.random.beta(2, 3, 1000)            # generate data
dictResults=Fit.fit_beta(dat_beta, 'Data', min=None, max=None) # fit
print("Fitting Beta:", dictResults)


# 3 fitting a beta-binomial distribution
betabinomial_dist = RVGs.BetaBinomial(100, 2, 3) # n, a, b
dat_betabin = np.array(get_samples(betabinomial_dist, np.random))
dictResults=Fit.fit_betaBinomial(dat_betabin, 'Data', n=100) # fit
print("Fitting BetaBinomial:", dictResults)

# 4 Binomial
dat_bin = np.random.binomial(100, 0.3, 1000)
dictResults=Fit.fit_binomial(dat_bin, 'Data', n=100) # fit
print("Fitting Binomial:", dictResults)

# 5 Empirical (for int data)
dat_em = np.random.poisson(30, 1000)
dictResults=Fit.fit_empirical(dat_em, 'Data') # fit
print("Fitting Empirical:", dictResults)

# 6 fitting a gamma distribution
dat_gamma = np.random.gamma(10, 2,1000)     # generate data
dictResults=Fit.fit_gamma(dat_gamma, 'Data')        # fit
print("Fitting Gamma:", dictResults)

# 9 fitting a JohnsonSb distribution
dat_JohnsonSb = scs.johnsonsb.rvs(a=10, b=3, loc=0, scale=1, size=1000)    # generate data
dictResults=Fit.fit_johnsonSb(dat_JohnsonSb, 'Data')    # fit
print("Fitting johnsonSb:", dictResults)

# 10 fitting a JohnsonSu distribution
dat_JohnsonSu = scs.johnsonsu.rvs(a=10, b=3, loc=0, scale=1, size=1000)    # generate data
dictResults=Fit.fit_johnsonSu(dat_JohnsonSu, 'Data')    # fit
print("Fitting johnsonSu:", dictResults)

# 11 LogNormal
dat_lognorm = np.random.lognormal(0, 1, 1000)    # mean, sigma
dictResults=Fit.fit_lognorm(dat_lognorm, 'Data')    # fit (scale=exp(mean))
print("Fitting LogNormal:", dictResults)


# 17 fitting a Weibull distribution
dat_weibull = np.random.weibull(5, 1000)    # generate data
dictResults=Fit.fit_weibull(dat_weibull, 'Data')    # fit
print("Fitting Weibull:", dictResults)

# 18 fitting a Poisson distribution
dat_poisson = np.random.poisson(30, 1000)    # generate data
dictResults=Fit.fit_poisson(dat_poisson, 'Data')    # fit
print("Fitting Poisson:", dictResults)
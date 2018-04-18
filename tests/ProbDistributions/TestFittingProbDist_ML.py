import scr.FittingProbDist_ML as Fit
import numpy as np
import scr.RandomVariantGenerators as RVGs
from tests.ProbDistributions.RVGtests import get_samples
import scipy.stats as scs

# simulate some data
np.random.seed(1)

# 1 fitting a exponential distribution
dat_exp = np.random.exponential(5, 1000)            # generate data
dictResults=Fit.fit_exp(dat_exp, 'Data')        # fit
print("Fitting Exponential:", dictResults)

# 2 fitting a beta distribution
dat_beta = 5 * np.random.beta(2, 3, 1000)            # generate data
dictResults=Fit.fit_beta(dat_beta, 'Data', minimum=None, maximum=None) # fit
print("Fitting Beta:", dictResults)

# 3 fitting a beta-binomial distribution
betabinomial_dist = RVGs.BetaBinomial(100, 2, 3) # n, a, b
dat_betabin = np.array(get_samples(betabinomial_dist, np.random))
dictResults=Fit.fit_beta_binomial(dat_betabin, 'Data', n=100) # fit
print("Fitting BetaBinomial:", dictResults)

# 4 Binomial
dat_bin = np.random.binomial(100, 0.3, 1000)
dictResults=Fit.fit_binomial(dat_bin, 'Data', n=100, fixed_location=0) # fit
print("Fitting Binomial:", dictResults)

# 5 Empirical (for int data)
dat_em = np.random.poisson(30, 1000)
dictResults=Fit.fit_empirical(dat_em, 'Data') # fit
print("Fitting Empirical:", dictResults)

# 6 fitting a gamma distribution
dat_gamma = np.random.gamma(10, 2, 1000)     # generate data
dictResults=Fit.fit_gamma(dat_gamma, 'Data')        # fit
print("Fitting Gamma:", dictResults)

# 7 GammaPoisson
gamma_poisson_dist = RVGs.GammaPoisson(a=2, gamma_scale=4)
dat_gamma_poisson = np.array(get_samples(gamma_poisson_dist, np.random))
#dictResults=Fit.fit_gamma_poisson(dat_gamma_poisson, 'Data') # fit
print("Fitting GammaPoisson:", dictResults)

# 8 Geometric
dat_geom = np.random.geometric(0.3, 1000)     # generate data
dictResults=Fit.fit_geometric(dat_geom, 'Data', fixed_location=0)        # fit
print("Fitting Geometric:", dictResults)

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

# 12 NegativeBinomial
dat_neg_bin = np.random.negative_binomial(3, 0.3, 1000)    # mean, sigma
dictResults=Fit.fit_negative_binomial(dat_neg_bin, 'Data')
print("Fitting NegativeBinomial:", dictResults)

# 13 Normal
dat_norm = np.random.normal(0, 1, 1000)    # mean, sigma
dictResults=Fit.fit_normal(dat_norm, 'Data')    # fit
print("Fitting Normal:", dictResults)

# 14 Triangular
dat_tri = scs.triang.rvs(c=0.5, loc=0, scale=1, size=1000)
dictResults=Fit.fit_triang(dat_tri, 'Data')    # fit
print("Fitting Triangular:", dictResults)

# 15 Uniform
dat_unif = np.random.uniform(0, 1, 1000)    # mean, sigma
dictResults=Fit.fit_uniform(dat_unif, 'Data')    # fit
print("Fitting Uniform:", dictResults)

# 16 UniformDiscrete
dat_unifDis = scs.randint.rvs(0,100,size=1000)
dictResults=Fit.fit_uniformDiscrete(dat_unifDis, 'Data')    # fit
print("Fitting UniformDiscrete:", dictResults)

# 17 fitting a Weibull distribution
dat_weibull = np.random.weibull(5, 1000)    # generate data
dictResults=Fit.fit_weibull(dat_weibull, 'Data')    # fit
print("Fitting Weibull:", dictResults)

# 18 fitting a Poisson distribution
dat_poisson = np.random.poisson(30, 1000)    # generate data
dictResults=Fit.fit_poisson(dat_poisson, 'Data', fixed_location=0)    # fit
print("Fitting Poisson:", dictResults)
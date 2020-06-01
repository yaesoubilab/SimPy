import SimPy.FittingProbDist_ML as Fit
import numpy as np
import SimPy.RandomVariateGenerators as RVGs
from tests.ProbDistributions.RVGtests import get_samples
import scipy.stats as scs

# simulate some data
np.random.seed(1)

# 1 fitting a exponential distribution
dist = RVGs.Exponential(5, 1)
dat_exp = np.array(get_samples(dist, np.random))           # generate data
dictResults=Fit.fit_exp(dat_exp, 'Data', fixed_location=1)        # fit
print("Fitting Exponential:", dictResults)


# 3 fitting a beta-binomial distribution
dist = RVGs.BetaBinomial(100, 2, 3, loc=1, scale=2) # n, a, b
dat_betabin = np.array(get_samples(dist, np.random))
dictResults=Fit.fit_beta_binomial(dat_betabin, 'Data', fixed_location=1, fixed_scale=2, bin_width=5) # fit
print("Fitting BetaBinomial:", dictResults)

# 4 Binomial
dist = RVGs.Binomial(100, 0.3, 1)
dat_bin = np.array(get_samples(dist, np.random))
dictResults=Fit.fit_binomial(dat_bin, 'Data', fixed_location=1) # fit
print("Fitting Binomial:", dictResults)

# 5 Empirical (for int data)
dat_em = np.random.poisson(30, 1000)
dictResults=Fit.fit_empirical(dat_em, 'Data', bin_width=2.5) # fit
print("Fitting Empirical:", dictResults)

# 6 fitting a gamma distribution
dist = RVGs.Gamma(10, 1, 2)
dat_gamma = np.array(get_samples(dist, np.random))    # generate data
dictResults=Fit.fit_gamma(dat_gamma, 'Data', fixed_location=1)        # fit
print("Fitting Gamma:", dictResults)

# 7 GammaPoisson
dist = RVGs.GammaPoisson(a=2, gamma_scale=4, loc=1, scale=2)
dat_gamma_poisson = np.array(get_samples(dist, np.random))
dictResults=Fit.fit_gamma_poisson(dat_gamma_poisson, 'Data', fixed_location=1, fixed_scale=2, bin_width=2) # fit
print("Fitting GammaPoisson:", dictResults)

# 8 Geometric
dist = RVGs.Geometric(0.3, 1)
dat_geom = np.array(get_samples(dist, np.random))    # generate data
dictResults=Fit.fit_geometric(dat_geom, 'Data', fixed_location=1)        # fit
print("Fitting Geometric:", dictResults)

# 9 fitting a JohnsonSb distribution
dist = RVGs.JohnsonSb(a=10, b=3, loc=1, scale=2)
dat_JohnsonSb = np.array(get_samples(dist, np.random))    # generate data
dictResults=Fit.fit_johnsonSb(dat_JohnsonSb, 'Data', fixed_location=1)    # fit
print("Fitting johnsonSb:", dictResults)

# 10 fitting a JohnsonSu distribution
dist = RVGs.JohnsonSu(a=10, b=3, loc=1, scale=2)
dat_JohnsonSu = np.array(get_samples(dist, np.random))    # generate data
dictResults=Fit.fit_johnsonSu(dat_JohnsonSu, 'Data', fixed_location=1)    # fit
print("Fitting johnsonSu:", dictResults)

# 11 LogNormal
dist = RVGs.LogNormal(s=1, loc=1, scale=2)
dat_lognorm = np.array(get_samples(dist, np.random))    # mean, sigma
dictResults=Fit.fit_lognorm(dat_lognorm, 'Data', fixed_location=1)    # fit (scale=exp(mean))
print("Fitting LogNormal:", dictResults)

# 12 NegativeBinomial
dist = RVGs.NegativeBinomial(3, 0.3, 1)
dat_neg_bin = np.array(get_samples(dist, np.random))    # mean, sigma
dictResults=Fit.fit_negative_binomial(dat_neg_bin, 'Data', fixed_location=1)
print("Fitting NegativeBinomial:", dictResults)

# 13 Normal
dist = RVGs.Normal(0,1)
dat_norm = np.array(get_samples(dist, np.random))   # mean, sigma
dictResults=Fit.fit_normal(dat_norm, 'Data')    # fit
print("Fitting Normal:", dictResults)

# 14 Triangular
dist = RVGs.Triangular(0.5, loc=1, scale=2)
dat_tri = np.array(get_samples(dist, np.random))
dictResults=Fit.fit_triang(dat_tri, 'Data', fixed_location=1)    # fit
print("Fitting Triangular:", dictResults)

# 15 Uniform
dist = RVGs.Uniform(0, 1)
dat_unif = np.array(get_samples(dist, np.random))    # mean, sigma
dictResults=Fit.fit_uniform(dat_unif, 'Data')    # fit
print("Fitting Uniform:", dictResults)

# 16 UniformDiscrete
dist = RVGs.UniformDiscrete(0, 100)
dat_unifDis = np.array(get_samples(dist, np.random))
dictResults=Fit.fit_uniformDiscrete(dat_unifDis, 'Data')    # fit
print("Fitting UniformDiscrete:", dictResults)

# 17 fitting a Weibull distribution
dist = RVGs.Weibull(5, 1, 2)
dat_weibull = np.array(get_samples(dist, np.random))    # generate data
dictResults=Fit.fit_weibull(dat_weibull, 'Data', fixed_location=1)    # fit
print("Fitting Weibull:", dictResults)

# 18 fitting a Poisson distribution
dist = RVGs.Poisson(30, 1)
dat_poisson = np.array(get_samples(dist, np.random))    # generate data
dictResults=Fit.fit_poisson(dat_poisson, 'Data', fixed_location=1)    # fit
print("Fitting Poisson:", dictResults)
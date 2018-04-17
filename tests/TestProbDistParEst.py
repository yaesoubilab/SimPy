import scr.ProbDistParEst as Est
import numpy as np
import scr.RandomVariantGenerators as RVGs
from tests.RVGtests import get_samples
import scipy.stats as scs

# simulate some data
np.random.seed(1)

# 1 fitting a exponential distribution
dat_exp = np.random.exponential(5, 1000)            # generate data
dictResults=Est.get_expon_params(np.mean(dat_exp))        # fit
print("Fitting Exponential:", dictResults)

# 2 fitting a beta distribution
dat_beta = np.random.beta(2, 3, 1000)            # generate data
dictResults=Est.get_beta_params(np.mean(dat_beta), np.std(dat_beta)) # fit
print("Fitting Beta:", dictResults)

# 3 fitting a beta-binomial distribution
betabinomial_dist = RVGs.BetaBinomial(100, 2, 3) # n, a, b
dat_betabin = np.array(get_samples(betabinomial_dist, np.random))
dictResults=Est.get_beta_binomial_paras(np.mean(dat_betabin),np.std(dat_betabin), n=100) # fit
print("Fitting BetaBinomial:", dictResults)

# 4 Binomial
dat_bin = np.random.binomial(100, 0.3, 1000)
dictResults=Est.get_binomial_parameters(np.mean(dat_bin), np.std(dat_bin), fixed_location=0) # fit
print("Fitting Binomial:", dictResults)

# 5 Empirical (for int data)
dat_em = np.random.poisson(30, 1000)
dictResults=Est.get_empirical_parameters(dat_em, 'Data') # fit
print("Fitting Empirical:", dictResults)

# 6 fitting a Gamma distribution
dat_gamma = np.random.gamma(10, 2, 1000)     # generate data
dictResults=Est.get_gamma_parameters(np.mean(dat_gamma), np.std(dat_gamma))        # fit
print("Fitting Gamma:", dictResults)

# 7 GammaPoisson
gamma_poisson_dist = RVGs.GammaPoisson(a=2, gamma_scale=4)
dat_gamma_poisson = np.array(get_samples(gamma_poisson_dist, np.random))
dictResults=Est.get_gamma_poisson_paras(np.mean(dat_gamma_poisson),np.std(dat_gamma_poisson)) # fit
print("Fitting GammaPoisson:", dictResults)

# 8 Geometric
dat_geom = np.random.geometric(0.3, 1000)     # generate data
dictResults=Est.get_geomertic_paras(np.mean(dat_geom), fixed_location=0)        # fit
print("Fitting Geometric:", dictResults)

# # 9 fitting a JohnsonSb distribution
# dat_JohnsonSb = scs.johnsonsb.rvs(a=10, b=3, loc=0, scale=1, size=1000)    # generate data
# dictResults=Est.fit_johnsonSb(dat_JohnsonSb, 'Data')    # fit
# print("Fitting johnsonSb:", dictResults)
#
# # 10 fitting a JohnsonSu distribution
# dat_JohnsonSu = scs.johnsonsu.rvs(a=10, b=3, loc=0, scale=1, size=1000)    # generate data
# dictResults=Est.fit_johnsonSu(dat_JohnsonSu, 'Data')    # fit
# print("Fitting johnsonSu:", dictResults)
#
# # 11 LogNormal
dat_lognorm = np.random.lognormal(0, 1, 1000)    # mean, sigma
dictResults=Est.get_log_normal_parameters(np.mean(dat_lognorm), np.std(dat_lognorm))    # fit (scale=exp(mean))
print("Fitting LogNormal:", dictResults)

# 12 NegativeBinomial
dat_neg_bin = np.random.negative_binomial(3, 0.3, 1000)    # mean, sigma
dictResults=Est.get_negative_binomial_paras(np.mean(dat_neg_bin),np.std(dat_neg_bin))
print("Fitting NegativeBinomial:", dictResults)

# 13 Normal
dictResults=Est.get_normal_param(5, 2)    # fit
print("Fitting Normal:", dictResults)

# # 14 Triangular
# dat_tri = scs.triang.rvs(c=0.5, loc=0, scale=1, size=1000)
# dictResults=Est.fit_triang(dat_tri, 'Data')    # fit
# print("Fitting Triangular:", dictResults)
#
# # 15 Uniform
# dat_unif = np.random.uniform(0, 1, 1000)    # mean, sigma
# dictResults=Est.fit_uniform(dat_unif, 'Data')    # fit
# print("Fitting Uniform:", dictResults)
#
# # 16 UniformDiscrete
# dat_unifDis = scs.randint.rvs(0,100,size=1000)
# dictResults=Est.fit_uniformDiscrete(dat_unifDis, 'Data')    # fit
# print("Fitting UniformDiscrete:", dictResults)
#
# # 17 fitting a Weibull distribution
# dat_weibull = np.random.weibull(5, 1000)    # generate data
# dictResults=Est.fit_weibull(dat_weibull, 'Data')    # fit
# print("Fitting Weibull:", dictResults)
#
# # 18 fitting a Poisson distribution
# dat_poisson = np.random.poisson(30, 1000)    # generate data
# dictResults=Est.fit_poisson(dat_poisson, 'Data', fixed_location=0)    # fit
# print("Fitting Poisson:", dictResults)
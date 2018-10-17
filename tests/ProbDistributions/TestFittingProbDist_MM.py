import SimPy.FittingProbDist_MM as Est
import numpy as np
import SimPy.RandomVariantGenerators as RVGs
from tests.ProbDistributions.RVGtests import get_samples


# simulate some data
np.random.seed(1)

# 1 fitting a exponential distribution
dist = RVGs.Exponential(5, 1)
dat_exp = np.array(get_samples(dist, np.random))
dictResults=Est.get_expon_params(np.mean(dat_exp), fixed_location=1)        # fit
print("Fitting Exponential:", dictResults)

# 2 fitting a beta distribution
dist = RVGs.Beta(2, 3, loc=1, scale=2)
dat_beta = np.array(get_samples(dist, np.random))
dictResults=Est.get_beta_params(np.mean(dat_beta), np.std(dat_beta), minimum=1, maximum=3) # fit
print("Fitting Beta:", dictResults)

# 3 fitting a beta-binomial distribution
betabinomial_dist = RVGs.BetaBinomial(100, 2, 3, loc=1, scale=2) # n, a, b
dat_betabin = np.array(get_samples(betabinomial_dist, np.random))
dictResults=Est.get_beta_binomial_params(np.mean(dat_betabin),np.std(dat_betabin), n=100, fixed_location=1, fixed_scale=2) # fit
print("Fitting BetaBinomial:", dictResults)

# 4 Binomial
dist = RVGs.Binomial(100, 0.3, 1)
dat_bin = np.array(get_samples(dist, np.random))
dictResults=Est.get_binomial_params(np.mean(dat_bin), np.std(dat_bin), fixed_location=1) # fit
print("Fitting Binomial:", dictResults)

# 5 Empirical (for int data)
dat_em = np.random.poisson(30, 1000)
dictResults=Est.get_empirical_params(dat_em, bin_size=2.5) # fit
print("Fitting Empirical:", dictResults)

# 6 fitting a Gamma distribution
dist = RVGs.Gamma(10, 1, 2)
dat_gamma = np.array(get_samples(dist, np.random))    # generate data
dictResults=Est.get_gamma_params(np.mean(dat_gamma), np.std(dat_gamma), fixed_location=1)        # fit
print("Fitting Gamma:", dictResults)

# 7 GammaPoisson
gamma_poisson_dist = RVGs.GammaPoisson(a=2, gamma_scale=4, loc=1, scale=2)
dat_gamma_poisson = np.array(get_samples(gamma_poisson_dist, np.random))
dictResults=Est.get_gamma_poisson_params(np.mean(dat_gamma_poisson),np.std(dat_gamma_poisson),
                                        fixed_location=1, fixed_scale=2) # fit
print("Fitting GammaPoisson:", dictResults)

# 8 Geometric
dist = RVGs.Geometric(0.3, 1)
dat_geom = np.array(get_samples(dist, np.random))    # generate data
dictResults=Est.get_geometric_params(np.mean(dat_geom), fixed_location=1)        # fit
print("Fitting Geometric:", dictResults)

# # 9 fitting a JohnsonSb distribution
# dat_JohnsonSb = scs.johnsonsb.rvs(a=10, b=3, loc=0, scale=1, size=1000)    # generate data
# dictResults=Est.fit_johnsonSb(dat_JohnsonSb, 'Data')    # fit
# print("Fitting johnsonSb:", dictResults)

# # 10 fitting a JohnsonSu distribution
# dat_JohnsonSu = scs.johnsonsu.rvs(a=10, b=3, loc=0, scale=1, size=1000)    # generate data
# dictResults=Est.fit_johnsonSu(dat_JohnsonSu, 'Data')    # fit
# print("Fitting johnsonSu:", dictResults)

# # 11 LogNormal
dist = RVGs.LogNormal(s=1, loc=1, scale=2)
dat_lognorm = np.array(get_samples(dist, np.random))    # mean, sigma
dictResults=Est.get_lognormal_params(np.mean(dat_lognorm), np.std(dat_lognorm), fixed_location=1)    # fit (scale=exp(mean))
print("Fitting LogNormal:", dictResults)

# 12 NegativeBinomial
dist = RVGs.NegativeBinomial(3, 0.3, 1)
dat_neg_bin = np.array(get_samples(dist, np.random))    # mean, sigma
dictResults=Est.get_negative_binomial_params(np.mean(dat_neg_bin), np.std(dat_neg_bin), fixed_location=1)
print("Fitting NegativeBinomial:", dictResults)

# 13 Normal
dictResults=Est.get_normal_params(5, 2)    # fit
print("Fitting Normal:", dictResults)

# # 14 Triangular
# dat_tri = scs.triang.rvs(c=0.5, loc=0, scale=1, size=1000)
# dictResults=Est.fit_triang(dat_tri, 'Data')    # fit
# print("Fitting Triangular:", dictResults)

# 15 Uniform
dist = RVGs.Uniform(0, 1)
dat_unif = np.array(get_samples(dist, np.random))    # mean, sigma
dictResults=Est.get_uniform_params(np.mean(dat_unif), np.std(dat_unif))    # fit
print("Fitting Uniform:", dictResults)

# 16 UniformDiscrete
dist = RVGs.UniformDiscrete(0, 100)
dat_unifDis = np.array(get_samples(dist, np.random))
dictResults=Est.get_uniform_discrete_params(np.mean(dat_unifDis), np.std(dat_unifDis))    # fit
print("Fitting UniformDiscrete:", dictResults)

# 17 fitting a Weibull distribution
dist = RVGs.Weibull(5, 1, 2)
dat_weibull = np.array(get_samples(dist, np.random))    # generate data
dictResults=Est.get_weibull_params(np.mean(dat_weibull), np.std(dat_weibull), fixed_location=1)    # fit
print("Fitting Weibull:", dictResults)

# 18 fitting a Poisson distribution
dist = RVGs.Poisson(30, 1)
dat_poisson = np.array(get_samples(dist, np.random))    # generate data
dictResults=Est.get_poisson_params(np.mean(dat_poisson), fixed_location=1)    # fit
print("Fitting Poisson:", dictResults)

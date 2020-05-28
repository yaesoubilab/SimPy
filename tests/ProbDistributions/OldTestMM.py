import numpy as np
import SimPy.RandomVariantGenerators as RVGs
from tests.ProbDistributions.RVGtests import get_samples


# simulate some data
np.random.seed(1)

# 1 fitting a exponential distribution
dist = RVGs.Exponential(5, 1)
dat_exp = np.array(get_samples(dist, np.random))
dictResults = RVGs.Exponential.fit_mm(np.mean(dat_exp), fixed_location=1)        # fit
print("Fitting Exponential:", dictResults)


# 4 Binomial
dist = RVGs.Binomial(100, 0.3, 1)
dat_bin = np.array(get_samples(dist, np.random))
dictResults = RVGs.Binomial.fit_mm(np.mean(dat_bin), np.std(dat_bin), fixed_location=1) # fit
print("Fitting Binomial:", dictResults)

# 5 Empirical (for int data)
dat_em = np.random.poisson(30, 1000)
dictResults = RVGs.Empirical.fit_mm(dat_em, bin_size=2.5) # fit
print("Fitting Empirical:", dictResults)

# 6 fitting a Gamma distribution
dist = RVGs.Gamma(10, 1, 2)
dat_gamma = np.array(get_samples(dist, np.random))    # generate data
dictResults = RVGs.Gamma.fit_mm(np.mean(dat_gamma), np.std(dat_gamma), fixed_location=1)        # fit
print("Fitting Gamma:", dictResults)

# 7 GammaPoisson
gamma_poisson_dist = RVGs.GammaPoisson(a=2, gamma_scale=4, loc=1, scale=2)
dat_gamma_poisson = np.array(get_samples(gamma_poisson_dist, np.random))
dictResults = RVGs.GammaPoisson.fit_mm(np.mean(dat_gamma_poisson),np.std(dat_gamma_poisson),
                                        fixed_location=1, fixed_scale=2) # fit
print("Fitting GammaPoisson:", dictResults)

# 8 Geometric
dist = RVGs.Geometric(0.3, 1)
dat_geom = np.array(get_samples(dist, np.random))    # generate data
dictResults = RVGs.Geometric.fit_mm(np.mean(dat_geom), fixed_location=1)        # fit
print("Fitting Geometric:", dictResults)

# # 11 LogNormal
dist = RVGs.LogNormal(s=1, loc=1, scale=2)
dat_lognorm = np.array(get_samples(dist, np.random))    # mean, sigma
dictResults = RVGs.LogNormal.fit_mm(np.mean(dat_lognorm), np.std(dat_lognorm), fixed_location=1)    # fit (scale=exp(mean))
print("Fitting LogNormal:", dictResults)

# 12 NegativeBinomial
dist = RVGs.NegativeBinomial(3, 0.3, 1)
dat_neg_bin = np.array(get_samples(dist, np.random))    # mean, sigma
dictResults = RVGs.NegativeBinomial.fit_mm(np.mean(dat_neg_bin), np.std(dat_neg_bin), fixed_location=1)
print("Fitting NegativeBinomial:", dictResults)

# 13 Normal
dictResults = RVGs.Normal.fit_mm(5, 2)    # fit
print("Fitting Normal:", dictResults)

# 18 fitting a Poisson distribution
dist = RVGs.Poisson(30, 1)
dat_poisson = np.array(get_samples(dist, np.random))    # generate data
dictResults = RVGs.Poisson.fit_mm(np.mean(dat_poisson), fixed_location=1)    # fit
print("Fitting Poisson:", dictResults)
# # 14 Triangular
# dat_tri = scs.triang.rvs(c=0.5, loc=0, scale=1, size=1000)
# dictResults=Est.fit_triang(dat_tri, 'Data')    # fit
# print("Fitting Triangular:", dictResults)

# 15 Uniform
dist = RVGs.Uniform(0, 1)
dat_unif = np.array(get_samples(dist, np.random))    # mean, sigma
dictResults = RVGs.Uniform.fit_mm(np.mean(dat_unif), np.std(dat_unif))    # fit
print("Fitting Uniform:", dictResults)

# 16 UniformDiscrete
dist = RVGs.UniformDiscrete(0, 100)
dat_unifDis = np.array(get_samples(dist, np.random))
dictResults = RVGs.UniformDiscrete.fit_mm(np.mean(dat_unifDis), np.std(dat_unifDis))    # fit
print("Fitting UniformDiscrete:", dictResults)

# 17 fitting a Weibull distribution
dist = RVGs.Weibull(5, 1, 2)
dat_weibull = np.array(get_samples(dist, np.random))    # generate data
dictResults = RVGs.Weibull.fit_mm(np.mean(dat_weibull), np.std(dat_weibull), fixed_location=1)    # fit
print("Fitting Weibull:", dictResults)



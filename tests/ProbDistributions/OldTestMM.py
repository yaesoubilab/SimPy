import numpy as np
import SimPy.RandomVariantGenerators as RVGs
from tests.ProbDistributions.RVGtests import get_samples


# simulate some data
np.random.seed(1)


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



import numpy as np
from tests import RVGtests as Tests

# use numpy random number generator
rnd = np.random
rnd.seed(1)

print('')

# tests
Tests.test_exponential(rnd, mean=10)
Tests.test_bernoulli(rnd, p=.2)
Tests.test_beta(rnd, a=2, b=5)
Tests.test_betabinomial(rnd, n=100, a=2, b=3)
Tests.test_binomial(rnd, n=1000, p=.2)
Tests.test_dirichlet(rnd, a=[1,2,3])
Tests.test_empirical(rnd, prob=[0.2,0.2,0.6])
Tests.test_gamma(rnd, shape=2, scale=4)
Tests.test_gammapoisson(rnd, shape=2, scale=4)
Tests.test_geometric(rnd, p=.2)
Tests.test_johnsonsb(rnd, a=10, b=3, loc=10, scale=100)
Tests.test_johnsonsu(rnd, a=10, b=3, loc=1, scale=2)
Tests.test_lognormal(rnd, mean=10, sigma=1.2)
Tests.test_multinomial(rnd, n=1000, pvals=.2)
Tests.test_negativebinomial(rnd, n=100, p=.2)
Tests.test_normal(rnd, mean=5, sigma=1.2)
Tests.test_poisson(rnd,lam=2)
Tests.test_triangular(rnd, l=2, m=6, r=7)
Tests.test_uniform(rnd,l=2, r=7)
Tests.test_uniformdiscrete(rnd, l=0, r=5)
Tests.test_weibull(rnd, a=0.5)
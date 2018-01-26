import numpy as np
from tests import RVGtests as Tests
import scipy.stats as scipy

# use numpy random number generator
rnd = np.random
rnd.seed(1)

print('')

# tests
Tests.test_exponential(rnd, mean=10)
Tests.test_bernoulli(rnd, p=.2)
Tests.test_beta(rnd, a=2, b=5)
#Test.test_betabinomial(rnd, n, a, b)
Tests.test_binomial(rnd, n=1000, p=.2)
#tests.test_dirichlet(rnd, a)
#tests.test_empirical(rnd,
Tests.test_gamma(rnd, shape=2, scale=4)
#tests.test_gammapoisson(rnd
Tests.test_geometric(rnd, p=.2)
#tests.test_johnsonsb(scipy,
#tests.test_johnsonSb(scipy,
#tests.test_johnsonSI(scipy,
#tests.test_johnsonSu(scipy,
Tests.test_lognormal(rnd, mean=10, sigma=1.2)
Tests.test_multinomial(rnd, n=1000, pvals=.2)
Tests.test_negativebinomial(rnd, n=100, p=.2)
Tests.test_normal(rnd, mean=5, sigma=1.2)
Tests.test_poisson(rnd,lam=2)
Tests.test_triangular(rnd, l=2, m=6, r=7)
Tests.test_uniform(rnd,l=2, r=7)
#tests.test_uniformdiscrete(rnd,
Tests.test_weibull(rnd, a=0.5)
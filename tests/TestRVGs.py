import numpy as np
from tests import RVGtests as Tests
import scr.RandomVariantGenerators as rndSupport

# use numpy random number generator
rng = rndSupport.RNG(1)
#rnd = np.random
#rnd.seed(1)

print('')

# tests
Tests.test_bernoulli(rng, p=.2)
Tests.test_beta(rng, a=2, b=5)
Tests.test_betabinomial(rng, n=100, a=2, b=3)
Tests.test_binomial(rng, n=1000, p=.2)
Tests.test_dirichlet(rng, a=[1, 2, 3])
Tests.test_empirical(rng, prob=[0.2, 0.2, 0.6])
Tests.test_exponential(rng, mean=10)
Tests.test_gamma(rng, shape=2, scale=4)
Tests.test_gammapoisson(rng, shape=2, scale=4)
Tests.test_geometric(rng, p=.2)
Tests.test_johnsonsb(rng, a=10, b=3, loc=10, scale=100)
Tests.test_johnsonsu(rng, a=10, b=3, loc=1, scale=2)
Tests.test_lognormal(rng, mean=10, sigma=1.2)
Tests.test_multinomial(rng, n=1000, pvals=.2)
Tests.test_negativebinomial(rng, n=100, p=.2)
Tests.test_normal(rng, mean=5, sigma=1.2)
Tests.test_poisson(rng, lam=2)
Tests.test_triangular(rng, l=2, m=6, r=7)
Tests.test_uniform(rng, l=2, r=7)
Tests.test_uniformdiscrete(rng, l=0, r=5)
Tests.test_weibull(rng, a=0.5)
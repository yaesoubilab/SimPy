from tests.ProbDistributions import RVGtests as Tests
import SimPy.RandomVariantGenerators as rndSupport

# use numpy random number generator
rng = rndSupport.RNG(1)

print('')

# tests

Tests.test_bernoulli(rng, p=.2)
Tests.test_beta(rng, a=2, b=5, loc=1,scale=2)
Tests.test_exponential(rng, scale=10, loc=1)
Tests.test_beta_binomial(rng, n=100, a=2, b=3, loc=1, scale=2)
Tests.test_binomial(rng, n=1000, p=.2, loc=1)
Tests.test_dirichlet(rng, a=[1, 2, 3])
Tests.test_empirical(rng, prob=[0.2, 0.2, 0.6])

Tests.test_gamma(rng, a=2, scale=4, loc=1)
Tests.test_gamma_poisson(rng, a=2, gamma_scale=4, loc=1, scale=1)
Tests.test_geometric(rng, p=.2, loc=1)
Tests.test_johnsonsb(rng, a=10, b=3, loc=10, scale=100)
Tests.test_johnsonsu(rng, a=10, b=3, loc=1, scale=2)
Tests.test_lognormal(rng, s=0.2, loc=2, scale=1.1)
Tests.test_multinomial(rng, n=1000, pvals=.2)
Tests.test_negative_binomial(rng, n=10, p=.2, loc=1)
Tests.test_normal(rng, loc=5, scale=1.2)
Tests.test_poisson(rng, mu=2)
Tests.test_triangular(rng, c=0.2, loc=6, scale=7)
Tests.test_uniform(rng, loc=2, scale=7)
Tests.test_uniform_discrete(rng, l=0, r=5)
Tests.test_weibull(rng, c=0.5, loc=1, scale=1.5)
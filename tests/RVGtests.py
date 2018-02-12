import numpy as np
import scr.RandomVariantGenerators as RVGs
import math
import scipy.stats as scipy


def print_test_results(dist_name, samples, expectation, variance):
    print('Testing ' + dist_name + ':')
    print('  E[x] = {ex:.{prec}f} | Sample mean = {sm:.{prec}f}'.format(
        ex=expectation, sm=np.average(samples), prec=3))
    print('  Var[x] = {var:.{prec}f} | Sample variance = {sv:.{prec}f}\n'.format(
        var=variance, sv=np.var(samples), prec=3))


def print_test_results_multivariate(dist_name, samples, expectation, variance):
    print('Testing ' + dist_name + ':')
    print("  E[x] = %(ex)s | Sample mean = %(sm)s" % {'ex':expectation, 'sm': np.average(samples,axis=1)})
    print("  Var[x] = %(var)s | Sample variance = %(sv)s" % \
          {'var': variance, 'sv': np.var(samples, axis=1)})


def get_samples(dist, rnd):
    samples = []
    for i in range(0, 10000):
        # get 10000 samples
        samples.append(dist.sample(rnd))
    return samples

def get_samples_multivariate(dist, rnd):
    samples = np.zeros([len(dist.a), 10000])
    for i in range(0, 10000):
        # get 10000 samples
        samples[:,i] = dist.sample(rnd)
    return samples


def test_exponential(rnd, mean):

    # exponential random variate generator
    exp_dist = RVGs.Exponential(mean)

    # obtain samples
    samples = get_samples(exp_dist, rnd)

    # report mean and variance
    print_test_results('Exponential', samples,
                       expectation=mean,
                       variance=mean**2)


def test_bernoulli(rnd, p):

    # bernoulli random variate generator
    bernoulli_dist = RVGs.Bernoulli(p)

    # obtain samples
    samples = get_samples(bernoulli_dist, rnd)

    # report mean and variance
    print_test_results('Bernoulli', samples,
                       expectation=p,
                       variance=p*(1-p))


def test_beta(rnd, a, b):

    # beta random variate generator
    beta_dist = RVGs.Beta(a, b)

    # obtain samples
    samples = get_samples(beta_dist, rnd)

    # report mean and variance
    print_test_results('Beta', samples,
                       expectation=a/(a + b),
                       variance=(a*b)/((a+b+1)*(a+b)**2))


def test_betabinomial(rnd, n, a, b):

    # beta random variate generator
    betabinomial_dist = RVGs.BetaBinomial(n, a, b)

    # obtain samples
    samples = get_samples(betabinomial_dist, rnd)

    # report mean and variance
    print_test_results('BetaBinomial', samples,
                       expectation=a*n/(a + b),
                       variance=(n*a*b*(a+b+n))/((a+b)**2*(a+b+1)))


def test_binomial(rnd, n, p):

    # bimonial random variate generator
    binomial_dist = RVGs.Binomial(n, p)

    # obtain samples
    samples = get_samples(binomial_dist, rnd)

    # report mean and variance
    print_test_results('Binomial', samples,
                       expectation=n*p,
                       variance=n*p*(1-p))


def test_dirichlet(rnd, a):
    # dirichlet random variate generator
    dirichlet_dist = RVGs.Dirichlet(a)

    # obtain samples
    samples = get_samples_multivariate(dirichlet_dist, rnd)

    # report mean and variance
    a0 = sum(a)
    if type(a) == list:
        a = np.array(a)
    mean = a * (1.0/a0)
    var = np.zeros(len(a))
    for i in range(len(a)):
        var[i] = (a[i]*(a0-a[i]))/(((a0)**2)*(a0+1.0))

    print_test_results_multivariate('Dirichlet', samples,
                       expectation=mean,
                       variance=var)


def test_empirical(rnd, prob):
    # empirical random variate generator
    empirical_dist = RVGs.Empirical(prob)

    # obtain samples
    samples = get_samples(empirical_dist, rnd)

    # report mean and variance
    if type(prob) == list:
        prob = np.array(prob)

    outcome = np.array(range(len(prob)))

    mean = sum(outcome*prob)
    var = sum((outcome**2)*prob) - mean**2

    print_test_results('Empirical', samples,
                       expectation=mean,
                       variance=var)


def test_gamma(rnd, shape, scale):
    # gamma random variate generator
    gamma_dist = RVGs.Gamma(shape, scale)

    # obtain samples
    samples = get_samples(gamma_dist, rnd)

    # report mean and variance
    print_test_results('Gamma', samples,
                       expectation=shape*scale,
                       variance=shape*scale**2)


def test_gammapoisson(rnd, shape, scale):
    # gamma-poisson random variate generator
    gammapoisson_dist = RVGs.GammaPoisson(shape, scale)

    # obtain samples
    samples = get_samples(gammapoisson_dist, rnd)

    # report mean and variance
    print_test_results('GammaPoisson', samples,
                       expectation=shape*scale,
                       variance=shape*scale + shape*(scale**2))


def test_geometric(rnd, p):
    # geometric random variate generator
    geometric_dist = RVGs.Geometric(p)

    # obtain samples
    samples = get_samples(geometric_dist, rnd)

    # report mean and variance
    print_test_results('Geometric', samples,
                       expectation=1/p,
                       variance=(1-p)/(p**2))


def test_johnsonsb(rnd, a, b, loc, scale):
    # johnsonSb random variate generator
    johnsonsb_dist = RVGs.JohnsonSb(a, b, loc, scale)

    # obtain samples
    samples = get_samples(johnsonsb_dist, rnd)

    # report mean and variance
    mean = scipy.johnsonsb.mean(a,b,loc,scale)
    var = scipy.johnsonsb.var(a,b,loc,scale)

    print_test_results('JohnsonSb', samples,
                       expectation=mean,
                       variance=var)


def test_johnsonsu(rnd, a, b, loc, scale):
    # johnsonSu random variate generator
    johnsonsu_dist = RVGs.JohnsonSu(a, b, loc, scale)

    # obtain samples
    samples = get_samples(johnsonsu_dist, rnd)

    # report mean and variance
    mean = scipy.johnsonsu.mean(a,b,loc,scale)
    var = scipy.johnsonsu.var(a,b,loc,scale)

    print_test_results('JohnsonSu', samples,
                       expectation=mean,
                       variance=var)


def test_lognormal(rnd, mean, sigma):
    #lognormal random variate generator
    lognormal_dist = RVGs.LogNormal(mean, sigma)

    # obtain samples
    samples = get_samples(lognormal_dist, rnd)

    # report mean and variance
    print_test_results('LogNormal', samples,
                       expectation=math.exp(mean +sigma**2/2),
                       variance=(math.exp(sigma**2-1))*math.exp(2*mean + sigma**2)
                       )


def test_multinomial(rnd, n, pvals):
    # multinomial random variate generator
    multinomial_dist = RVGs.Binomial(n, pvals)

    # obtain samples
    samples = get_samples(multinomial_dist, rnd)

    # report mean and variance
    print_test_results('Multinomial', samples,
                       expectation=n*pvals,
                       variance=n*pvals*(1-pvals)
                       )


def test_negativebinomial(rnd, n, p):

    # negative bimonial random variate generator
    negativebinomial_dist = RVGs.NegativeBinomial(n, p)

    # obtain samples
    samples = get_samples(negativebinomial_dist, rnd)

    # report mean and variance
    print_test_results('Negative Binomial', samples,
                       expectation=(n*p)/(1-p),
                       variance=(n*p)/((1-p)**2)
                       )


def test_normal(rnd, mean, sigma):
    #normal random variate generator
    normal_dist = RVGs.Normal(mean, sigma)

    # obtain samples
    samples = get_samples(normal_dist, rnd)

    # report mean and variance
    print_test_results('Normal', samples,
                       expectation=mean,
                       variance=sigma**2
                       )


def test_poisson(rnd, lam):
    # poisson random variate generator
    poisson_dist = RVGs.Poisson(lam)

    # obtain samples
    samples = get_samples(poisson_dist, rnd)

    # report mean and variance
    print_test_results('Poisson', samples,
                       expectation=lam,
                       variance=lam
                       )


def test_triangular(rnd, l, m, r):
    # triangular random variate generator
    triangular_dist = RVGs.Triangular(l, m, r)

    # obtain samples
    samples = get_samples(triangular_dist, rnd)

    # report mean and variance
    print_test_results('Triangular', samples,
                       expectation=(l+m+r)/3,
                       variance=(l**2 + m**2 + r**2 -l*r - l*m - r*m)/18
                       )


def test_uniform(rnd, l, r):
    # uniform random variate generator
    uniform_dist = RVGs.Uniform(l, r)

    # obtain samples
    samples = get_samples(uniform_dist, rnd)

    # report mean and variance
    print_test_results('Uniform', samples,
                       expectation=(l + r) / 2,
                       variance=(r-l)**2/12
                       )


def test_uniformdiscrete(rnd, l, r):
    # uniform discrete random variate generator
    uniformdiscrete_dist = RVGs.UniformDiscrete(l, r)

    # obtain samples
    samples = get_samples(uniformdiscrete_dist, rnd)

    # report mean and variance
    print_test_results('Uniform Discrete', samples,
                       expectation=(l + r) / 2.0,
                       variance=((r-l+1)**2 - 1)/12.0
                       )


def test_weibull(rnd, a):
    # weibull random variate generator
    weibull_dist = RVGs.Weibull(a)

    # obtain samples
    samples = get_samples(weibull_dist, rnd)

    # report mean and variance
    print_test_results('Weibull', samples,
                       expectation=math.gamma(1 + 1/a),
                       variance=math.gamma(1 + 2/a) - (math.gamma(1 + 1/a)**2)
                       )

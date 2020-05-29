import numpy as np
import SimPy.RandomVariantGenerators as RVGs
import math
import scipy.stats as scipy


def print_test_results(dist_name, samples, expectation, variance):
    print('Testing ' + dist_name + ':')
    print('  E[x] = {ex:.{prec}f} | Sample mean = {sm:.{prec}f}'.format(
        ex=expectation, sm=np.average(samples), prec=3))
    print('  Var[x] = {var:.{prec}f} | Sample variance = {sv:.{prec}f}\n'.format(
        var=variance, sv=np.var(samples), prec=3))


def print_test_results_multivariate(dist_name, samples, expectation, variance, axis):
    print('Testing ' + dist_name + ':')
    print("  E[x] = %(ex)s | Sample mean = %(sm)s" % {'ex':expectation, 'sm': np.average(samples, axis=axis)})
    print("  Var[x] = %(var)s | Sample variance = %(sv)s\n" % \
          {'var': variance, 'sv': np.var(samples, axis=axis)})


def get_samples(dist, rnd):
    """ sampling from a uni-variate distribution """

    samples = []
    for i in range(0, 10000):
        # get 10000 samples
        samples.append(dist.sample(rnd))
    return samples


def get_samples_multivariate(dist, rnd):
    """ sampling from a multi-variate distribution """

    samples = np.zeros([len(dist.a), 10000])
    for i in range(0, 10000):
        # get 10000 samples
        samples[:, i] = dist.sample(rnd)
    return samples


def test_rng(rnd):
    # obtain samples
    samples=[]
    for i in range(10000):
        samples.append(rnd.sample())
    
    # report mean and variance
    print_test_results('RNG', samples,
                       expectation=0.5,
                       variance=1/12)


def test_exponential(rnd, scale, loc=0):

    # exponential random variate generator
    exp_dist = RVGs.Exponential(scale, loc)

    # obtain samples
    samples = get_samples(exp_dist, rnd)

    # report mean and variance
    print_test_results('Exponential', samples,
                       expectation=scale+loc,
                       variance=scale**2)


def test_bernoulli(rnd, p):

    # bernoulli random variate generator
    bernoulli_dist = RVGs.Bernoulli(p)

    # obtain samples
    samples = get_samples(bernoulli_dist, rnd)

    # report mean and variance
    print_test_results('Bernoulli', samples,
                       expectation=p,
                       variance=p*(1-p))


def test_beta(rnd, a, b, loc=0, scale=1):

    # beta random variate generator
    beta_dist = RVGs.Beta(a, b, loc, scale)

    # obtain samples
    samples = get_samples(beta_dist, rnd)

    # report mean and variance
    print_test_results('Beta', samples,
                       expectation=(a*1.0/(a + b))*scale + loc,
                       variance=((a*b)/((a+b+1)*(a+b)**2.0)) * scale**2)


def test_beta_binomial(rnd, n, a, b, loc=0):

    # beta random variate generator
    beta_binomial_dist = RVGs.BetaBinomial(n, a, b, loc)

    # obtain samples
    samples = get_samples(beta_binomial_dist, rnd)

    # report mean and variance
    print_test_results('BetaBinomial', samples,
                       expectation=(a*n/(a + b)) + loc,
                       variance=(n*a*b*(a+b+n))/((a+b)**2*(a+b+1)))


def test_binomial(rnd, n, p, loc=0):

    # bimonial random variate generator
    binomial_dist = RVGs.Binomial(n, p, loc)

    # obtain samples
    samples = get_samples(binomial_dist, rnd)

    # report mean and variance
    print_test_results('Binomial', samples,
                       expectation=n*p + loc,
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
                                    variance=var,
                                    axis=1)


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


def test_gamma(rnd, a, loc=0, scale=1):
    # gamma random variate generator
    gamma_dist = RVGs.Gamma(a=a, scale=scale, loc=loc)

    # obtain samples
    samples = get_samples(gamma_dist, rnd)

    # report mean and variance
    print_test_results('Gamma', samples,
                       expectation=a*scale + loc,
                       variance=a*scale**2)


def test_gamma_poisson(rnd, a, gamma_scale, loc=0, scale=1):
    # gamma-poisson random variate generator
    gamma_poisson_dist = RVGs.GammaPoisson(a=a, gamma_scale=gamma_scale, scale=scale, loc=loc)

    # obtain samples
    samples = get_samples(gamma_poisson_dist, rnd)

    # report mean and variance
    print_test_results('GammaPoisson', samples,
                       expectation=(a*gamma_scale)*scale + loc,
                       variance=(a*gamma_scale + a*(gamma_scale**2))*scale**2)


def test_geometric(rnd, p, loc=0):
    # geometric random variate generator
    geometric_dist = RVGs.Geometric(p, loc)

    # obtain samples
    samples = get_samples(geometric_dist, rnd)

    # report mean and variance
    print_test_results('Geometric', samples,
                       expectation=1/p + loc,
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


def test_lognormal(rnd, mu, sigma, loc):
    #lognormal random variate generator
    lognormal_dist = RVGs.LogNormal(mu=mu, sigma=sigma,loc=loc)

    # obtain samples
    samples = get_samples(lognormal_dist, rnd)

    # report mean and variance
    print_test_results('LogNormal', samples,
                       expectation=np.exp(mu + 0.5*sigma**2) + loc,
                       variance=(np.exp(sigma**2) - 1.0) * np.exp(2*mu + sigma**2)
                       )


def test_multinomial(rnd, n, pvals):
    # multinomial random variate generator
    multinomial_dist = RVGs.Multinomial(n, pvals)

    # obtain samples
    samples = get_samples(multinomial_dist, rnd)

    pvals = np.array(pvals)

    # report mean and variance
    print_test_results_multivariate('Multinomial', samples,
                                    expectation=n*pvals,
                                    variance=n*pvals*(1-pvals),
                                    axis=0
                                    )


def test_negative_binomial(rnd, n, p, loc=0):

    # negative bimonial random variate generator
    negative_binomial_dist = RVGs.NegativeBinomial(n, p, loc)

    # obtain samples
    samples = get_samples(negative_binomial_dist, rnd)

    # get theoretical mean and variance
    mean = scipy.nbinom.stats(n, p, loc, moments='m')
    mean = np.asarray(mean).item()
    var = scipy.nbinom.stats(n, p, loc, moments='v')
    var = np.asarray(var).item()

    # report mean and variance
    print_test_results('Negative Binomial', samples,
                       expectation=mean,
                       variance=var
                       )


def test_non_homogeneous_exponential(rnd, rates, delta_t=1):
    # non homogeneous exponential random variate generator
    nhexp_dist = RVGs.NonHomogeneousExponential(rates=rates, delta_t=delta_t)

    # obtain samples
    samples = get_samples(nhexp_dist, rnd)

    # report mean and variance
    print_test_results('Non-Homogeneous Exponential', samples,
                       expectation=0,
                       variance=0
                       )


def test_normal(rnd, loc=0, scale=1):
    #normal random variate generator
    normal_dist = RVGs.Normal(loc, scale)

    # obtain samples
    samples = get_samples(normal_dist, rnd)

    # report mean and variance
    print_test_results('Normal', samples,
                       expectation=loc,
                       variance=scale**2
                       )


def test_poisson(rnd, mu, loc=0):
    # poisson random variate generator
    poisson_dist = RVGs.Poisson(mu, loc)

    # obtain samples
    samples = get_samples(poisson_dist, rnd)

    # report mean and variance
    print_test_results('Poisson', samples,
                       expectation=mu+loc,
                       variance=mu
                       )


def test_triangular(rnd, c, loc=0, scale=1):
    # triangular random variate generator
    triangular_dist = RVGs.Triangular(c, loc, scale)

    # obtain samples
    samples = get_samples(triangular_dist, rnd)

    # get theoretical variance
    var = scipy.triang.stats(c, loc, scale, moments='v')
    var = np.asarray(var).item()

    # report mean and variance
    print_test_results('Triangular', samples,
                       expectation=(3*loc + scale+ c*scale)/3.0,
                       variance=var
                       )


def test_uniform(rnd, loc=0, scale=1):
    # uniform random variate generator
    uniform_dist = RVGs.Uniform(scale=scale, loc=loc)

    # obtain samples
    samples = get_samples(uniform_dist, rnd)

    # report mean and variance
    print_test_results('Uniform', samples,
                       expectation=(2*loc + scale) / 2.0,
                       variance=scale**2/12.0
                       )


def test_uniform_discrete(rnd, l, r):
    # uniform discrete random variate generator
    uniformdiscrete_dist = RVGs.UniformDiscrete(l, r)

    # obtain samples
    samples = get_samples(uniformdiscrete_dist, rnd)

    # report mean and variance
    print_test_results('Uniform Discrete', samples,
                       expectation=(l + r) / 2.0,
                       variance=((r-l+1)**2 - 1)/12.0
                       )


def test_weibull(rnd, a, loc=0, scale=1):
    # weibull random variate generator
    weibull_dist = RVGs.Weibull(a=a, scale=scale,loc=loc)

    # obtain samples
    samples = get_samples(weibull_dist, rnd)

    # get theoretical variance
    var = scipy.weibull_min.stats(a, loc, scale, moments='v')
    var = np.asarray(var).item()

    # report mean and variance
    print_test_results('Weibull', samples,
                       expectation=math.gamma(1.0 + 1/a) * scale + loc,
                       variance=var
                       )

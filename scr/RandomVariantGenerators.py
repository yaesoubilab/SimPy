import scipy.stats as scipy
import numpy as np

class RVG(object):
    def __init__(self):
        pass

    def sample(self, numpy_rnd):
        """
        :param numpy_rnd: numpy .random object
        :returns one realization from the defined probability distribution """

        # abstract method to be overridden in derived classes to process an event
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")


class Exponential(RVG):
    def __init__(self, mean):
        """
        E[X] = mean
        Var[X] = mean**2
        """
        RVG.__init__(self)
        self.mean = mean

    def sample(self, numpy_rnd):
        return numpy_rnd.exponential(self.mean)


class Bernoulli(RVG):
    def __init__(self, p):
        """
        E[X] = p
        Var[X] = p(1-p)
        """
        RVG.__init__(self)
        self.p = p

    def sample(self, numpy_rnd):
        sample = 0
        if numpy_rnd.sample() <= self.p:
            sample = 1
        return sample


class Beta(RVG):
    def __init__(self, a, b):
        """
        E[X] = a/(a + b)
        Var[X] = ab/[(a + b)**2(a + b + 1)]
        """
        RVG.__init__(self)
        self.a = a
        self.b = b

    def sample(self, numpy_rnd):
        return numpy_rnd.beta(self.a, self.b)


class BetaBinomial(RVG):
    def __init__(self, n, a, b):
        """
        E[X] = na/(a+b)
        Var[X] = (nab(a+b+n))/((a+b)**2(a+b+1))
        """
        RVG.__init__(self)
        self.n = n
        self.a = a
        self.b = b

    def sample(self, numpy_rnd):
        """
        ref: https://blogs.sas.com/content/iml/2017/11/20/simulate-beta-binomial-sas.html
        :param numpy_rnd: numpy.random object
        :return: a realization from the Beta Binomial distribution
        """
        sample_p = numpy_rnd.beta(self.a, self.b)
        sample = numpy_rnd.binomial(self.n, sample_p)

        return sample


class Binomial(RVG):
    def __init__(self, N, p):
        """
        E[X] = Np
        Var[X] = Np(1-p)
        """
        RVG.__init__(self)
        self.N = N
        self.p = p

    def sample(self, numpy_rnd):
        return numpy_rnd.binomial(self.N, self.p)


class Dirichlet(RVG):
    def __init__(self, a):
        """
        E[Xi] = ai/a0
        Var[Xi] = (ai(a0-ai))/((a0)**2(a0+1)) where a0 = sum all ai.
        :param a: array or list
        """
        RVG.__init__(self)
        self.a = a

    def sample(self, numpy_rnd):
        """
        :param numpy_rnd: numpy .random object
        :return: a realization from the Dirichlet distribution, length K.
        """
        return numpy_rnd.dirichlet(self.a)


class Empirical(RVG):
    def __init__(self, probabilities):
        """
        assuming outcomes = [0, 1, 2, 3, ...]
        E[X] = sum(outcome*prob)
        Var[X] = sum((outcome**2)*prob) - E[X]**2
        """
        RVG.__init__(self)

        self.prob = np.array(probabilities)
        self.nOutcomes = len(self.prob)

        if self.prob.sum() != 1:
            raise ValueError('Probabilities should sum to 1.')
        self.prob = probabilities

    def sample(self, numpy_rnd):
        # this works for both numpy array and list
        # ref:https://stackoverflow.com/questions/4265988/generate-random-numbers-with-a-given-numerical-distribution
        return numpy_rnd.choice(range(self.nOutcomes), size=1, p=self.prob)


class Gamma(RVG):
    def __init__(self, shape, scale):
        """
        E[X] = shape*scale
        Var[X] = shape*scale**2
        """
        RVG.__init__(self)
        self.shape = shape
        self.scale = scale

    def sample(self, numpy_rnd):
        return numpy_rnd.gamma(self.shape, self.scale)


class GammaPoisson(RVG):
    # ref: http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Gammapoisson.pdf
    # in this article shape is beta, scale is alpha, change to Wiki version below
    # with shape-alpha, scale-theta
    def __init__(self, shape, scale):
        """
        E[X] = shape*scale
        Var[X] = shape*scale + shape*(scale**2)
        """
        RVG.__init__(self)
        self.shape = shape
        self.scale = scale

    def sample(self, numpy_rnd):
        sample_rate = Gamma(self.shape, self.scale).sample(numpy_rnd)
        sample_poisson = Poisson(sample_rate)
        return sample_poisson.sample(numpy_rnd)


class Geometric(RVG):
    def __init__(self, p):
        """
        E[X] = 1/p
        Var[X] = (1-p)/p**2
        """
        RVG.__init__(self)
        self.p = p

    def sample(self, numpy_rnd):
        return numpy_rnd.geometric(self.p)


class JohnsonSb(RVG):
    def __init__(self, a, b, loc, scale):
        """
        The moments of the Johnson SB distribution do not have a simple expression.
        E[X] = theoretical value give by SciPy johnsonsb.mean(a,b,loc,scale)
        Var[X] = theoretical value give by SciPy johnsonsb.var(a,b,loc,scale)
        """
        RVG.__init__(self)
        self.a = a
        self.b = b
        self.loc = loc
        self.scale = scale

    def sample(self, numpy_rnd):
        return scipy.johnsonsb.rvs(self.a, self.b, self.loc, self.scale, random_state=numpy_rnd)


class JohnsonSu(RVG):
    def __init__(self, a, b, loc, scale):
        """
        The moments of the Johnson SU distribution do not have a simple expression.
        E[X] = theoretical value give by SciPy johnsonsu.mean(a,b,loc,scale)
        Var[X] = theoretical value give by SciPy johnsonsu.var(a,b,loc,scale)
        """
        RVG.__init__(self)
        self.a = a
        self.b = b
        self.loc = loc
        self.scale = scale

    def sample(self, numpy_rnd):
        return scipy.johnsonsu.rvs(self.a, self.b, self.loc, self.scale, random_state=numpy_rnd)


class LogNormal(RVG):
    def __init__(self, mean, sigma):
        """
        E[X] = exp(mean +sigma**2/2)
        Var[X] = [exp(sigma**2-1)]exp(2*mean + sigma**2)
        """
        RVG.__init__(self)
        self.mean = mean
        self.sigma = sigma

    def sample(self, numpy_rnd):
        return numpy_rnd.lognormal(self.mean, self.sigma)


class Multinomial(RVG):
    def __init__(self, N, pvals):
        """
        E[X_i] = N*p_i
        Var[X] = N*p_i(1-p_i)
        """
        RVG.__init__(self)
        self.N = N
        self.pvals = pvals

    def sample(self, numpy_rnd):
        return numpy_rnd.multinomial(self.N, self.pvals)


class NegativeBinomial(RVG):
    def __init__(self, r, p):
        """
        :param r: number of failures until the experiment is stopped
        :param p: success probability in each trial
        E[X] = (n*p)/(1-p)
        Var[X] = (n*p)/((1-p)**2)
        """
        RVG.__init__(self)
        self.n = r
        self.p = 1-p

    def sample(self, numpy_rnd):
        """
        :return: a realization from the NegativeBinomial distribution
        (the number of successes before a specified number of failures, r, occurs.)
        """
        return numpy_rnd.negative_binomial(self.n, self.p)


class Normal(RVG):
    def __init__(self, mean, st_dev):
        """
        E[X] = mean
        Var[X] = st_dev**2
        """
        RVG.__init__(self)
        self.mean = mean
        self.stDev = st_dev

    def sample(self, numpy_rnd):
        return numpy_rnd.normal(self.mean, self.stDev)


class Poisson(RVG):
    def __init__(self, rate):
        """
        E[X] = rate
        Var[X] = rate
        """
        RVG.__init__(self)
        self.rate = rate

    def sample(self, numpy_rnd):
        return numpy_rnd.poisson(self.rate)


class Triangular(RVG):
    def __init__(self, l, mode, u):
        """
        E[X] = (l+mode+u)/3
        Var[X] = (l**2 + mode**2 + u**2 -l*u - l*mode - u*mode)/18
        """
        RVG.__init__(self)
        self.l = l
        self.mode = mode
        self.u = u

    def sample(self, numpy_rnd):
        return numpy_rnd.triangular(self.l, self.mode, self.u)


class Uniform(RVG):
    def __init__(self, l, r):
        """
        E[X] = (l+r)/2
        Var[X] = (r-l)**2/12
        """
        RVG.__init__(self)
        self.l = l
        self.r = r

    def sample(self, numpy_rnd):
        return numpy_rnd.uniform(self.l, self.r)


class UniformDiscrete(RVG):
    def __init__(self, l, r):
        """
        E[X] = (l+r)/2
        Var[X] = ((r-l+1)**2 - 1)/12
        :param l: int
        :param r: int
        """
        RVG.__init__(self)
        self.l = l
        self.r = r

    def sample(self, numpy_rnd):
        return numpy_rnd.randint(low=self.l, high=self.r+1)


class Weibull(RVG):
    def __init__(self, a):
        """
        E[X] = math.gamma(1 + 1/a)
        Var[X] = math.gamma(1 + 2/a) - (math.gamma(1 + 1/a)**2)
        """
        RVG.__init__(self)
        self.a = a

    def sample(self, numpy_rnd):
        return numpy_rnd.weibull(self.a)
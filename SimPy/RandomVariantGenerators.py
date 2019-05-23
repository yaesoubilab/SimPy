import numpy as np
import scipy.stats as scipy
from numpy.random import RandomState
import math

class RNG(RandomState):
    def __init__(self, seed):
        RandomState.__init__(self, seed)

    def sample(self):
        return self.random_sample()


class RVG:
    def __init__(self):
        pass

    def sample(self, rng, arg=None):
        """
        :param rng: an instant of RNG class
        :param rng: optional arguments
        :returns one realization from the defined probability distribution """

        # abstract method to be overridden in derived classes to process an event
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")


class Constant (RVG):
    def __init__(self, value):
        RVG.__init__(self)
        self.value = value

    def sample(self, rng, arg=None):
        return self.value


class Exponential(RVG):
    def __init__(self, scale, loc=0):
        """
        E[X] = scale + loc
        Var[X] = scale**2
        """
        RVG.__init__(self)
        self.scale = scale
        self.loc = loc

    def sample(self, rng, arg=None):
        return scipy.expon.rvs(loc=self.loc, scale=self.scale, random_state=rng)


class Bernoulli(RVG):
    def __init__(self, p):
        """
        E[X] = p
        Var[X] = p(1-p)
        """
        RVG.__init__(self)
        self.p = p

    def sample(self, rng, arg=None):
        sample = 0
        if rng.random_sample() <= self.p:
            sample = 1
        return sample


class Beta(RVG):
    def __init__(self, a, b, loc=0, scale=1):
        """
        E[X] = a/(a + b)*scale + loc
        Var[X] = (scale**2) ab/[(a + b)**2(a + b + 1)]
        min[X] = loc
        max[x] = min[X] + scale
        """
        RVG.__init__(self)
        self.a = a
        self.b = b
        self.scale = scale
        self.loc = loc

    def sample(self, rng, arg=None):
        return scipy.beta.rvs(self.a, self.b, self.loc, self.scale, random_state=rng)


class BetaBinomial(RVG):
    def __init__(self, n, a, b, loc=0, scale=1):
        """
        E[X] = (na/(a+b))*scale + loc
        Var[X] = [(nab(a+b+n))/((a+b)**2(a+b+1))] * scale**2
        """
        RVG.__init__(self)
        self.n = n
        self.a = a
        self.b = b
        self.loc = loc
        self.scale = scale

    def sample(self, rng, arg=None):
        """
        ref: https://blogs.sas.com/content/iml/2017/11/20/simulate-beta-binomial-sas.html
        :return: a realization from the Beta Binomial distribution
        """
        sample_p = rng.beta(self.a, self.b)
        sample = rng.binomial(self.n, sample_p)

        return sample * self.scale + self.loc


class Binomial(RVG):
    def __init__(self, N, p, loc=0):
        """
        E[X] = Np + loc
        Var[X] = Np(1-p)
        """
        RVG.__init__(self)
        self.N = N
        self.p = p
        self.loc = loc

    def sample(self, rng, arg=None):
        return scipy.binom.rvs(self.N, self.p, self.loc, random_state=rng)


class Dirichlet(RVG):
    def __init__(self, a):
        """
        E[Xi] = ai/a0
        Var[Xi] = (ai(a0-ai))/((a0)**2(a0+1)) where a0 = sum of ai's.
        :param a: array or list
        """
        RVG.__init__(self)
        self.a = a

    def sample(self, rng, arg=None):
        """
        :return: (array) a realization from the Dirichlet distribution
        """
        return rng.dirichlet(self.a)


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

        if self.prob.sum() < 0.99999 or self.prob.sum() > 1.00001:
            raise ValueError('Probabilities should sum to 1.')
        self.prob = probabilities

    def sample(self, rng, arg=None):
        """
        :return: (int) from possible outcomes [0, 1, 2, 3, ...]
        """
        # this works for both numpy array and list
        # ref:https://stackoverflow.com/questions/4265988/generate-random-numbers-with-a-given-numerical-distribution
        return rng.choice(range(self.nOutcomes), size=1, p=self.prob)[0]


class Gamma(RVG):
    def __init__(self, a, loc=0, scale=1):
        """
        E[X] = a*scale + loc
        Var[X] = a*scale**2
        """
        RVG.__init__(self)
        self.a = a
        self.loc = loc
        self.scale = scale

    def sample(self, rng, arg=None):
        return scipy.gamma.rvs(self.a, self.loc, self.scale, random_state=rng)


class GammaPoisson(RVG):
    # ref: http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Gammapoisson.pdf
    # in this article shape is beta, scale is alpha, change to Wiki version below
    # with shape-alpha, scale-theta
    def __init__(self, a, gamma_scale, loc=0, scale=1):
        """
        E[X] = (a*gamma_scale)*scale + loc
        Var[X] = [a*gamma_scale + a*(gamma_scale**2)] * scale **2
        """
        RVG.__init__(self)
        self.a = a
        self.gamma_scale = gamma_scale
        self.loc = loc
        self.scale = scale

    def sample(self, rng, arg=None):
        sample_rate = Gamma(a=self.a, scale=self.gamma_scale).sample(rng)
        sample_poisson = Poisson(mu=sample_rate)
        return sample_poisson.sample(rng) * self.scale + self.loc


class Geometric(RVG):
    def __init__(self, p, loc=0):
        """
        E[X] = 1/p+loc
        Var[X] = (1-p)/p**2
        """
        RVG.__init__(self)
        self.p = p
        self.loc = loc

    def sample(self, rng, arg=None):
        return scipy.geom.rvs(self.p, self.loc, random_state=rng)


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

    def sample(self, rng, arg=None):
        return scipy.johnsonsb.rvs(self.a, self.b, self.loc, self.scale, random_state=rng)


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

    def sample(self, rng, arg=None):
        return scipy.johnsonsu.rvs(self.a, self.b, self.loc, self.scale, random_state=rng)


class LogNormal(RVG):
    def __init__(self, s, loc=0, scale=1):
        """
        E[X] = exp(scale + 1/2 * s**2)
        Var[X] = (exp(s**2)-1)*exp(2*loc+s**2)
        """
        RVG.__init__(self)
        self.s = s
        self.loc = loc
        self.scale = scale

    def sample(self, rng, arg=None):
        return scipy.lognorm.rvs(self.s, self.loc, self.scale, random_state=rng)


class Multinomial(RVG):
    def __init__(self, N, pvals):
        """
        E[X_i] = N*p_i
        Var[X] = N*p_i(1-p_i)
        :param N: (int) number of trials
        :param pvals: (array) probabilities of success for each category
        """
        RVG.__init__(self)
        self.N = N
        self.pvals = pvals

    def sample(self, rng, arg=None):
        return rng.multinomial(self.N, self.pvals)


class NegativeBinomial(RVG):
    def __init__(self, n, p, loc=0):
        """
        The probability distribution for number of failure before n successes
        :param n: number of the number of successes
        :param p: p is the probability of a single success
        E[X] = (n*p)/(1-p) + loc
        Var[X] = (n*p)/((1-p)**2)
        """
        RVG.__init__(self)
        self.n = n
        self.p = p
        self.loc = loc

    def sample(self, rng, arg=None):
        """
        :return: a realization from the NegativeBinomial distribution
        (the number of failure before a specified number of successes, n, occurs.)
        """
        return scipy.nbinom.rvs(self.n, self.p, self.loc, random_state=rng)


class NonHomogeneousExponential(RVG):
    def __init__(self, rates, delta_t=1):
        """
        :param rates: (list) of rates over each period (e.g. [1, 2])
        :param delta_t: length of each period
        """

        if rates[-1] == 0:
            raise ValueError('For a non-homogeneous exponential distribution, '
                             'the rate of the last period should be greater than 0.')

        RVG.__init__(self)
        self.rates = rates
        self.deltaT = delta_t

    def sample(self, rng, arg=None):
        """
        :param arg: current time (age)
        :return: a realization from the NonHomogeneousExponential distribution
        """

        if_occurred = False
        if arg is None:
            i = 0
        else:
            i = min(math.floor(arg/self.deltaT), len(self.rates)-1)
        while not if_occurred:
            if self.rates[i] > 0:
                exp = Exponential(scale=1/self.rates[i])
                t = exp.sample(rng)
            else:
                t = float('inf')

            if i == len(self.rates)-1 or t < self.deltaT:
                if_occurred = True
                return t + i*self.deltaT
            else:
                i += 1


class Normal(RVG):
    def __init__(self, loc=0, scale=1):
        """
        E[X] = loc
        Var[X] = scale**2
        """
        RVG.__init__(self)
        self.loc = loc
        self.scale = scale

    def sample(self, rng, arg=None):
        return scipy.norm.rvs(self.loc, self.scale, random_state=rng)


class Poisson(RVG):
    def __init__(self, mu, loc=0):
        """
        E[X] = mu + loc
        Var[X] = mu
        """
        RVG.__init__(self)
        self.mu = mu
        self.loc = loc

    def sample(self, rng, arg=None):
        return scipy.poisson.rvs(self.mu, self.loc, random_state=rng)


class Triangular(RVG):
    def __init__(self, c, loc=0, scale=1):
        """
        l = loc, u = loc+scale, mode = loc + c*scale
        E[X] = (l+mode+u)/3
        Var[X] = (l**2 + mode**2 + u**2 -l*u - l*mode - u*mode)/18
        """
        RVG.__init__(self)
        self.c = c
        self.loc = loc
        self.scale = scale

    def sample(self, rng, arg=None):
        return scipy.triang.rvs(self.c, self.loc, self.scale, random_state=rng)


class Uniform(RVG):
    def __init__(self, loc=0, scale=1):
        """
        setting l = loc, u = loc + scale
        E[X] = (l+u)/2
        Var[X] = (u-l)**2/12
        """
        RVG.__init__(self)
        self.loc = loc
        self.scale = scale

    def sample(self, rng, arg=None):
        return scipy.uniform.rvs(self.loc, self.scale, random_state=rng)


class UniformDiscrete(RVG):
    def __init__(self, l, u):
        """
        E[X] = (l+u)/2
        Var[X] = ((r-u+1)**2 - 1)/12
        :param l: (int) inclusive lower bound
        :param u: (int) inclusive upper bound
        """
        RVG.__init__(self)
        self.l = l
        self.u = u

    def sample(self, rng, arg=None):
        return rng.randint(low=self.l, high=self.u + 1)


class Weibull(RVG):
    def __init__(self, a, loc=0, scale=1):
        """
        E[X] = math.gamma(1 + 1/a) * scale + loc
        Var[X] = [math.gamma(1 + 2/a) - (math.gamma(1 + 1/a)**2)] * scale**2
        """
        RVG.__init__(self)
        self.a = a
        self.loc = loc
        self.scale = scale

    def sample(self, rng, arg=None):
        return scipy.weibull_min.rvs(self.a, self.loc, self.scale, random_state=rng)
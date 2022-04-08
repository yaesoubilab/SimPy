import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats as stat
from numpy.random import RandomState
from scipy.optimize import fmin_slsqp

warnings.filterwarnings("ignore")


def AIC(k, log_likelihood):
    """ :returns Akaike information criterion"""
    return 2 * k - 2 * log_likelihood


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
        :param arg: optional arguments
        :returns one realization from the defined probability distribution """

        # abstract method to be overridden in derived classes to process an event
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")

    def get_mean_st_dev(self):
        """ :returns: (tuple) of mean and standard deviation """
        pass

    def get_percentile_interval(self, alpha=0.05):
        """ :returns: [l, u], where l and u are the lower and upper critical values
                of the distribution"""

    @staticmethod
    def _get_percentile_interval(alpha, dist, params):
        """
        :param alpha: confidence level
        :param dist: a distribution with .ppf method
        :param params: parameters of the distribution
        :return:
        """
        return [dist.ppf(alpha / 2, *params),
                dist.ppf(1 - alpha / 2, *params)]


class Constant (RVG):
    def __init__(self, value):
        RVG.__init__(self)
        self.value = value

    def sample(self, rng, arg=None):
        return self.value


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
        return rng.beta(self.a, self.b) * self.scale + self.loc

    def get_percentile_interval(self, alpha=0.05):

        return self._get_percentile_interval(
            alpha=alpha, dist=stat.beta,
            params=[self.a, self.b, self.loc, self.scale])

    @staticmethod
    def fit_mm(mean, st_dev, minimum=0, maximum=1):
        """
        :param mean: sample mean
        :param st_dev: sample standard deviation
        :param minimum: fixed minimum
        :param maximum: fixed maximum
        :return: dictionary with keys "a", "b", "loc" and "scale"
        """

        if not (minimum <= mean <= maximum):
            raise ValueError('Mean should be between minimum and maximum.')

        # shift the distribution by loc and scale
        mean = (mean - minimum) * 1.0 / (maximum - minimum)
        st_dev = st_dev * 1.0 / (maximum - minimum)

        a_plus_b = mean * (1 - mean) / pow(st_dev, 2) - 1
        a = mean * a_plus_b

        return {"a": a, "b": a_plus_b - a, "loc": minimum, "scale": maximum - minimum}

    @staticmethod
    def fit_ml(data, minimum=None, maximum=None):
        """
        :param data: (numpy.array) observations
        :param minimum: minimum of data (calculated from data if not provided)
        :param maximum: maximum of data (calculated from data if not provided)
        :returns: dictionary with keys "a", "b", "loc", "scale", and "AIC"
        """

        # transform data into [0,1]
        L = np.min(data) if minimum is None else minimum
        U = np.max(data) if maximum is None else maximum
        data = (data - L) / (U - L)

        # estimate the parameters
        a, b, loc, scale = stat.beta.fit(data, floc=0)

        # calculate AIC
        aic = AIC(k=3,
                  log_likelihood=np.sum(stat.beta.logpdf(data, a, b, loc, scale)))

        # report results in the form of a dictionary
        return {"a": a, "b": b, "loc": L, "scale": U - L, "AIC": aic}


class BetaBinomial(RVG):
    def __init__(self, n, a, b, loc=0):
        """
        E[X] = na/(a+b) + loc
        Var[X] = (nab(a+b+n))/((a+b)**2(a+b+1))
        """
        RVG.__init__(self)
        self.n = n
        self.a = a
        self.b = b
        self.loc = loc

    def sample(self, rng, arg=None):
        """
        ref: https://blogs.sas.com/content/iml/2017/11/20/simulate-beta-binomial-sas.html
        :return: a realization from the Beta Binomial distribution
        """
        sample_p = rng.beta(self.a, self.b)
        sample_n = rng.binomial(self.n, sample_p)

        return sample_n + self.loc

    def get_percentile_interval(self, alpha=0.05):

        return self._get_percentile_interval(
            alpha=alpha, dist=stat.betabinom,
            params=[self.n, self.a, self.b, self.loc])

    @staticmethod
    def fit_mm(mean, st_dev, n, fixed_location=0):
        """
        # ref: https://en.wikipedia.org/wiki/Beta-binomial_distribution
        :param mean: sample mean of an observation set
        :param st_dev: sample standard deviation of an observation set
        :param n: the number of trials in the Binomial distribution
        :param fixed_location: location, 0 by default
        :return: dictionary with keys "a", "b", "n", "loc", and "scale"
        """
        mean = 1.0 * (mean - fixed_location)
        variance = st_dev ** 2.0
        m2 = variance + mean ** 2  # second moment

        a1 = n * mean - m2
        a2 = n * (m2 / mean - mean - 1) + mean
        a = a1 / a2
        b1 = (n - mean) * (n - m2 / mean)
        b2 = a2
        b = b1 / b2

        return {"a": a, "b": b, "n": n, "loc": fixed_location}

    @staticmethod
    def ln_pmf(a, b, n, k):
        part_1 = sp.special.comb(n, k)
        part_2 = sp.special.betaln(k + a, n - k + b)
        part_3 = sp.special.betaln(a, b)
        result = (np.log(part_1) + part_2) - part_3
        return result

    # define log_likelihood function: sum of log(pmf) for each data point
    @staticmethod
    def ln_likelihood(a, b, n, data):
        n = int(np.round(n, 0))
        result = 0
        for i in range(len(data)):
            result += BetaBinomial.ln_pmf(a=a, b=b, n=n, k=data[i])
        return result

    @staticmethod
    def fit_ml(data, fixed_location=0):
        """
        :param data: (numpy.array) observations
        :param fixed_location: fixed location
        :returns: dictionary with keys "a", "b", "n" and "AIC"
        """

        data = 1.0 * (data - fixed_location)

        def neg_ln_l(a_b_n):
            return -BetaBinomial.ln_likelihood(a=a_b_n[0], b=a_b_n[1], n=a_b_n[2], data=data)

        # estimate the parameters by minimize negative log-likelihood
        # initialize parameters
        a_b_n_0 = [1, 1, np.max(data)]
        # call Scipy optimizer to minimize the target function
        # with bounds for a [0, 10], b [0, 10] and n [0,100+max(data)]
        fitted_a_b_n, value, iter, imode, smode \
            = fmin_slsqp(neg_ln_l, a_b_n_0,
                         bounds=[(0.0, 10000.0), (0.0, 10000.0), (0, np.max(data) + 100)],
                         disp=False, full_output=True)

        # calculate AIC
        aic = AIC(
            k=3,
            log_likelihood=BetaBinomial.ln_likelihood(
                a=fitted_a_b_n[0], b=fitted_a_b_n[1], n=fitted_a_b_n[2], data=data)
        )

        # report results in the form of a dictionary
        return {"a": fitted_a_b_n[0], "b": fitted_a_b_n[1], "n": fitted_a_b_n[2], "loc": fixed_location, "AIC": aic}


class Binomial(RVG):
    def __init__(self, n, p, loc=0):
        """
        E[X] = np + loc
        Var[X] = np(1-p)
        """
        RVG.__init__(self)
        self.n = n
        self.p = p
        self.loc = loc

    def sample(self, rng, arg=None):
        return rng.binomial(self.n, self.p) + self.loc

    def get_percentile_interval(self, alpha=0.05):

        return self._get_percentile_interval(
            alpha=alpha, dist=stat.binom,
            params=[self.n, self.p, self.loc])

    @staticmethod
    def fit_mm(mean, st_dev, fixed_location=0):
        """
        :param mean: sample mean
        :param st_dev: sample standard deviation
        :param fixed_location: fixed location, 0 by default
        :return: dictionary with keys "p", "n" and "loc"
        """
        mean = mean - fixed_location
        p = 1.0 - (st_dev ** 2) / mean
        n = mean / p

        return {"n": n, "p": p, "loc": fixed_location}

    @staticmethod
    def fit_ml(data, fixed_location=0):
        """
        :param data: (numpy.array) observations
        :param fixed_location: fixed location
        :returns: dictionary with keys "p", "n", "loc" and "AIC"
        """

        # the MLE of p is x/n
        # if we have N data point with Xi~Bin(n,p), then sum(Xi)~Bin(n*N,p), p_hat = sum(xi)/(n*N)
        # # https://onlinecourses.science.psu.edu/stat504/node/28

        data = data - fixed_location

        mean = np.mean(data)
        st_dev = np.std(data)
        p = 1.0 - (st_dev ** 2) / mean
        n = mean / p

        # calculate AIC
        aic = AIC(
            k=1,
            log_likelihood=np.sum(stat.binom.logpmf(data, n, p))
        )

        # report results in the form of a dictionary
        return {"n": n, "p": p, "loc": fixed_location, "AIC": aic}


class Dirichlet(RVG):
    def __init__(self, a, if_ignore_0s=False):
        """
        E[Xi] = ai/a0 where a0 = sum of ai's.
        Var[Xi] = (ai(a0-ai))/((a0)**2(a0+1)) where a0 = sum of ai's.
        :param a: array or list
        :param if_ignore_0s: (bool) numpy requires all elements of 'a' to be >0. Setting this
            parameter to True allows 'a' to contain 0s.
        """
        RVG.__init__(self)
        self.a = a
        self.ifIgnore0s = if_ignore_0s
        self.nonZeroA = []
        self.idxOfNonZeroA = []
        for i, value in enumerate(a):
            if value > 0:
                self.nonZeroA.append(value)
                self.idxOfNonZeroA.append(i)

    def sample(self, rng, arg=None):
        """
        :return: (array) a realization from the Dirichlet distribution
        """
        if self.ifIgnore0s:
            sample = rng.dirichlet(self.nonZeroA)
            result = [0]*len(self.a)
            for i, idx in enumerate(self.idxOfNonZeroA):
                result[idx] = sample[i]
            return result
        else:
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

    @staticmethod
    def fit_mm(data, bin_size=1):
        """
        :param data: (numpy.array) observations
        :param bin_size: float, the width of histogram's bins
        :returns: dictionary keys of "bins" and "freq"
        """

        l = np.min(data)
        u = np.max(data) + bin_size
        n_bins = math.floor((u - l)/bin_size) + 1
        bins = np.linspace(l, u, n_bins)

        result = plt.hist(data, bins=bins)

        bins = result[1]  # bins are in the form of [a,b)
        freq = result[0] * 1.0 / len(data)

        return {"bins": bins, "freq": freq}


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
        return rng.exponential(scale=self.scale) + self.loc

    def get_percentile_interval(self, alpha=0.05):

        return self._get_percentile_interval(
            alpha=alpha, dist=stat.expon,
            params=[self.loc, self.scale])

    @staticmethod
    def fit_mm(mean, fixed_location=0):
        """
        :param mean: sample mean
        :param fixed_location: minimum of the exponential distribution, set to 0 by default
        :return: dictionary with keys "loc" and "scale"
        """
        mean = mean - fixed_location
        scale = mean

        return {"scale": scale, "loc": fixed_location}

    @ staticmethod
    def fit_ml(data, fixed_location=0):
        """
        :param data: (numpy.array) observations
        :param fixed_location: fixed location
        :returns: dictionary with keys "loc", "scale", and "AIC"
        """
        # estimate the parameters of exponential
        loc, scale = stat.expon.fit(data, floc=fixed_location)

        # calculate AIC
        aic = AIC(
            k=1,
            log_likelihood=np.sum(stat.expon.logpdf(data, loc, scale)))

        # report results in the form of a dictionary
        return {"scale": scale, "loc": loc, "AIC": aic}


class Gamma(RVG):
    def __init__(self, a, scale=1, loc=0):
        """
        E[X] = a*scale + loc
        Var[X] = a*scale**2
        """
        RVG.__init__(self)
        self.a = a
        self.scale = scale
        self.loc = loc

    def sample(self, rng, arg=None):
        return rng.gamma(self.a, self.scale) + self.loc

    def get_mean_st_dev(self):
        return self.a*self.scale + self.loc, math.sqrt(self.a*self.scale**2)

    def get_percentile_interval(self, alpha=0.05):

        return self._get_percentile_interval(alpha=alpha, dist=stat.gamma,
                                             params=[self.a, self.loc, self.scale])

    @staticmethod
    def fit_mm(mean, st_dev, fixed_location=0):
        """
        :param mean: sample mean of an observation set
        :param st_dev: sample standard deviation of an observation set
        :param fixed_location: location, 0 by default
        :return: dictionary with keys "a", "loc" and "scale"
        """
        mean = mean - fixed_location

        shape = (mean / st_dev) ** 2
        scale = (st_dev ** 2) / mean

        # report results in the form of a dictionary
        return {"a": shape, "scale": scale, "loc": fixed_location}

    @staticmethod
    def fit_ml(data, fixed_location=0):
        """
        :param data: (numpy.array) observations
        :param fixed_location: fixed location
        :returns: dictionary with keys "a", "scale", "loc", and "AIC"
        """

        # estimate the parameters of gamma
        # alpha = a, beta = 1/scale
        a, loc, scale = stat.gamma.fit(data, floc=fixed_location)

        # calculate AIC
        aic = AIC(
            k=2,
            log_likelihood=np.sum(stat.gamma.logpdf(data, a, loc, scale)))

        # report results in the form of a dictionary
        return {"a": a, "scale": scale, "loc": loc, "AIC": aic}


class GammaPoisson(RVG):
    #
    def __init__(self, a, gamma_scale, loc=0):
        """
        E[X] = (a*gamma_scale) + loc
        Var[X] = a*gamma_scale + a*(gamma_scale**2)
        """
        RVG.__init__(self)
        self.a = a
        self.gamma_scale = gamma_scale
        self.loc = loc

    def sample(self, rng, arg=None):
        sample_rate = Gamma(a=self.a, scale=self.gamma_scale).sample(rng)
        sample_poisson = Poisson(mu=sample_rate)
        return sample_poisson.sample(rng) + self.loc

    def get_percentile_interval(self, alpha=0.05):

        return [self.ppf(q=alpha/2), self.ppf(q=1-alpha/2)]

    def pmf(self, k):
        # https: // en.wikipedia.org / wiki / Negative_binomial_distribution  # Gamma%E2%80%93Poisson_mixture
        # with r= a and p = scale/(scale + 1)

        if type(k) == int:
            k = k - self.loc
            p = self.gamma_scale / (self.gamma_scale + 1)
            part1 = 1.0 * sp.special.gamma(self.a + k) / (sp.special.gamma(self.a) * sp.special.factorial(k))
            part2 = (p ** k) * ((1 - p) ** self.a)

            return part1 * part2
        else:
            result = []
            for this_k in k:
                result.append(self.pmf(this_k))
            return result

    def ppf(self, q):
        """ :returns: percentage point function (inverse of cdf)"""

        cum = 0
        k = self.loc

        while True:
            cum += self.pmf(k)
            k += 1
            if cum > q:
                break
        return k - 1

    @staticmethod
    def ln_pmf(a, gamma_scale, loc, k):
        # https: // en.wikipedia.org / wiki / Negative_binomial_distribution  # Gamma%E2%80%93Poisson_mixture
        # with r= a and p = scale/(scale + 1)

        k = k - loc
        p = gamma_scale / (gamma_scale + 1)
        part1 = np.log(sp.special.gamma(a + k)) \
                - np.log(sp.special.gamma(a)) \
                - np.log(sp.special.factorial(k))
        part2 = k * np.log(p) + a * np.log(1 - p)

        return part1 + part2

    # define log_likelihood function: sum of log(pmf) for each data point
    @staticmethod
    def ln_likelihood(a, scale, loc, data):
        result = 0
        for k in data:
            v = GammaPoisson.ln_pmf(a=a, gamma_scale=scale, loc=loc, k=k)
            result += v
        return result

    @staticmethod
    def fit_mm(mean, st_dev, fixed_location=0):
        """
        :param mean: sample mean
        :param st_dev: sample standard deviation
        :param fixed_location: location, 0 by default
        :return: dictionary with keys "a", "gamma_scale", "loc" and "scale"
        """
        # ref: http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Gammapoisson.pdf
        # scale = 1/beta

        mean = mean - fixed_location
        variance = st_dev**2.0

        gamma_scale = (variance-mean)/mean
        a = mean/gamma_scale

        return {"a": a, "gamma_scale": gamma_scale, "loc": fixed_location}

    @staticmethod
    def fit_ml(data, fixed_location=0):
        """
        :param data: (numpy.array) observations
        :param fixed_location: location
        :return: dictionary with keys "a", "gamma_scale", "loc", and "AIC"
        """

        data = data - fixed_location

        # define negative log-likelihood, the target function to minimize
        def neg_ln_l(a_scale):
            return -GammaPoisson.ln_likelihood(a=a_scale[0], scale=a_scale[1], loc=0, data=data)

        # estimate the parameters by minimize negative log-likelihood
        # initialize parameters
        a_scale = [1, 1]
        # call Scipy optimizer to minimize the target function
        # with bounds for p [0,1] and r [0,10]
        fitted_a_scale, value, iter, imode, smode = fmin_slsqp(neg_ln_l, a_scale,
                                                               #bounds=[(0.0, 10.0), (0, 1)],
                                                               disp=False, full_output=True)

        # calculate AIC (note that the data has already been shifted by loc)
        aic = AIC(
            k=2,
            log_likelihood=GammaPoisson.ln_likelihood(
                a=fitted_a_scale[0], scale=fitted_a_scale[1], loc=0, data=data)
        )
        return {"a": fitted_a_scale[0], "gamma_scale": fitted_a_scale[1], "loc": fixed_location, "AIC": aic}


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
        return rng.geometric(self.p) + self.loc

    def get_percentile_interval(self, alpha=0.05):

        return self._get_percentile_interval(
            alpha=alpha, dist=stat.geom,
            params=[self.p, self.loc])

    @staticmethod
    def fit_mm(mean, fixed_location=0):
        """
        :param mean: sample mean
        :param fixed_location: location, 0 by default
        :return: dictionary with keys "p", "loc"
        """
        mean = mean - fixed_location
        p = 1.0/mean

        return {"p": p, "loc": fixed_location}

    @staticmethod
    def fit_ml(data, fixed_location=0):
        """
        :param data: (numpy.array) observations
        :param fixed_location: location, 0 by default
        :return: dictionary with keys "p", "loc"
        """

        # https://www.projectrhea.org/rhea/index.php/MLE_Examples:_Exponential_and_Geometric_Distributions_Old_Kiwi
        data = data - fixed_location
        p = len(data) * 1.0 / np.sum(data)

        # calculate AIC
        aic = AIC(
            k=1,
            log_likelihood=np.sum(stat.geom.logpmf(data, p)))

        # report results in the form of a dictionary
        return {"p": p, "loc": fixed_location, "AIC": aic}


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
        return stat.johnsonsb.rvs(self.a, self.b, self.loc, self.scale, random_state=rng)

    def get_percentile_interval(self, alpha=0.05):

        return self._get_percentile_interval(
            alpha=alpha, dist=stat.johnsonsb,
            params=[self.a, self.b, self.loc, self.scale])

    @staticmethod
    def fit_ml(data, fixed_location=0):
        """
        :param data: (numpy.array) observations
        :param fixed_location: location, 0 by default
        :returns: dictionary with keys "a", "b", "loc", "scale", and "AIC"
        """

        # estimate the parameters
        a, b, loc, scale = stat.johnsonsb.fit(data, floc=fixed_location)

        # calculate AIC
        aic = AIC(
            k=3,
            log_likelihood=np.sum(stat.johnsonsb.logpdf(data, a, b, loc, scale)))

        # report results in the form of a dictionary
        return {"a": a, "b": b, "loc": loc, "scale": scale, "AIC": aic}


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
        return stat.johnsonsu.rvs(self.a, self.b, self.loc, self.scale, random_state=rng)

    def get_percentile_interval(self, alpha=0.05):

        return self._get_percentile_interval(
            alpha=alpha, dist=stat.johnsonsu,
            params=[self.a, self.b, self.loc, self.scale])

    @staticmethod
    def fit_ml(data, fixed_location=0):
        """
        :param data: (numpy.array) observations
        :param fixed_location: location, 0 by default
        :returns: dictionary with keys "a", "b", "loc", "scale", and "AIC"
        """

        # estimate the parameters
        a, b, loc, scale = stat.johnsonsu.fit(data, floc=fixed_location)

        # calculate AIC
        aic = AIC(
            k=3,
            log_likelihood=np.sum(stat.johnsonsu.logpdf(data, a, b, loc, scale)))

        # report results in the form of a dictionary
        return {"a": a, "b": b, "loc": loc, "scale": scale, "AIC": aic}


class LogNormal(RVG):
    def __init__(self, mu, sigma, loc=0):
        """
        E[X] = exp(mu + 1/2 * sigma**2) + loc
        Var[X] = (exp(sigma**2)-1)*exp(2*mu + s**2)
        """
        RVG.__init__(self)
        self.mu = mu
        self.sigma = sigma
        self.loc = loc

    def sample(self, rng, arg=None):
        return rng.lognormal(mean=self.mu, sigma=self.sigma) + self.loc

    def get_percentile_interval(self, alpha=0.05):

        return self._get_percentile_interval(
            alpha=alpha, dist=stat.lognorm,
            params=[self.sigma, self.loc, np.exp(self.mu)])

    def pdf(self, x):
        return np.exp(-(np.log(x) - self.mu)**2 / (2 * self.sigma**2)) / (x * self.sigma * np.sqrt(2 * np.pi))

    def ppf(self, q):
        return stat.lognorm.ppf(q, s=self.sigma, loc=self.loc, scale=np.exp(self.mu))

    @staticmethod
    def fit_mm(mean, st_dev, fixed_location=0):
        """
        :param mean: sample mean of an observation set
        :param st_dev: sample standard deviation of an observation set
        :param fixed_location: location, 0 by default
        :return: dictionary with keys "mu", "sigma" and "loc"
        """

        mean = mean-fixed_location

        mu = np.log(
            mean**2 / np.sqrt(st_dev**2 + mean**2)
        )
        sigma = np.sqrt(
            np.log(1 + st_dev**2 / mean**2)
        )

        return {"mu": mu, "sigma": sigma, "loc": fixed_location}

    @staticmethod
    def fit_ml(data, fixed_location=0):
        """
        :param data: (numpy.array) observations
        :param fixed_location: location, 0 by default
        :returns: dictionary with keys "mu", "sigma" and "loc"
        """

        # estimate the parameters
        s, loc, scale = stat.lognorm.fit(data, floc=fixed_location)
        sigma = s
        mu = np.log(scale)

        # calculate AIC
        aic = AIC(
            k=2,
            log_likelihood=np.sum(stat.lognorm.logpdf(data, s, loc, scale)))

        # report results in the form of a dictionary
        return {"mu": mu, "sigma": sigma, "loc": loc, "AIC": aic}


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
        return rng.negative_binomial(self.n, self.p) + self.loc

    def get_percentile_interval(self, alpha=0.05):

        return self._get_percentile_interval(
            alpha=alpha, dist=stat.nbinom,
            params=[self.n, self.p, self.loc])

    @staticmethod
    def ln_likelihood(n, p, data):
        result = 0
        for d in data:
            result += stat.nbinom.logpmf(d, n, p)
        return result

    @staticmethod
    def fit_mm(mean, st_dev, fixed_location=0):
        """
        :param mean: sample mean of an observation set
        :param st_dev: sample standard deviation of an observation set
        :param fixed_location: location, 0 by default
        :returns: dictionary with keys "n", "p" and "loc"
        """
        # in Scipy, n is the number of successes, p is the probability of a single success.
        # in Wiki, r is the number of failure, p is success probability
        # to match the moments, define r = n is the number of successes, 1-p is the failure probability
        mean = mean - fixed_location

        p = mean/st_dev**2.0
        n = mean*p/(1-p)

        return {"n": n, "p": p, "loc": fixed_location}

    @staticmethod
    def fit_ml(data, fixed_location=0):
        """
        :param data: (numpy.array) observations
        :param fixed_location: location, 0 by default
        :returns: dictionary with keys "n", "p" and "loc"
        """

        ## not working
        return

        data = data - fixed_location

        def neg_ln_l(n_p):
            return -NegativeBinomial.ln_likelihood(n=n_p[0], p=n_p[1], data=data)

        n_p_loc = NegativeBinomial.fit_mm(mean=np.mean(data), st_dev=np.std(data), fixed_location=0)
        n_p = [n_p_loc['n'], n_p_loc['p']]
        M = 2*np.max(data)

        # call Scipy optimizer to minimize the target function
        # with bounds for p [0,1] and n [0,M]
        fitted_n_p, value, iter, imode, smode = fmin_slsqp(neg_ln_l, n_p,
                                                           #bounds=[(0.0, M), (0, 1)],
                                                           disp=False, full_output=True)

        # calculate AIC
        aic = AIC(
            k=2,
            log_likelihood=np.sum(stat.nbinom.logpmf(data, fitted_n_p[0], fitted_n_p[1], 0)))

        # report results in the form of a dictionary
        return {"n": fitted_n_p[0], "p": fitted_n_p[1], "loc": fixed_location, "AIC": aic}


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
            arg = 0
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
                return max(0, t + i*self.deltaT - arg)
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
        return rng.normal(self.loc, self.scale)

    def get_percentile_interval(self, alpha=0.05):

        return self._get_percentile_interval(
            alpha=alpha, dist=stat.norm,
            params=[self.loc, self.scale])

    @staticmethod
    def fit_mm(mean, st_dev):
        """
        :param mean: sample mean of an observation set
        :param st_dev: sample standard deviation of an observation set
        :return: dictionary with keys "loc" and "scale"
        """

        return {"loc": mean, "scale": st_dev}

    @staticmethod
    def fit_ml(data):
        """
        :param data: (numpy.array) observations
        :return: dictionary with keys "loc", "scale", and "AIC"
        """

        mean = np.average(data)
        st_dev = np.std(data)
        # calculate AIC
        aic = AIC(
            k=2,
            log_likelihood=np.sum(stat.norm.logpdf(data, mean, st_dev)))

        return {"loc": mean, "scale": st_dev, "AIC": aic}


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
        return rng.poisson(self.mu) + self.loc

    def get_percentile_interval(self, alpha=0.05):

        return self._get_percentile_interval(
            alpha=alpha, dist=stat.poisson,
            params=[self.mu, self.loc])

    @staticmethod
    def fit_mm(mean, fixed_location=0):
        """
        :param mean: sample mean of an observation set
        :param fixed_location: location, 0 by default
        :returns: dictionary with keys "mu" and "loc"
        """

        mu = mean - fixed_location

        return {"mu": mu, "loc": fixed_location}

    @staticmethod
    def fit_ml(data, fixed_location=0):
        """
        :param data: (numpy.array) observations
        :param fixed_location: location, 0 by default
        :returns: dictionary with keys "mu" and "loc"
        """

        # fit poisson distribution: the MLE of lambda is the sample mean
        # https://en.wikipedia.org/wiki/Poisson_distribution#Maximum_likelihood
        data = data - fixed_location
        mu = data.mean()

        # calculate AIC
        aic = AIC(
            k=1,
            log_likelihood=np.sum(stat.poisson.logpmf(data, mu)))

        # report results in the form of a dictionary
        return {"mu": mu, "loc": fixed_location, "AIC": aic}


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
        return stat.triang.rvs(self.c, self.loc, self.scale, random_state=rng)

    def get_percentile_interval(self, alpha=0.05):

        return self._get_percentile_interval(
            alpha=alpha, dist=stat.triang,
            params=[self.c, self.loc, self.scale])

    @staticmethod
    def fit_ml(data, fixed_location=0):
        """
        :param data: (numpy.array) observations
        :param fixed_location: location, 0 by default
        :return: dictionary with keys "c", "loc", "scale", and "AIC"
        """

        c, loc, scale = stat.triang.fit(data, floc=fixed_location)

        # calculate AIC
        aic = AIC(
            k=2,
            log_likelihood=np.sum(stat.triang.logpdf(data, c, loc, scale))
        )

        # report results in the form of a dictionary
        return {"c": c, "loc": loc, "scale": scale, "AIC": aic}


class Uniform(RVG):
    def __init__(self, scale=1, loc=0):
        """
        setting l = loc, u = loc + scale
        E[X] = (l+u)/2
        Var[X] = (u-l)**2/12
        """
        RVG.__init__(self)
        self.scale = scale
        self.loc = loc

    def sample(self, rng, arg=None):
        return rng.uniform(low=self.loc, high=self.loc+self.scale)

    def get_percentile_interval(self, alpha=0.05):

        return self._get_percentile_interval(
            alpha=alpha, dist=stat.uniform,
            params=[self.loc, self.scale])

    @staticmethod
    def fit_mm(mean, st_dev):
        """
        :param mean: sample mean of an observation set
        :param st_dev: sample standard deviation of an observation set
        :return: dictionary with keys "loc" and "scale"
        """

        b = 0.5*(2*mean + np.sqrt(12)*st_dev)
        a = 2.0*mean - b

        loc = a
        scale = b-a

        return {"loc": loc, "scale": scale,}

    @staticmethod
    def fit_ml(data):
        """
        :param data: (numpy.array) observations
        :return: dictionary with keys "loc", "scale", and "AIC"
        """

        # estimate the parameters
        loc, scale = stat.uniform.fit(data)

        # calculate AIC
        aic = AIC(
            k=2,
            log_likelihood=np.sum(stat.uniform.logpdf(data, loc, scale))
        )

        # report results in the form of a dictionary
        return {"loc": loc, "scale": scale, "AIC": aic}


class UniformDiscrete(RVG):
    def __init__(self, l, u):
        """
        E[X] = (l+u)/2
        Var[X] = ((u-l+1)**2 - 1)/12
        :param l: (int) inclusive lower bound
        :param u: (int) inclusive upper bound
        """
        RVG.__init__(self)
        self.l = l
        self.u = u

    def sample(self, rng, arg=None):
        return rng.randint(low=self.l, high=self.u + 1)

    def get_percentile_interval(self, alpha=0.05):

        return self._get_percentile_interval(
            alpha=alpha, dist=stat.randint,
            params=[self.l, self.u])

    @staticmethod
    def fit_mm(mean, st_dev):
        """
        :param mean: sample mean of an observation set
        :param st_dev: sample standard deviation of an observation set
        :return: dictionary with keys "l" and "u"
        """
        variance = st_dev**2
        b = (np.sqrt(12.0*variance + 1) + 2.0*mean-1)*0.5
        a = (-np.sqrt(12.0*variance + 1) + 2.0*mean+1)*0.5

        return {"l": a, "u": b}

    @staticmethod
    def fit_ml(data):
        """
        :param data: (numpy.array) observations
        :return: dictionary with keys "l" and "u"
        """
        # estimate the parameters
        # as likelihood = 1/(high-low)^n, so the smaller the range, the higher the likelihood
        # the MLE is
        low = np.min(data)
        high = np.max(data)

        # calculate AIC
        aic = AIC(
            k=2,
            log_likelihood=np.sum(stat.randint.logpmf(data, low, high))
        )

        # report results in the form of a dictionary
        return {"l": low, "u": high, "AIC": aic}


class Weibull(RVG):
    def __init__(self, a, scale=1, loc=0):
        """
        E[X] = math.gamma(1 + 1/a) * scale + loc
        Var[X] = [math.gamma(1 + 2/a) - (math.gamma(1 + 1/a)**2)] * scale**2
        """
        RVG.__init__(self)
        self.a = a
        self.scale = scale
        self.loc = loc

    def sample(self, rng, arg=None):
        return rng.weibull(self.a) * self.scale + self.loc

    def get_percentile_interval(self, alpha=0.05):

        return self._get_percentile_interval(
            alpha=alpha, dist=stat.weibull_min,
            params=[self.a, self.loc, self.scale])

    @staticmethod
    def fit_mm(mean, st_dev, fixed_location=0):
        """
        :param mean: sample mean of an observation set
        :param st_dev: sample standard deviation of an observation set
        :param fixed_location: location, 0 by default
        :returns: dictionary with keys "c", "loc" and "scale"
        """
        mean = mean - fixed_location

        c = (st_dev*1.0/mean)**(-1.086)
        scale = mean/sp.special.gamma(1 + 1.0/c)

        return {"c": c, "scale": scale, "loc": fixed_location}

    @staticmethod
    def fit_ml(data, fixed_location=0):
        """
        :param data: (numpy.array) observations
        :param fixed_location: location, 0 by default
        :returns: dictionary with keys "c", "loc" and "scale"
        """

        # estimate the parameters of weibull
        c, loc, scale = stat.weibull_min.fit(data, floc=fixed_location)

        # calculate AIC
        aic = AIC(
            k=2,
            log_likelihood=np.sum(stat.weibull_min.logpdf(data, c, loc, scale))
        )

        # report results in the form of a dictionary
        return {"c": c, "loc": loc, "scale": scale, "AIC": aic}
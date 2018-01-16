import sys
import numpy as numpy
import scipy.stats as stat
import math
import SupportFunctions as Support


class Statistics(object):
    def __init__(self, name):
        self.name = name        # name of this statistics
        self._n = 0              # number of data points
        self._mean = 0           # sample mean
        self._stDev = 0          # sample standard deviation
        self._max = -sys.float_info.max  # maximum
        self._min = sys.float_info.max   # minimum

    def get_mean(self):
        """ abstract method to be overridden in derived classes
        :returns mean (to be calculated in the subclass) """
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")

    def get_stdev(self):
        """ abstract method to be overridden in derived classes
        :returns standard deviation (to be calculated in the subclass) """
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")

    def get_min(self):
        """ abstract method to be overridden in derived classes
        :returns minimum (to be calculated in the subclass) """
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")

    def get_max(self):
        """ abstract method to be overridden in derived classes
        :returns maximum (to be calculated in the subclass) """
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")

    def get_percentile(self, q):
        """ abstract method to be overridden in derived classes
        :param q: percentile to compute (q in range [0, 100])
        :returns percentile (to be calculated in the subclass) """
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")

    def get_t_half_length(self, alpha):
        """
        :param alpha: significance level
        :returns half-length of 100(1-alpha)% t-confidence interval """

        return stat.t.ppf(1 - alpha / 2, self._n - 1) * self.get_stdev() / numpy.sqrt(self._n)

    def get_t_CI(self,alpha):
        """ calculates t-based confidence interval
        :param alpha: significance level
        :return: a list [l, u]
        """
        mean = self.get_mean()
        hl = self.get_t_half_length(alpha)

        return [mean - hl, mean + hl]

    def get_bootstrap_CI(self, alpha, num_samples):
        """ calculates empirical bootstrap confidence interval (abstract method to be overridden in derived classes)
        :param alpha: significance level
        :param num_samples: number of bootstrap samples
        :returns a list [L, U] """
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")

    def get_PI(self, alpha):
        """ calculates percentile interval (abstract method to be overridden in derived classes)
        :param alpha: significance level
        :returns a list [L, U]
         """
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")

    def get_summary(self, alpha, digits):
        """
        :param alpha: significance level
        :param digits: digits to round the numbers to
        :return: a list ['name', 'mean', 'confidence interval', 'prediction interval', 'st dev', 'min', 'max']
        """
        return [self.name,
                Support.format_number(self.get_mean(), digits),
                Support.format_interval(self.get_t_CI(alpha), digits),
                Support.format_interval(self.get_PI(alpha), digits),
                Support.format_number(self.get_stdev(), digits),
                Support.format_number(self.get_min(), digits),
                Support.format_number(self.get_max(), digits)]


class SummaryStat(Statistics):
    def __init__(self, name, data):
        """:param data: a list of data points"""

        Statistics.__init__(self, name)
        self._data = data
        self._n = len(data)
        self._mean = numpy.mean(data)
        self._stDev = numpy.std(data, ddof=1)  # unbiased estimator of the standard deviation

    def get_mean(self):
        return self._mean

    def get_stdev(self):
        return self._stDev

    def get_min(self):
        return numpy.min(self._data)

    def get_max(self):
        return numpy.max(self._data)

    def get_percentile(self, q):
        """
        :param q: percentile to compute (q in range [0, 100]
        :returns: qth percentile """

        return numpy.percentile(self._data, q)

    def get_bootstrap_CI(self, alpha, num_samples):
        """ calculates the empirical bootstrap confidence interval
        :param alpha: significance level
        :param num_samples: number of bootstrap samples
        :return: a list [l, u]
        """

        # set random number generator seed
        numpy.random.seed(1)

        # initialize delta array
        delta = numpy.zeros(num_samples)

        # obtain bootstrap samples
        for i in range(num_samples):
            sample_i = numpy.random.choice(self._data, size=self._n, replace=True)
            delta[i] = sample_i.mean() - self.get_mean()

        # return [l, u]
        return -numpy.percentile(delta, [100*(1 - alpha / 2.0), 100*alpha / 2.0]) + self.get_mean()

    def get_PI(self, alpha):
        """
        :param alpha: significance level
        :return: percentile interval in the format of list [l, u]
        """
        return [self.get_percentile(100*(1-alpha/2)), self.get_percentile(100*alpha/2)]


class DiscreteTimeStat(Statistics):
    """ to calculate statistics on observations accumulating over time """
    def __init__(self, name):
        Statistics.__init__(self, name)
        self._total = 0
        self._sumSquared = 0

    def record(self, obs):
        """ gets the next observation and update the current information"""
        self._total += obs
        self._sumSquared += obs ** 2
        self._n += 1
        if obs > self._max:
            self._max = obs
        if obs < self._min:
            self._min = obs

    def get_mean(self):
        if self._n > 0:
            return self._total / self._n
        else:
            return 0

    def get_stdev(self):
        if self._n>1:
            return math.sqrt(
                (self._sumSquared - self._total ** 2 / self._n)
                / (self._n - 1)
            )
        else:
            return 0

    def get_min(self):
        return self._min

    def get_max(self):
        return self._max

    def get_percentile(self, q):
        """ percentiles cannot be calculated for this statistics """
        return None

    def get_bootstrap_CI(self, alpha, num_samples):
        """ bootstrap confidence intervals cannot be calculated for this statistics """
        return None

    def get_PI(self, alpha):
        """ percentile intervals cannot be calculated for this statistics """
        return None


class ContinuousTimeStat(Statistics):
    """ to calculate statistics on the area-under-the-curve for observations accumulating over time """
    def __init__(self, name,  initial_time):
        """
        :param initial_time: it is assumed that the value of this sample path is zero at the initial time
        """
        Statistics.__init__(self, name)
        self._initialTime = initial_time
        self._area = 0
        self._areaSquared = 0
        self._lastObsTime = initial_time
        self._lastObsValue = 0

    def record(self, time, increment):
        """ gets the next observation and update the current information
        :param time: time of this change in the sample path
        :param increment: the amount of change (can be positive or negative) in the sample path
        """

        if time < self._initialTime:
            return

        if self._lastObsValue > self._max:
            self._max = self._lastObsValue
        if time == self._initialTime:
            self._min = self._lastObsValue
        elif self._lastObsValue < self._min:
            self._min = self._lastObsValue

        self._n += 1
        self._area += self._lastObsValue * (time - self._lastObsTime)
        self._areaSquared += (self._lastObsValue ** 2) * (time - self._lastObsTime)
        self._lastObsTime = time
        self._lastObsValue += increment

    def get_mean(self):
        if self._lastObsTime - self._initialTime > 0:
            return self._area / (self._lastObsTime - self._initialTime)
        else:
            return 0

    def get_stdev(self):
        var = 0
        if self._lastObsTime - self._initialTime > 0:
            var = self._areaSquared / (self._lastObsTime - self._initialTime) - self.get_mean() ** 2

        return math.sqrt(var)

    def get_min(self):
        return self._min

    def get_max(self):
        return self._max

    def get_percentile(self, q):
        """ percentiles cannot be calculated for this statistics """
        return None

    def get_bootstrap_CI(self, alpha, num_samples):
        """ bootstrap confidence intervals cannot be calculated for this statistics """
        return None

    def get_PI(self, alpha):
        """ percentile intervals cannot be calculated for this statistics """
        return None


class ComparativeStat(Statistics):
    '''
    list or array of sample data, with same length
    '''

    def __init__(self, x, y):
        self.x = x
        self.y = y

class DifferenceStat(ComparativeStat):

    def __init__(self, x, y):
        ComparativeStat.__init__(self, x, y)


class DifferenceStatIndp(DifferenceStat):

    def diff_ind(self, alpha):
        # confidence interval for independent data
        n = len(self.x)
        m = len(self.y)
        sig_x = np.std(self.x)
        sig_y = np.std(self.y)

        alpha = alpha/100.0

        # E[X] - E[Y]
        diff = np.mean(self.x) - np.mean(self.y)

        # calculate CI using formula: Welch's t-interval
        # ref: https://onlinecourses.science.psu.edu/stat414/node/203
        df_n = (sig_x ** 2.0 / n + sig_y ** 2.0 / m) ** 2.0
        df_d = (sig_x ** 2.0 / n) ** 2 / (n - 1) + (sig_y ** 2.0 / m) ** 2 / (m - 1)
        df = round(df_n / df_d, 0)

        # t distribution quantile
        q = scs.t.ppf(1 - (alpha / 2), df)
        c = (sig_x ** 2.0 / n + sig_y ** 2.0 / m) ** 0.5

        x_y = [diff - q * c, diff + q * c]

        return diff, x_y

    def percentile_d_ind(self, alpha, M):
        # 95% CI and mu for alpha th percentile of X-Y
        delta = np.ones(M)

        for i in range(M):
            x_i = np.random.choice(self.x, size=len(self.x), replace=True)
            y_i = np.random.choice(self.y, size=len(self.y), replace=True)

            delta[i] = np.percentile(x_i - y_i, alpha)

        q = np.percentile(delta, [2.5, 50, 97.5])

        return q

    def get_mean(self):
        return  self.x.mean() - self.y.mean()


class DifferenceStatPaired(DifferenceStat):

    def __init__(self, x, y):
        self.sumStat = SummaryStat(name, x - y)

    def get_mean(self):
        return self.sumStat.get_mean()

    def diff_paired(self, alpha):

        # confidence interval for paired data
        n = len(self.x)
        alpha = alpha/100.0

        # X-Y
        d = self.x - self.y
        d_mean = d.mean()

        # calculate CI using formula: paired t-interval
        # ref: https://onlinecourses.science.psu.edu/stat414/node/202
        df = n - 1

        # t distribution quantile
        q = scs.t.ppf(1 - (alpha / 2), df)
        c = (d.var() / n) ** 0.5
        x_y = [d_mean - q * c, d_mean + q * c]

        return d_mean, x_y

    def percentile_d_paired(self, alpha):
        d = self.x - self.y
        q = np.percentile(d, alpha)

        return q

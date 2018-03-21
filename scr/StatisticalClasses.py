import sys
import numpy as numpy
import scipy.stats as stat
import math

from scr import FormatFunctions as Support


class _Statistics(object):
    def __init__(self, name):
        """ abstract method to be overridden in derived classes"""
        self._name = name        # name of this statistics
        self._y_n = 0              # number of data points
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
        :param alpha: significance level (between 0 and 1)
        :returns half-length of 100(1-alpha)% t-confidence interval """

        return stat.t.ppf(1 - alpha / 2, self._y_n - 1) * self.get_stdev() / numpy.sqrt(self._y_n)

    def get_t_CI(self, alpha):
        """ calculates t-based confidence interval for population mean
        :param alpha: significance level (between 0 and 1)
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
        :param alpha: significance level (between 0 and 1)
        :returns a list [L, U]
         """
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")

    def get_summary(self, alpha, digits):
        """
        :param alpha: significance level
        :param digits: digits to round the numbers to
        :return: a list ['name', 'mean', 'confidence interval', 'percentile interval', 'st dev', 'min', 'max']
        """
        return [self._name,
                Support.format_number(self.get_mean(), digits),
                Support.format_interval(self.get_t_CI(alpha), digits),
                Support.format_interval(self.get_PI(alpha), digits),
                Support.format_number(self.get_stdev(), digits),
                Support.format_number(self.get_min(), digits),
                Support.format_number(self.get_max(), digits)]


class SummaryStat(_Statistics):
    def __init__(self, name, data):
        """:param data: a list or numpy.array of data points"""

        _Statistics.__init__(self, name)
        # convert data to numpy array if needed
        if type(data) == list:
            self._data = numpy.array(data)
        else:
            self._data = data

        self._n = len(self._data)
        self._mean = numpy.mean(self._data)
        self._stDev = numpy.std(self._data, ddof=1)  # unbiased estimator of the standard deviation

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
        :param q: percentile to compute (q in range [0, 100])
        :returns: qth percentile """

        return numpy.percentile(self._data, q)

    def get_bootstrap_CI(self, alpha, num_samples):
        """ calculates the empirical bootstrap confidence interval
        :param alpha: significance level (between 0 and 1)
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
        return self.get_mean() - numpy.percentile(delta, [100*(1-alpha / 2.0), 100*alpha / 2.0])

    def get_PI(self, alpha):
        """
        :param alpha: significance level (between 0 and 1)
        :return: percentile interval in the format of list [l, u]
        """
        return [self.get_percentile(100*alpha/2), self.get_percentile(100*(1-alpha/2))]


class DiscreteTimeStat(_Statistics):
    """ to calculate statistics on observations accumulating over time """
    def __init__(self, name):
        _Statistics.__init__(self, name)
        self._total = 0
        self._sumSquared = 0

    def record(self, obs):
        """ gets the next observation and update the current information"""
        self._total += obs
        self._sumSquared += obs ** 2
        self._y_n += 1
        if obs > self._max:
            self._max = obs
        if obs < self._min:
            self._min = obs

    def get_mean(self):
        if self._y_n > 0:
            return self._total / self._y_n
        else:
            return 0

    def get_stdev(self):
        if self._y_n>1:
            return math.sqrt(
                (self._sumSquared - self._total ** 2 / self._y_n)
                / (self._y_n - 1)
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


class ContinuousTimeStat(_Statistics):
    """ to calculate statistics on the area-under-the-curve for observations accumulating over time """
    def __init__(self, name,  initial_time):
        """
        :param initial_time: it is assumed that the value of this sample path is zero at the initial time
        """
        _Statistics.__init__(self, name)
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

        self._y_n += 1
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


class ComparativeStat(_Statistics):
    def __init__(self, name, x, y_ref):
        """
        :param x: list or numpy.array of first set of observations
        :param y_ref: list or numpy.array of second set of observations (the reference) 
        """
        _Statistics.__init__(self, name)

        if type(x) == list:
            self._x = numpy.array(x)
        else:
            self._x = x

        if type(y_ref) == list:
            self._y_ref = numpy.array(y_ref)
        else:
            self._y_ref = y_ref

        self._x_n = len(self._x)  # number of observations for x
        self._y_n = len(self._y_ref)    # number of observations for y_ref


class _DifferenceStat(ComparativeStat):

    def __init__(self, name, x, y_ref):
        """
        :param x: list or numpy.array of first set of observations
        :param y_ref: list or numpy.array of second set of observations
        """
        ComparativeStat.__init__(self, name, x, y_ref)


class DifferenceStatPaired(_DifferenceStat):

    def __init__(self, name, x, y_ref):
        """
        :param x: list or numpy.array of first set of observations
        :param y_ref: list or numpy.array of second set of observations
        """
        _DifferenceStat.__init__(self, name, x, y_ref)
        # create a summary statistics for the element-wise difference

        if len(self._x) != len(self._y_ref):
            raise ValueError('Two samples should have the same size.')

        self._dStat = SummaryStat(name, self._x - self._y_ref)

    def get_mean(self):
        return self._dStat.get_mean()

    def get_stdev(self):
        return self._dStat.get_stdev()

    def get_min(self):
        return self._dStat.get_min()

    def get_max(self):
        return self._dStat.get_max()

    def get_percentile(self, q):
        return self._dStat.get_percentile(q)

    def get_bootstrap_CI(self, alpha, num_samples):
        return self._dStat.get_bootstrap_CI(alpha, num_samples)

    def get_PI(self, alpha):
        return self._dStat.get_PI(alpha)


class DifferenceStatIndp(_DifferenceStat):

    def __init__(self, name, x, y_ref):
        """
        :param x: list or numpy.array of first set of observations
        :param y_ref: list or numpy.array of second set of observations
        """
        _DifferenceStat.__init__(self, name, x, y_ref)

        # generate random realizations for random variable X - Y
        # this will be used for calculating the projection interval
        numpy.random.seed(1)
        # find the maximum of the number of observations
        max_n = max(self._x_n, self._y_n, 1000)
        x_i = numpy.random.choice(self._x, size=max_n, replace=True)
        y_i = numpy.random.choice(self._y_ref, size=max_n, replace=True)
        self._sum_stat_sample_delta = SummaryStat(name, x_i - y_i)

    def get_mean(self):
        """
        for independent variable x and y, E(x-y) = E(x) - E(y)
        :return: sample mean of (x-y)
        """
        return numpy.mean(self._x) - numpy.mean(self._y_ref)

    def get_stdev(self):
        """
        for independent variable x and y, var(x-y) = var_x + var_y
        :returns: sample standard deviation
        """
        var_x = numpy.var(self._x)
        var_y = numpy.var(self._y_ref)
        return numpy.sqrt(var_x + var_y)

    def get_min(self):
        return None

    def get_max(self):
        return None

    def get_percentile(self, q):
        """
        for independent variable x and y, percentiles are given after re-sampling
        :param q: the percentile want to return, in [0, 100]
        :return: qth percentile of sample (x-y)
        """
        return self._sum_stat_sample_delta.get_percentile(q)

    def get_bootstrap_CI(self, alpha, num_samples):
        """
        :param alpha: confidence level
        :param num_samples: number of samples
        :return: empirical bootstrap confidence interval
        """
        # set random number generator seed
        numpy.random.seed(1)

        # initialize difference array
        diff = numpy.zeros(num_samples)

        # obtain bootstrap samples
        for i in range(num_samples):
            x_i = numpy.random.choice(self._x, size=self._y_n, replace=True)
            y_i = numpy.random.choice(self._y_ref, size=self._y_n, replace=True)
            d_temp = x_i - y_i
            diff[i] = numpy.mean(d_temp)

        return numpy.percentile(diff, [100*alpha/2.0, 100*(1-alpha/2.0)])

    def get_t_half_length(self, alpha):
        """
        Independent x_bar - y_bar is t distribution
        ref: https://onlinecourses.science.psu.edu/stat414/node/203
        :param alpha: confidence level
        :return: confidence interval of x_bar - y_bar
        """

        sig_x = numpy.std(self._x)
        sig_y = numpy.std(self._y_ref)

        alpha = alpha / 100.0

        # calculate CI using formula: Welch's t-interval
        df_n = (sig_x ** 2.0 / self._x_n + sig_y ** 2.0 / self._y_n) ** 2.0
        df_d = (sig_x ** 2.0 / self._x_n) ** 2 / (self._x_n - 1) \
               + (sig_y ** 2.0 / self._y_n) ** 2 / (self._y_n - 1)
        df = round(df_n / df_d, 0)

        # t distribution quantile
        t_q = stat.t.ppf(1 - (alpha / 2), df)
        st_dev = (sig_x ** 2.0 / self._x_n + sig_y ** 2.0 / self._y_n) ** 0.5

        return t_q*st_dev

    def get_t_CI(self, alpha):

        interval = self.get_t_half_length(alpha)
        diff = numpy.mean(self._x) - numpy.mean(self._y_ref)

        return [diff - interval, diff + interval]

    def get_PI(self, alpha):
        return self._sum_stat_sample_delta.get_PI(alpha)


class _RatioStat(ComparativeStat):

    def __init__(self, name, x, y_ref):
        """
        :param x: list or numpy.array of first set of observations
        :param y_ref: list or numpy.array of second set of observations
        """
        ComparativeStat.__init__(self, name, x, y_ref)
        # make sure no 0 in the denominator variable
        if not (self._y_ref != 0).all():
            raise ValueError('invalid value of y, the ratio is not computable')


class RatioStatPaired(_RatioStat):

    def __init__(self, name, x, y_ref):
        """
        :param x: list or numpy.array of first set of observations
        :param y_ref: list or numpy.array of second set of observations
        """
        _RatioStat.__init__(self, name, x, y_ref)

        if len(self._x) != len(self._y_ref):
            raise ValueError('Two samples should have the same size.')

        # add element-wise ratio
        ratio = numpy.divide(self._x, self._y_ref)
        self.ratioStat = SummaryStat(name, ratio)

    def get_mean(self):
        return self.ratioStat.get_mean()

    def get_stdev(self):
        return self.ratioStat.get_stdev()

    def get_min(self):
        return self.ratioStat.get_min()

    def get_max(self):
        return self.ratioStat.get_max()

    def get_percentile(self, q):
        return self.ratioStat.get_percentile(q)

    def get_bootstrap_CI(self, alpha, num_samples):
        return self.ratioStat.get_bootstrap_CI(alpha, num_samples)

    def get_PI(self, alpha):
        return self.ratioStat.get_PI(alpha)


class RatioStatIndp(_RatioStat):

    def __init__(self, name, x, y_ref):
        """
        :param x: list or numpy.array of first set of observations
        :param y_ref: list or numpy.array of second set of observations (reference)
        """

        _RatioStat.__init__(self, name, x, y_ref)

        # generate random realizations for random variable X/Y
        numpy.random.seed(1)
        # find the maximum of the number of observations
        max_n = max(self._x_n, self._y_n, 1000)
        x_resample = numpy.random.choice(self._x, size=max_n, replace=True)
        y_resample = numpy.random.choice(self._y_ref, size=max_n, replace=True)

        self._sum_stat_sample_ratio = SummaryStat(name, numpy.divide(x_resample, y_resample))

    def get_mean(self):
        return self._sum_stat_sample_ratio.get_mean()

    def get_stdev(self):
        """
        for independent variable x and y, var(x/y) = E(x^2)*E(1/y^2)-E(x)^2*(E(1/y)^2)
        :return: std(x/y)
        """
        if self._y_ref.mean() == 0:
            raise ValueError('invalid value of mean of y, the ratio is not computable')

        var = numpy.mean(self._x ** 2) * numpy.mean(1.0 / self._y_ref ** 2) - \
              (numpy.mean(self._x) ** 2) * (numpy.mean(1.0 / self._y_ref) ** 2)
        return numpy.sqrt(var)

    def get_min(self):
        return self._sum_stat_sample_ratio.get_min()

    def get_max(self):
        return self._sum_stat_sample_ratio.get_max()

    def get_percentile(self, q):
        """
        for independent variable x and y, percentiles are given after re-sampling
        :param q: the percentile want to return, in [0,100]
        :return: qth percentile of sample (x/y)
        """
        return self._sum_stat_sample_ratio.get_percentile(q)

    def get_t_half_length(self, alpha):
        return self._sum_stat_sample_ratio.get_t_half_length(alpha)

    def get_t_CI(self, alpha):
        return self._sum_stat_sample_ratio.get_t_CI(alpha)

    def get_bootstrap_CI(self, alpha, num_samples):
        """
        :param alpha: confidence level
        :param num_samples: number of samples
        :return: empirical bootstrap confidence interval
        """
        # set random number generator seed
        numpy.random.seed(1)

        # initialize ratio array
        ratio = numpy.zeros(num_samples)

        # obtain bootstrap samples
        n = max(self._x_n, self._y_n)
        for i in range(num_samples):
            x_i = numpy.random.choice(self._x, size=n, replace=True)
            y_i = numpy.random.choice(self._y_ref, size=n, replace=True)
            r_temp = numpy.divide(x_i, y_i)
            ratio[i] = numpy.mean(r_temp)

        return numpy.percentile(ratio, [100*alpha/2.0, 100*(1-alpha/2.0)])

    def get_PI(self, alpha):
        return self._sum_stat_sample_ratio.get_PI(alpha)


class _RelativeDifference(ComparativeStat):
    """ class to make inference about (X-Y_ref)/Y_ref"""

    def __init__(self, name, x, y_ref):
        """
        :param x: list or numpy.array of first set of observations
        :param y_ref: list or numpy.array of second set of observations used as the reference values
        """
        ComparativeStat.__init__(self, name, x, y_ref)

        # make sure no 0 in the denominator variable y
        if not (self._y_ref != 0).all():
            raise ValueError('invalid value of x, the ratio is not computable')


class RelativeDifferencePaired(_RelativeDifference):

    def __init__(self, name, x, y_ref):
        """
        :param x: list or numpy.array of first set of observations
        :param y_ref: list or numpy.array of second set of observations
        """
        _RelativeDifference.__init__(self, name, x, y_ref)

        if len(self._x) != len(self._y_ref):
            raise ValueError('Two samples should have the same size.')

        # add element-wise ratio
        ratio = numpy.divide(self._x, self._y_ref)
        self.relativeDiffStat = SummaryStat(name, ratio - 1)

    def get_mean(self):
        return self.relativeDiffStat.get_mean()

    def get_stdev(self):
        return self.relativeDiffStat.get_stdev()

    def get_min(self):
        return self.relativeDiffStat.get_min()

    def get_max(self):
        return self.relativeDiffStat.get_max()

    def get_percentile(self, q):
        return self.relativeDiffStat.get_percentile(q)

    def get_bootstrap_CI(self, alpha, num_samples):
        return self.relativeDiffStat.get_bootstrap_CI(alpha, num_samples)

    def get_PI(self, alpha):
        return self.relativeDiffStat.get_PI(alpha)


class RelativeDifferenceIndp(_RelativeDifference):
    def __init__(self, name, x, y_ref):
        """
        :param x: list or numpy.array of first set of observations
        :param y: list or numpy.array of second set of observations
        """
        _RelativeDifference.__init__(self, name, x, y_ref)

        # generate random realizations for random variable (X-Y)/Y
        numpy.random.seed(1)
        # find the maximum of the number of observations
        max_n = max(self._x_n, self._y_n, 1000)
        x_resample = numpy.random.choice(self._x, size=max_n, replace=True)
        y_resample = numpy.random.choice(self._y_ref, size=max_n, replace=True)

        self._sum_stat_sample_relativeRatio = SummaryStat(name, numpy.divide(x_resample, y_resample) - 1)

    def get_mean(self):
        return self._sum_stat_sample_relativeRatio.get_mean()

    def get_stdev(self):
        """
        for independent variable x and y, var(x/y) = E(x^2)*E(1/y^2)-E(x)^2*(E(1/y)^2)
        and var(x/y - 1) = var(x/y)
        :return: std(x/y - 1)
        """
        if self._y_ref.mean() == 0:
            raise ValueError('invalid value of mean of y, the ratio is not computable')

        var = numpy.mean(self._x ** 2) * numpy.mean(1.0 / self._y_ref ** 2) - \
              (numpy.mean(self._x) ** 2) * (numpy.mean(1.0 / self._y_ref) ** 2)
        return numpy.sqrt(var)

    def get_min(self):
        return self._sum_stat_sample_relativeRatio.get_min()

    def get_max(self):
        return self._sum_stat_sample_relativeRatio.get_max()

    def get_percentile(self, q):
        """
        for independent variable x and y, percentiles are given after re-sampling
        :param q: the percentile want to return, in [0,100]
        :return: qth percentile of sample (x-y)/y
        """
        return self._sum_stat_sample_relativeRatio.get_percentile(q)

    def get_t_half_length(self, alpha):
        return self._sum_stat_sample_relativeRatio.get_t_half_length(alpha)

    def get_t_CI(self, alpha):
        return self._sum_stat_sample_relativeRatio.get_t_CI(alpha)

    def get_bootstrap_CI(self, alpha, num_samples):
        """
        :param alpha: confidence level
        :param num_samples: number of samples
        :return: empirical bootstrap confidence interval
        """
        # set random number generator seed
        numpy.random.seed(1)

        # initialize ratio array
        ratio = numpy.zeros(num_samples)

        # obtain bootstrap samples
        n = max(self._x_n, self._y_n)
        for i in range(num_samples):
            x_i = numpy.random.choice(self._x, size=n, replace=True)
            y_i = numpy.random.choice(self._y_ref, size=n, replace=True)
            r_temp = numpy.divide(x_i, y_i) - 1
            ratio[i] = numpy.mean(r_temp)

        return numpy.percentile(ratio, [100*alpha/2.0, 100*(1-alpha/2.0)])

    def get_PI(self, alpha):
        return self._sum_stat_sample_relativeRatio.get_PI(alpha)

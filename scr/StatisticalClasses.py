import sys
import numpy as numpy
import scipy.stats as stat
# no 'math' module under python 2.7, comment out for my version
# import math

# need to set the working dictionary to the sub-file to import SupportFunctions.py
sys.path.append('../HPM573_SupportLib/scr')
import SupportFunctions as Support


class Statistics(object):
    def __init__(self, name):
        """ abstract method to be overridden in derived classes"""
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

        return stat.t.ppf(1 - alpha / 200.0, self._n - 1) * self.get_stdev() / numpy.sqrt(self._n)

    def get_t_CI(self,alpha):
        """ calculates t-based confidence interval for population mean
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
        :return: a list ['name', 'mean', 'confidence interval', 'percentile interval', 'st dev', 'min', 'max']
        """
        return [self.name,
                Support.format_number(self.get_mean(), digits),
                Support.format_interval(self.get_t_CI(alpha), digits),
                Support.format_interval(self.get_PI(alpha), digits),
                Support.format_number(self.get_stdev(), digits),
                Support.format_number(self.get_min(), digits),
                Support.format_number(self.get_max(), digits)]


# no change
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
        return -numpy.percentile(delta, [100 - alpha / 2.0, alpha / 2.0]) + self.get_mean()

    def get_PI(self, alpha):
        """
        :param alpha: significance level
        :return: percentile interval in the format of list [l, u]
        """
        return [self.get_percentile(100*(1-alpha/2)), self.get_percentile(100*alpha/2)]

# no change
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

# no change
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
    def __init__(self, name, x, y):
        Statistics.__init__(self,name)
        self.x = x
        self.y = y


class RatioStat(ComparativeStat):

    def __init__(self, name, x, y):
        ComparativeStat.__init__(self, name, x, y)
        # make sure no 0 in the denominator variable
        if (self.y != 0).all() == False:
            raise ValueError('invalid value of y, the ratio is not computable')

class RatioStatIndp(RatioStat):

    def __init__(self, name, x, y):
        RatioStat.__init__(self, name, x, y)
        self._n = len(self.x)              # number of data points
        # since the calling of mean and stdev functions here,
        # if mean(y) == 0, the RatioStatIndp object can not be define
        self._mean = self.get_mean()       # sample mean
        self._stDev = self.get_stdev()     # sample standard deviation
        self._max = self.get_max()         # maximum
        self._min = self.get_max()         # minimum



    def get_mean(self):
        '''
        for independent variable x and y, E(x/y)=E(x)/E(y)
        :return: E(x)/E(y)
        '''
        if self.y.mean() == 0:
            raise ValueError('invalid value of mean of y, the ratio is not computable')

        mu = self.x.mean()/self.y.mean()

        return mu

    def get_stdev(self):
        '''
        for independent variable x and y, var(x/y) = E(x^2)/E(y^2)-E(x)^2/E(y)^2
        :return: std(x/y)
        '''
        if self.y.mean() == 0:
            raise ValueError('invalid value of mean of y, the ratio is not computable')

        var = numpy.mean(self.x**2) * numpy.mean(1.0/self.y**2) - \
              (numpy.mean(self.x)**2)/(numpy.mean(self.y)**2)
        std = numpy.sqrt(var)

        return std

    def get_min(self):
        '''
        for independent variable x and y, sample min, max, and percentiles are given after permutation
        re-sampling with sample size M
        :return: min(x/y)
        '''
        M = self._n
        x_i = numpy.random.choice(self.x, size= M, replace=True)
        y_i = numpy.random.choice(self.y, size= M, replace=True)
        r_temp = numpy.divide(x_i, y_i)

        min = numpy.min(r_temp)

        return min

    def get_max(self):
        '''
        for independent variable x and y, sample min, max, and percentiles are given after permutation
        re-sampling with sample size M
        :return: max(x/y)
        '''
        M = self._n
        x_i = numpy.random.choice(self.x, size=M, replace=True)
        y_i = numpy.random.choice(self.y, size=M, replace=True)
        r_temp = numpy.divide(x_i, y_i)

        max = numpy.max(r_temp)

        return max

    def get_percentile(self, q):
        '''
        for independent variable x and y, sample min, max, and percentiles are given after permutation
        re-sampling with sample size M
        :return: q th percentile of (x/y)
        '''
        M = self._n
        x_i = numpy.random.choice(self.x, size=M, replace=True)
        y_i = numpy.random.choice(self.y, size=M, replace=True)
        r_temp = numpy.divide(x_i, y_i)

        qth = numpy.percentile(r_temp, q)

        return qth

    def get_t_half_length(self):
        # Independent Gaussian ratio distribution is Cauchy
        return "This variable is not applicable for t distribution"

    def get_t_CI(self):
        # Independent Gaussian ratio distribution is Cauchy
        return "This variable is not applicable for t distribution"

    def get_bootstrap_CI(self, alpha, M, method):
        '''
        :param alpha: confidence level
        :param M: number of samples
        :param method: choose to calculate 'for_mean': E(x)/E(y) or 'for_ratio': E(x/y)
        :return: bootstrap confidence interval
        '''
        if method == 'for_mean':
            # calculate the CI of E(x)/E(y) with confidence 100-alpha
            delta = numpy.ones(M)
            # assert all the means should not be 0
            if self.y.mean() == 0:
                raise ValueError('invalid value of mean of y, the ratio is not computable')

            r_bar = self.x.mean() / self.y.mean()

            for i in range(M):
                x_i = numpy.random.choice(self.x, size=len(self.x), replace=True)
                y_i = numpy.random.choice(self.y, size=len(self.y), replace=True)

                # assert all the means should not be 0
                if y_i.mean() == 0:
                    raise ValueError('invalid value of mean of y, the ratio is not computable')
                ri_bar = x_i.mean() / y_i.mean()

                delta[i] = ri_bar - r_bar

            result = -numpy.percentile(delta, [100 - alpha / 2.0, alpha / 2.0]) + r_bar
        elif method == 'for_ratio':
            # calculate the CI of E(x/y) with confidence 100-alpha
            ratio = numpy.ones(M)

            for i in range(M):
                x_i = numpy.random.choice(self.x, size=len(self.x), replace=True)
                y_i = numpy.random.choice(self.y, size=len(self.y), replace=True)
                r_temp = numpy.divide(x_i, y_i)
                ratio[i] = numpy.mean(r_temp)

            r_bar = numpy.mean(ratio)
            result = numpy.percentile(ratio, [alpha / 2.0, 100 - alpha / 2.0])

        return result
    
    def get_PI(self, alpha):
        # significant level alpha percentile for random sample of x/y with size M
        M = self._n
        x_i = numpy.random.choice(self.x, size= M, replace=True)
        y_i = numpy.random.choice(self.y, size= M, replace=True)
        r_temp = numpy.divide(x_i, y_i)

        q = numpy.percentile(r_temp, [alpha/2.0, 100-alpha/2.0])

        return q

    def get_summary(self, alpha, digits):
        """
        due to change of argument, need override origin summary function
        :param alpha: significance level
        :param digits: digits to round the numbers to
        :return: a list ['name', 'mean', 'percentile interval', 'st dev', 'min', 'max', 'confidence interval',
        'bootstrap confidence interval']
        """
        return [self.name,
                Support.format_number(self.get_mean(), digits),
                self.get_t_CI(),
                Support.format_interval(self.get_PI(alpha), digits),
                Support.format_number(self.get_stdev(), digits),
                Support.format_number(self.get_min(), digits),
                Support.format_number(self.get_max(), digits),
                Support.format_number(self.get_percentile(10), digits),
                Support.format_interval(self.get_bootstrap_CI(alpha, 1000,'for_mean'), digits)]

class RatioStatPaired(RatioStat):

    def __init__(self, name, x, y):
        RatioStat.__init__(self, name, x, y)
        # add element-wise ratio
        self.r = numpy.divide(self.x, self.y)
        self._n = len(self.r)              # number of data points
        self._mean = self.get_mean()       # sample mean
        self._stDev = self.get_stdev()     # sample standard deviation
        self._max = self.get_max()         # maximum
        self._min = self.get_max()         # minimum

    def get_mean(self):
        '''
        for paired variable x and y, calculate ratio first
        :return: E(x/y)
        '''
        mu = numpy.mean(self.r)

        return mu

    def get_stdev(self):
        '''
        for paired variable x and y, calculate ratio first
        :return: std(r)
        '''
        std = numpy.std(self.r)

        return std

    def get_min(self):
        '''
        for paired variable x and y, calculate ratio first
        :return: min(r)
        '''
        min = numpy.min(self.r)

        return min

    def get_max(self):
        '''
        for paired variable x and y, calculate ratio first
        :return: max(r)
        '''
        max = numpy.max(self.r)

        return max

    def get_percentile(self, q):
        '''
        for paired variable x and y, calculate ratio first
        :return: q th percentile of (x/y)
        '''
        qth = numpy.percentile(self.r, q)

        return qth

    def get_bootstrap_CI(self, alpha, M, method=''):
        '''
        :param alpha: confidence level
        :param M: number of samples
        :param method: choose to calculate 'for_mean': E(x)/E(y) or 'for_ratio': E(x/y)
        :return: bootstrap confidence interval
        '''

        if method == 'for_mean':
            # calculate the CI of E(x)/E(y) with confidence 100-alpha
            delta = numpy.ones(M)

            # assert all the means should not be 0
            if self.y.mean() == 0:
                raise ValueError('invalid value of mean of y, the ratio is not computable')

            r_bar = self.x.mean() / self.y.mean()

            for i in range(M):
                # choice paired sample by index
                ind = numpy.random.choice(range(len(self.x)), size=len(self.x), replace=True)
                x_i = self.x[ind]
                y_i = self.y[ind]

                # assert all the means should not be 0
                if y_i.mean() == 0:
                    raise ValueError('invalid value of mean of y, the ratio is not computable')
                ri_bar = x_i.mean() / y_i.mean()

                delta[i] = ri_bar - r_bar

            result = -numpy.percentile(delta, [100 - alpha / 2.0, alpha / 2.0]) + r_bar

        elif method == 'for_ratio':
            # calculate the CI of E(x/y) with confidence 100-alpha
            a = SummaryStat('tempr',self.r)
            result = a.get_bootstrap_CI(alpha, M)

        return result

    def get_PI(self, alpha):
        # significant level alpha percentile for ratio vector
        q = numpy.percentile(self.r, [alpha/2.0, 100-alpha/2.0])

        return q

    def get_summary(self, alpha, digits):
        """
        due to change of argument, need override origin summary function
        :param alpha: significance level
        :param digits: digits to round the numbers to
        :return: a list ['name', 'mean', 'percentile interval', 'st dev', 'min', 'max', 'confidence interval',
        'bootstrap confidence interval']
        """
        return [self.name,
                Support.format_number(self.get_mean(), digits),
                Support.format_interval(self.get_t_CI(alpha), digits),
                Support.format_interval(self.get_PI(alpha), digits),
                Support.format_number(self.get_stdev(), digits),
                Support.format_number(self.get_min(), digits),
                Support.format_number(self.get_max(), digits),
                Support.format_number(self.get_percentile(10), digits),
                Support.format_interval(self.get_bootstrap_CI(alpha, 1000, 'for_ratio'), digits)]



class DifferenceStat(ComparativeStat):

    def __init__(self, name, x, y):
        ComparativeStat.__init__(self, name, x, y)

class DifferenceStatIndp(DifferenceStat):

    def __init__(self, name, x, y):
        DifferenceStat.__init__(self, name, x, y)
        # add element-wise difference
        self.d = self.x - self.y

        self._n = len(self.d)              # number of data points
        self._mean = self.get_mean()       # sample mean
        self._stDev = self.get_stdev()     # sample standard deviation
        self._max = self.get_max()         # maximum
        self._min = self.get_max()         # minimum

    def get_mean(self):
        '''
        for independent variable x and y, E(x-y) = E(x) - E(y)
        :return: sample mean of (x-y)
        '''
        mu = numpy.mean(self.x) - numpy.mean(self.y)

        return mu

    def get_stdev(self):
        '''
        for independent variable x and y, (x-y) ~ Normal(mu_x-mu_y, var_x+var_y)
        var(x-y) = var(x) + var(y)
        :return: sample standard deviation
        '''
        var_x = numpy.var(self.x)
        var_y = numpy.var(self.y)
        std = numpy.sqrt(var_x + var_y)

        return std

    def get_min(self):
        '''
        for independent variable x and y, sample min, max, and percentiles are given after permutation
        re-sampling with sample size M
        :return: sample min of (x-y)
        '''
        M = self._n
        x_i = numpy.random.choice(self.x, size= M, replace=True)
        y_i = numpy.random.choice(self.y, size= M, replace=True)
        d_temp = x_i - y_i

        min = numpy.min(d_temp)

        return min

    def get_max(self):
        '''
        for independent variable x and y, sample min, max, and percentiles are given after permutation
        re-sampling with sample size M
        :return: sample max of (x-y)
        '''
        M = self._n
        x_i = numpy.random.choice(self.x, size= M, replace=True)
        y_i = numpy.random.choice(self.y, size= M, replace=True)
        d_temp = x_i - y_i

        max = numpy.max(d_temp)

        return max

    def get_percentile(self, q):
        '''
        for independent variable x and y, sample min, max, and percentiles are given after permutation
        re-sampling with sample size M
        :return: sample quantile of (x-y)
        '''
        M = self._n
        x_i = numpy.random.choice(self.x, size=M, replace=True)
        y_i = numpy.random.choice(self.y, size=M, replace=True)
        d_temp = x_i - y_i

        qth = numpy.percentile(d_temp, q)

        return qth

    def get_bootstrap_CI(self, alpha, M, method):
        '''
        :param alpha: confidence level
        :param M: number of samples
        :param method: choose to calculate CI 'for_mean' (mu_x - mu_y) or 'for_diff': (x - y),
        get_t_CI is equivalent to 'for_mean' method in 'get_bootstrap_CI', theoretical
        :return: bootstrap confidence interval
        '''
        if method == 'for_mean':
            # confidence interval for independent data's (mu_x - mu_y)
            delta = numpy.ones(M)
            d_bar = self.x.mean() - self.y.mean()

            for i in range(M):
                x_i = numpy.random.choice(self.x, size=len(self.x), replace=True)
                y_i = numpy.random.choice(self.y, size=len(self.y), replace=True)

                # assert all the means should not be 0
                di_bar = x_i.mean() - y_i.mean()

                delta[i] = di_bar - d_bar

            result = -numpy.percentile(delta, [100 - alpha / 2.0, alpha / 2.0]) + d_bar

        elif method == 'for_diff':
            # calculate the CI of mu_(x-y) with confidence 100-alpha
            delta = numpy.ones(M)

            for i in range(M):
                x_i = numpy.random.choice(self.x, size=len(self.x), replace=True)
                y_i = numpy.random.choice(self.y, size=len(self.y), replace=True)
                d_temp = x_i - y_i
                delta[i] = numpy.mean(d_temp)

            d_bar = numpy.mean(delta)
            result = numpy.percentile(delta, [alpha / 2.0, 100 - alpha / 2.0])

        return result

    def get_t_half_length(self, alpha):
        '''
        Independent x_bar - y_bar is t distribution
        calculate CI using formula: Welch's t-interval
        ref: https://onlinecourses.science.psu.edu/stat414/node/203
        :return: confidence interval of x_bar - y_bar
        '''
        n = len(self.x)
        m = len(self.y)
        sig_x = numpy.std(self.x)
        sig_y = numpy.std(self.y)

        alpha = alpha / 100.0

        # E[X] - E[Y]
        diff = numpy.mean(self.x) - numpy.mean(self.y)

        # calculate CI using formula: Welch's t-interval
        # ref: https://onlinecourses.science.psu.edu/stat414/node/203
        df_n = (sig_x ** 2.0 / n + sig_y ** 2.0 / m) ** 2.0
        df_d = (sig_x ** 2.0 / n) ** 2 / (n - 1) + (sig_y ** 2.0 / m) ** 2 / (m - 1)
        df = round(df_n / df_d, 0)

        # t distribution quantile
        q = stat.t.ppf(1 - (alpha / 2), df)
        c = (sig_x ** 2.0 / n + sig_y ** 2.0 / m) ** 0.5

        return q*c

    def get_t_CI(self, alpha):

        interval = self.get_t_half_length(alpha)
        diff = numpy.mean(self.x) - numpy.mean(self.y)
        x_y = [diff - interval, diff + interval]

        return x_y

    def get_PI(self, alpha):
        '''
        for independent variable x and y, sample min, max, and percentiles are given after permutation
        re-sampling with sample size M
        :return: significant level alpha percentile for random sample of x-y with size M
        '''
        M = self._n
        x_i = numpy.random.choice(self.x, size= M, replace=True)
        y_i = numpy.random.choice(self.y, size= M, replace=True)
        d_temp = x_i - y_i

        q = numpy.percentile(d_temp, [alpha/2.0, 100-alpha/2.0])

        return q

    def get_summary(self, alpha, digits):
        """
        due to change of argument, need override origin summary function
        :param alpha: significance level
        :param digits: digits to round the numbers to
        :return: a list ['name', 'mean', 'percentile interval', 'st dev', 'min', 'max', 'confidence interval',
        'bootstrap confidence interval']
        """
        return [self.name,
                Support.format_number(self.get_mean(), digits),
                Support.format_interval(self.get_t_CI(alpha), digits),
                Support.format_interval(self.get_PI(alpha), digits),
                Support.format_number(self.get_stdev(), digits),
                Support.format_number(self.get_min(), digits),
                Support.format_number(self.get_max(), digits),
                Support.format_number(self.get_percentile(10), digits),
                Support.format_interval(self.get_bootstrap_CI(alpha, 1000, 'for_mean'), digits)]

class DifferenceStatPaired(DifferenceStat):

    def __init__(self, name, x, y):
        DifferenceStat.__init__(self, name, x, y)
        # add element-wise difference
        self.d = self.x - self.y

        self._n = len(self.d)              # number of data points
        self._mean = self.get_mean()       # sample mean
        self._stDev = self.get_stdev()     # sample standard deviation
        self._max = self.get_max()         # maximum
        self._min = self.get_max()         # minimum

    def get_mean(self):
        mu = numpy.mean(self.d)

        return mu

    def get_stdev(self):
        std = numpy.std(self.d)

        return std

    def get_min(self):
        min = numpy.min(self.d)

        return min

    def get_max(self):
        max = numpy.max(self.d)

        return max

    def get_percentile(self, q):
        qth = numpy.percentile(self.d, q)

        return qth

    def get_bootstrap_CI(self, alpha, M):
        '''
        :param alpha: confidence level
        :param M: number of samples
        :return: bootstrap confidence interval
        '''
        # confidence interval for paired data
        n = len(self.x)
        alpha = alpha / 100.0

        # X-Y
        d = self.x - self.y
        d_mean = d.mean()

        # calculate CI using formula: paired t-interval
        # ref: https://onlinecourses.science.psu.edu/stat414/node/202
        df = n - 1

        # t distribution quantile
        q = stat.t.ppf(1 - (alpha / 2), df)
        c = (d.var() / n) ** 0.5
        x_y = [d_mean - q * c, d_mean + q * c]

        return x_y

    def get_PI(self, alpha):
        # significant level alpha percentile for difference vector
        q = numpy.percentile(self.d, [alpha/2.0, 100-alpha/2.0])

        return q

    def get_summary(self, alpha, digits):
        """
        due to change of argument, need override origin summary function
        :param alpha: significance level
        :param digits: digits to round the numbers to
        :return: a list ['name', 'mean', 'percentile interval', 'st dev', 'min', 'max', 'confidence interval',
        'bootstrap confidence interval']
        """
        return [self.name,
                Support.format_number(self.get_mean(), digits),
                Support.format_interval(self.get_t_CI(alpha), digits),
                Support.format_interval(self.get_PI(alpha), digits),
                Support.format_number(self.get_stdev(), digits),
                Support.format_number(self.get_min(), digits),
                Support.format_number(self.get_max(), digits),
                Support.format_number(self.get_percentile(10), digits),
                Support.format_interval(self.get_bootstrap_CI(alpha, 1000), digits)]




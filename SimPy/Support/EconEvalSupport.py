from scipy.optimize import minimize
import SimPy.StatisticalClasses as Stat
import SimPy.RandomVariantGenerators as RVG
import numpy as np


def assert_np_list(obs, error_message):
    """
    :param obs: list of observations to convert to np.array
    :param error_message: error message to display if conversion failed
    :return: np.array of obs
    """

    if type(obs) is not list and type(obs) is not np.ndarray:
        raise ValueError(error_message)

    try:
        new_array = np.array(obs)
    except ValueError:
        raise ValueError(error_message)

    return new_array


def inmb_u(d_effect, d_cost):
    """ higher d_effect represents better health """
    return lambda w: w*d_effect - d_cost


def inmb_d(d_effect, d_cost):
    """ higher d_effect represents worse health """
    return lambda w: -w * d_effect - d_cost


def inmb2_u(d_effect, d_cost):
    return lambda w_gain, w_loss: \
        inmb_u(d_effect, d_cost)(w_gain) if d_effect >= 0 else inmb_u(d_effect, d_cost)(w_loss)


def inmb2_d(d_effect, d_cost):
    return lambda w_gain, w_loss: \
        inmb_d(d_effect, d_cost)(w_gain) if d_effect >= 0 else inmb_d(d_effect, d_cost)(w_loss)


def get_d_cost(strategy):
    # this is defined for sorting strategies
    return strategy.dCost.get_mean()


def get_d_effect(strategy):
    # this is defined for sorting strategies
    return strategy.dEffect.get_mean()


def get_index(strategy):
    # this is defined for sorting strategies
    return strategy.idx


def find_intersecting_wtp(w0, u_new, u_base):

    if u_new(w0) > u_base(w0):
        return None

    else:
        f = lambda w: (u_new(w)-u_base(w))**2
        res = minimize(f, (w0))
        w_star = res.x[0]

        if abs(u_new(w_star)-u_base(w_star)) > 0.01:
            return None

        if w_star > w0:
            return w_star
        else:
            return None


def utility_sample_stat(utility, d_cost_samples, d_effect_samples,
                        wtp_random_variate, n_samples, rnd):

    discrete_rnd = RVG.UniformDiscrete(
        l=0, u=len(d_cost_samples)-1)

    samples = []
    for i in range(n_samples):
        j = discrete_rnd.sample(rnd)

        u = utility(d_effect=d_effect_samples[j],
                    d_cost=d_cost_samples[j])

        w = wtp_random_variate.sample(rnd)
        samples.append(u(w))

    return Stat.SummaryStat(name='', data=samples)


class _Curve:
    def __init__(self, label, color, wtps):
        self.label = label
        self.color = color
        self.wtps = wtps

        # range of wtp values over which this curve has the highest value
        self.rangeWTPHighestValue = [None, None]

        # this is to make sure that whenever the range of wtp values over which
        # this curve hast the highest value gets updated, the new wtp value
        # is larger than the last recorded wtp value.
        self.lastWTPValue = 0

    def update_range_with_highest_value(self, wtp):
        """
        updates the range of wtp values over which this curve has the highest value
        :param wtp: the new wtp value
        """

        # check if the wtp values are increasing
        if wtp < self.lastWTPValue:
            raise ValueError('Recorded WTP values should be increasing.')
        else:
            self.lastWTPValue = wtp

        # if the lower range is not recorded, use this wtp to determine the lower bound
        if self.rangeWTPHighestValue[0] is None:
            self.rangeWTPHighestValue[0] = wtp

        # update the upper range
        self.rangeWTPHighestValue[1] = wtp


class INMBCurve(_Curve):
    # incremental net monetary benefit curve of one strategy
    def __init__(self, label, color, wtps, inmb_stat, interval_type='n'):
        """
        :param label: (string) label of this incremental NMB curve
        :param color: color code of this curve
        :param wtps: wtp values over which this curve should be calcualted
        :param inmb_stat: incremental NMB statistics
        :param interval_type: (string) 'n' for no interval
                                       'c' for t-based confidence interval,
                                       'p' for percentile interval
        """

        _Curve.__init__(self, label, color, wtps)
        self.ys = []        # expected net monetary benefits over a range of wtp values
        self.l_errs = []    # lower error length of NMB over a range of wtp values
        self.u_errs = []    # upper error length of NMB over a range of wtp values

        self._calculate_ys_lerrs_uerrs(inmb_stat=inmb_stat, wtps=wtps, interval_type=interval_type)

    def _calculate_ys_lerrs_uerrs(self, inmb_stat, wtps, interval_type):
        """
        calculates the expected incremental NMB and the confidence or prediction intervals over the specified
        range of wtp values.
        :param inmb_stat: incremental NMB statistics
        :param wtps: list of wtp values
        :param interval_type: (string) 'n' for no interval
                                       'c' for t-based confidence interval,
                                       'p' for percentile interval
        """

        # get the NMB values for each wtp
        self.ys = np.array([inmb_stat.get_INMB(x) for x in wtps])

        if interval_type == 'c':
            y_intervals = np.array([inmb_stat.get_CI(x, alpha=0.05) for x in wtps])
        elif interval_type == 'p':
            y_intervals = np.array([inmb_stat.get_PI(x, alpha=0.05) for x in wtps])
        elif interval_type == 'n':
            y_intervals = None
        else:
            raise ValueError('Invalid value for internal_type.')

        # reshape confidence interval to plot
        if y_intervals is not None:
            self.u_errs = np.array([p[1] for p in y_intervals]) - self.ys
            self.l_errs = self.ys - np.array([p[0] for p in y_intervals])
        else:
            self.u_errs, self.l_errs = None, None


class AcceptabilityCurve(_Curve):
    # cost-effectiveness acceptability curve of one strategy
    def __init__(self, label, color, wtps):

        _Curve.__init__(self, label, color, wtps)
        self.probs = []     # probability that this strategy is optimal over a range of wtp values
        self.optWTPs = []   # wtp values over which this strategy has the highest expected net monetary benefit
        self.optProbs = []  # probabilities that correspond to optWTPs
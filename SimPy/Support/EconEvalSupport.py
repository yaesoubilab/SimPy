import numpy as np
from scipy.optimize import minimize

import SimPy.RandomVariateGenerators as RVG
import SimPy.Statistics as Stat


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
    def __init__(self, label, color, short_label):
        self.label = label
        self.shortLabel = label if short_label is None else short_label
        self.color = color
        self.xs = []
        self.ys = []
        self.optXs = []
        self.optYs = []

    def convert_lists_to_arrays(self):
        self.xs = np.array(self.xs)
        self.ys = np.array(self.ys)
        self.optXs = np.array(self.optXs)
        self.optYs = np.array(self.optYs)


class INMBCurve(_Curve):
    # incremental net monetary benefit curve of one strategy
    def __init__(self, label, color, wtp_values, inmb_stat, interval_type='n', short_label=None):
        """
        :param label: (string) label of this incremental NMB curve
        :param color: color code of this curve
        :param wtp_values: wtp values over which this curve should be calcualted
        :param inmb_stat: incremental NMB statistics
        :param interval_type: (string) 'n' for no interval
                                       'c' for t-based confidence interval,
                                       'p' for percentile interval
        :param short_label: (string) to display on the curves
        """

        _Curve.__init__(self=self, label=label, color=color, short_label=short_label)
        self.inmbStat = inmb_stat
        self.xs = wtp_values
        self.intervalType = interval_type
        self.l_errs = []    # lower error length of NMB over a range of wtp values
        self.u_errs = []    # upper error length of NMB over a range of wtp values

        self._calculate_ys_lerrs_uerrs()

    def _calculate_ys_lerrs_uerrs(self):
        """
        calculates the expected incremental NMB and the confidence or prediction intervals over the specified
        range of wtp values.
        """

        # get the NMB values for each wtp
        self.ys = [self.inmbStat.get_INMB(x) for x in self.xs]

        if self.intervalType == 'c':
            y_intervals = [self.inmbStat.get_CI(x, alpha=0.05) for x in self.xs]
        elif self.intervalType == 'p':
            y_intervals = [self.inmbStat.get_PI(x, alpha=0.05) for x in self.xs]
        elif self.intervalType == 'n':
            y_intervals = None
        else:
            raise ValueError('Invalid value for internal_type.')

        # reshape confidence interval to plot
        if y_intervals is not None:
            self.u_errs = np.array([p[1] for p in y_intervals]) - self.ys
            self.l_errs = self.ys - np.array([p[0] for p in y_intervals])
        else:
            self.u_errs, self.l_errs = None, None
    
    def get_switch_wtp(self):
        
        return self.inmbStat.get_switch_wtp()
    
    def get_switch_wtp_and_interval(self):

        return self.inmbStat.get_switch_wtp_and_interval(
            wtp_range=[self.xs[0], self.xs[-1]],
            interval_type=self.intervalType)    


class AcceptabilityCurve(_Curve):
    # cost-effectiveness acceptability curve of one strategy
    def __init__(self, label, color, short_label=None):

        _Curve.__init__(self, label=label, color=color, short_label=short_label)


class ExpHealthCurve(_Curve):

    def __init__(self, label, color, effect_stat, interval_type='n', short_label=None):
        """
        :param label: (string) label of this incremental NMB curve
        :param color: color code of this curve
        :param effect_stat: statistics of health outcomes
        :param interval_type: (string) 'n' for no interval
                                       'c' for t-based confidence interval,
                                       'p' for percentile interval
        :param short_label: (string) to display on the curves
        """

        _Curve.__init__(self, label=label, color=color, short_label=short_label)

        self.dEffectMean = effect_stat.get_mean()

        if interval_type == 'c':
            interval = effect_stat.get_t_CI(alpha=0.05)
        elif interval_type == 'p':
            interval = effect_stat.get_PI(alpha=0.05)
        elif interval_type == 'n':
            interval = None
        else:
            raise ValueError('Invalid value for internal_type.')

        if interval:
            self.l_err = effect_stat.get_mean() - interval[0]
            self.u_err = interval[1] - effect_stat.get_mean()
        else:
            self.l_err = None
            self.u_err = None

        self.l_errs = []  # lower error length of health outcome over a range of budget values
        self.u_errs = []  # upper error length of health outcome over a range of budget values

    def update_feasibility(self, b):
        self.xs.append(b)
        self.ys.append(self.dEffectMean)
        self.l_errs.append(self.l_err)
        self.u_errs.append(self.u_err)




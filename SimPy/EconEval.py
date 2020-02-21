import string
import warnings
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import SimPy.InOutFunctions as IO
import SimPy.FormatFunctions as F
from SimPy.Support.EconEvalSupport import *
from SimPy.Support.SupportClasses import *
import matplotlib.patches as patches


NUM_OF_BOOTSTRAPS = 1000  # number of bootstrap samples to calculate confidence intervals for ICER
LEGEND_FONT_SIZE = 7.5


def pv_single_payment(payment, discount_rate, discount_period, discount_continuously=False):
    """ calculates the present value of a single future payment
    :param payment: payment to calculate the present value for
    :param discount_rate: discount rate
    :param discount_period: number of periods to discount the payment
    :param discount_continuously: set to True to discount continuously
    :return: payment/(1+discount_rate)^discount_period for discrete discounting
             payment * exp(-discounted_rate*discount_period) for continuous discounting """

    # error checking
    if discount_continuously:
        pass
    else:
        assert type(discount_period) is int, "discount_period should be an integer number."
    if discount_rate < 0 or discount_rate > 1:
        raise ValueError("discount_rate should be a number between 0 and 1.")
    if discount_period <= 0:
        raise ValueError("discount_period should be greater than 0.")

    # calculate the present value
    if discount_continuously:
        return payment * np.exp(-discount_rate * discount_period)
    else:
        return payment * pow(1 + discount_rate, -discount_period)


def pv_continuous_payment(payment, discount_rate, discount_period):
    """ calculates the present value of a future continuous payment (discounted continuously)
    :param payment: payment to calculate the present value for
    :param discount_rate: discount rate
    :param discount_period: (tuple) in the form of (l, u) specifying the period where
                             the continuous payment is received
    :return: payment * (exp(-discount_rate*l - exp(-discount_rate*u))/discount_rate
    """
    assert type(discount_period) is tuple, "discount_period should be a tuple (l, u)."
    if discount_rate < 0 or discount_rate > 1:
        raise ValueError("discount_rate should be a number between 0 and 1.")

    if discount_rate == 0:
        return payment * (discount_period[1] - discount_period[0])
    else:
        return payment/discount_rate * \
               (np.exp(-discount_rate*discount_period[0])
                - np.exp(-discount_rate*discount_period[1]))


def equivalent_annual_value(present_value, discount_rate, discount_period):
    """  calculates the equivalent annual value of a present value
    :param present_value:
    :param discount_rate: discount rate (per period)
    :param discount_period: number of periods to discount the payment
    :return: discount_rate*present_value/(1-pow(1+discount_rate, -discount_period))
    """

    # error checking
    assert type(discount_period) is int, "discount_period should be an integer number."
    if discount_rate < 0 or discount_rate > 1:
        raise ValueError("discount_rate should be a number between 0 and 1.")
    if discount_period < 0:
        raise ValueError("discount_period cannot be less than 0.")

    # calculate the equivalent annual value
    return discount_rate*present_value/(1-pow(1+discount_rate, -discount_period))


class Strategy:
    def __init__(self, name, cost_obs, effect_obs, color=None, marker='o'):
        """
        :param name: name of the strategy
        :param cost_obs: list or numpy.array of cost observations
        :param effect_obs: list or numpy.array of effect observations
        :param color: (string) color code
                (https://www.webucator.com/blog/2015/03/python-color-constants-module/)
        :param marker: (string) marker code
                (https://matplotlib.org/3.1.1/api/markers_api.html)
        """

        assert color is None or type(color) is str, "color argument should be a string."

        self.idx = 0        # index of the strategy
        self.name = name
        self.color = color
        self.marker = marker

        self.ifDominated = False
        self.switchingWTP = 0
        self.switchingWTPInterval = []

        self.costObs = assert_np_list(cost_obs,
                                      error_message='cost_obs should be a list or a np.array')
        self.dCostObs = None    # (list) cost observations with respect to base
        self.incCostObs = None  # (list) incremental cost observations
        self.cost = None        # summary statistics for cost
        self.dCost = None       # summary statistics for cost with respect to base
        self.incCost = None     # summary statistics for incremental cost

        self.effectObs = assert_np_list(effect_obs,
                                        error_message='effect_obs should be a list or a np.array')
        self.dEffectObs = None      # (list) effect observations with respect to base
        self.incEffectObs = None    # (list) incremental effect observations
        self.effect = None          # summary statistics for effect
        self.dEffect = None         # summary statistics for effect with respect to base
        self.incEffect = None       # summary statistics for incremental effect

        self.cer = None         # cost-effectiveness ratio with respect to base
        self.icer = None        # icer summary statistics
        self.eIncNMB = None     # summary statistics for expected incremental net health benefit
                                # integrated over a wtp distribution

        self.cost = Stat.SummaryStat(name='Cost of '+name, data=self.costObs)
        self.effect = Stat.SummaryStat(name='Effect of '+name, data=self.effectObs)

    def reset(self):
        """ set class attributes that will be calculated later to None """

        self.ifDominated = False
        self.dCostObs = None
        self.incCostObs = None
        self.dCost = None
        self.incCost = None

        self.dEffectObs = None
        self.incEffectObs = None
        self.dEffect = None
        self.incEffect = None

        self.cer = None
        self.icer = None

    def get_cost_err_interval(self, interval_type, alpha=0.05, multiplier=1):
        """
        :param interval_type: (string) 'c' for t-based confidence interval,
                                       'cb' for bootstrap confidence interval, and
                                       'p' for percentile interval
        :param alpha: significance level
        :param multiplier: to multiply the estimate and the interval by the provided value
        :return: list [err_l, err_u] for the lower and upper error length
                of confidence or prediction intervals of cost observations.
                NOTE: [err_l, err_u] = [mean - L, mean + U], where [L, U] is the confidence or prediction interval

        """
        interval = self.cost.get_interval(interval_type, alpha, multiplier)
        return [self.cost.get_mean()*multiplier - interval[0],
                interval[1] - self.cost.get_mean()*multiplier]

    def get_effect_err_interval(self, interval_type, alpha, multiplier=1):
        """
        :param interval_type: (string) 'c' for t-based confidence interval,
                                       'cb' for bootstrap confidence interval, and
                                       'p' for percentile interval
        :param alpha: significance level
        :param multiplier: to multiply the estimate and the interval by the provided value
        :return: list [err_l, err_u] for the lower and upper error length
                of confidence or prediction intervals of effect observations.
                NOTE: [err_l, err_u] = [mean - L, mean + U], where [L, U] is the confidence or prediction interval

        """
        interval = self.effect.get_interval(interval_type, alpha, multiplier)
        return [self.effect.get_mean()*multiplier - interval[0],
                interval[1] - self.effect.get_mean()*multiplier]


class _EconEval:
    """ master class for cost-effective analysis (CEA) and cost-benefit analysis (CBA) """

    def __init__(self, strategies, if_paired, health_measure='u', if_reset_strategies=False):
        """
        :param strategies: the list of strategies (assumes that the first strategy represents the "base" strategy)
        :param if_paired: set to true to indicate that the strategies are paired
        :param health_measure: (string) choose 'u' if higher "effect" implies better health
        (e.g. when QALY is used) and set to 'd' if higher "effect" implies worse health
        (e.g. when DALYS is used)
        :param if_reset_strategies: set to True if the cost and effect with respect to
            base, incremental cost and effect, ICER, and CER of strategies should be recalculated.
        """

        if health_measure not in ['u', 'd']:
            raise ValueError("health_measure can be either 'u' (for utility) or 'd' (for disutility).")

        self.strategies = strategies  # list of strategies
        # assign the index of each strategy
        for i, s in enumerate(strategies):
            s.idx = i
            if if_reset_strategies:
                s.reset()

        self._n = len(strategies)  # number of strategies
        self._ifPaired = if_paired  # if cost and effect outcomes are paired across strategies
        self._healthMeasure = health_measure  # utility of disutility
        self._u_or_d = 1 if health_measure == 'u' else -1

        # assign colors to strategies
        self.__assign_colors()

    def __assign_colors(self):

        # decide about the color of each curve
        rainbow_colors = cm.rainbow(np.linspace(0, 1, self._n))
        for i, s in enumerate(self.strategies):
            if s.color is None:
                s.color = rainbow_colors[i]

    def _find_shifted_strategies(self):
        """ find shifted strategies.
        In calculating the change in effect, it accounts for whether QALY or DALY is used.
        """

        # shift all strategies such that the base strategy (first in the list) lies on the origin
        # if observations are paired across strategies
        if self._ifPaired:

            for i, s in enumerate(self.strategies):

                s.dCostObs = s.costObs - self.strategies[0].costObs
                s.dCost = Stat.DifferenceStatPaired(name='Cost with respect to base',
                                                    x=s.costObs,
                                                    y_ref=self.strategies[0].costObs)
                # if health measure is utility
                if self._healthMeasure == 'u':
                    s.dEffectObs = s.effectObs - self.strategies[0].effectObs
                    s.dEffect = Stat.DifferenceStatPaired(name='Effect with respect to base',
                                                          x=s.effectObs,
                                                          y_ref=self.strategies[0].effectObs)

                else:  # if health measure is disutility
                    s.dEffectObs = self.strategies[0].effectObs - s.effectObs
                    s.dEffect = Stat.DifferenceStatPaired(name='Effect with respect to base',
                                                          x=self.strategies[0].effectObs,
                                                          y_ref=s.effectObs)
                # cost-effectiveness ratio
                if i > 0:
                    s.cer = Stat.RatioStatPaired(name='Cost-effectiveness ratio of ' + s.name,
                                                 x=s.dCostObs,
                                                 y_ref=s.dEffectObs)

        else:  # if not paired
            # get average cost and effect of the base strategy
            base_ave_cost = self.strategies[0].cost.get_mean()
            base_ave_effect = self.strategies[0].effect.get_mean()

            for i, s in enumerate(self.strategies):
                s.dCostObs = s.costObs - base_ave_cost
                s.dCost = Stat.DifferenceStatIndp(name='Cost with respect to base',
                                                  x=s.costObs,
                                                  y_ref=self.strategies[0].costObs)
                if self._healthMeasure == 'u':
                    s.dEffectObs = s.effectObs - base_ave_effect
                    s.dEffect = Stat.DifferenceStatIndp(name='Effect with respect to base',
                                                        x=s.effectObs,
                                                        y_ref=self.strategies[0].effectObs)

                else:  # if health measure is disutility
                    s.dEffectObs = base_ave_effect - s.effectObs
                    s.dEffect = Stat.DifferenceStatIndp(name='Effect with respect to base',
                                                        x=self.strategies[0].effectObs,
                                                        y_ref=s.effectObs)

                # cost-effectiveness ratio
                if i > 0:
                    s.cer = Stat.RatioStatIndp(name='Cost-effectiveness ratio',
                                               x=s.dCostObs,
                                               y_ref=s.dEffectObs)

    @staticmethod
    def _format_ax(ax,
                   y_range,
                   min_x, max_x, delta_x=None,
                   delta_y=None, if_y_axis_prob=True,
                   if_format_y_numbers=True, y_axis_decimal=1):

        # format x-axis
        if delta_x is None:
            vals_x = ax.get_xticks()
        else:
            vals_x = []
            x = min_x
            while x <= max_x:
                vals_x.append(x)
                x += delta_x

        # format y-axis
        if y_range is not None:
            ax.set_ylim(y_range)

        if delta_y is None:
            vals_y = ax.get_yticks()
        else:
            vals_y = []
            y = y_range[0]
            while y <= y_range[1]:
                vals_y.append(y)
                y += delta_y

        ax.set_xticks(vals_x)
        ax.set_xticklabels(['{:,.{prec}f}'.format(x, prec=0) for x in vals_x])

        d = 2 * (max_x - min_x) / 200
        ax.set_xlim([min_x - d, max_x + d])



        ax.set_yticks(vals_y)
        if if_y_axis_prob:
            ax.set_yticklabels(['{:.{prec}f}'.format(x, prec=1) for x in vals_y])
        elif if_format_y_numbers:
            ax.set_yticklabels(['{:,.{prec}f}'.format(x, prec=y_axis_decimal) for x in vals_y])

        if if_y_axis_prob and y_range is None:
            ax.set_ylim((-0.01, 1.01))

        if not if_y_axis_prob:
            ax.axhline(y=0, c='k', ls='--', linewidth=0.5)

    def _add_curves_to_ax(self, ax, curves, x_values, title, x_label, y_label, y_range=None,
                          y_axis_multiplier=1, y_axis_decimal=1,
                          delta_x=None,
                          transparency_lines=0.4,
                          transparency_intervals=0.2,
                          show_legend=False,
                          show_frontier=True,
                          if_y_axis_prob=False,
                          if_format_y_numbers=True):

        for curve in curves:
            # plot line
            ax.plot(curve.xs, curve.ys * y_axis_multiplier,
                    c=curve.color, alpha=transparency_lines, label=curve.label)
            # plot intervals
            if curve.l_errs is not None and curve.u_errs is not None:
                ax.fill_between(curve.xs,
                                (curve.ys - curve.l_errs) * y_axis_multiplier,
                                (curve.ys + curve.u_errs) * y_axis_multiplier,
                                color=curve.color, alpha=transparency_intervals)
            # plot frontier
            if show_frontier:
                # check if this strategy is not dominated
                if curve.optXs is not None:
                    ax.plot(curve.optXs, curve.optYs * y_axis_multiplier,
                            c=curve.color, alpha=1, linewidth=4)

        if show_legend:
            ax.legend(loc=2, fontsize=LEGEND_FONT_SIZE)

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_ylim(y_range)

        # do the other formatting
        self._format_ax(ax=ax, y_range=y_range,
                        min_x=x_values[0], max_x=x_values[-1], delta_x=delta_x,
                        if_y_axis_prob=if_y_axis_prob,
                        if_format_y_numbers=if_format_y_numbers,
                        y_axis_decimal=y_axis_decimal)


class CEA(_EconEval):
    """ master class for cost-effective analysis (CEA) and cost-benefit analysis (CBA) """

    def __init__(self, strategies, if_paired, health_measure='u', if_reset_strategies=False):
        """
        :param strategies: the list of strategies (assumes that the first strategy represents the "base" strategy)
        :param if_paired: set to true to indicate that the strategies are paired
        :param health_measure: (string) choose 'u' if higher "effect" implies better health
        (e.g. when QALY is used) and set to 'd' if higher "effect" implies worse health
        (e.g. when DALYS is used)
        :param if_reset_strategies: set to True if the cost and effect with respect to
            base, incremental cost and effect, ICER, and CER of strategies should be recalculated.
        """

        _EconEval.__init__(self, strategies=strategies,
                           if_paired=if_paired,
                           health_measure=health_measure,
                           if_reset_strategies=if_reset_strategies)

        self._strategies_on_frontier = []   # list of strategies on the frontier
        self._ifFrontierIsCalculated = False  # CE frontier is not calculated yet
        self._ifPairwiseCEAsAreCalculated = False
        self._pairwise_ceas = []  # list of list to cea's

        # shift the strategies
        self._find_shifted_strategies()

        # find the cost-effectiveness frontier
        self.__find_frontier()

    def get_strategies_on_frontier(self):
        """
        :return: strategies on the frontier sorted in the increasing order of effect
        """

        if not self._ifFrontierIsCalculated:
            self.__find_frontier()

        return self._strategies_on_frontier
    
    def get_strategies_not_on_frontier(self):
        """
        :return: strategies not on the frontier 
        """
        
        if not self._ifFrontierIsCalculated:
            self.__find_frontier()
        
        return [s for s in self.strategies if s.ifDominated]

    def build_CE_table(self,
                       interval_type='n',
                       alpha=0.05,
                       cost_digits=0, effect_digits=2, icer_digits=1,
                       cost_multiplier=1, effect_multiplier=1,
                       file_name='myCSV.csv', directory=''):
        """
        :param interval_type: (string) 'n' for no interval,
                                       'c' for confidence interval,
                                       'p' for percentile interval
        :param alpha: significance level
        :param cost_digits: digits to round cost estimates to
        :param effect_digits: digits to round effect estimate to
        :param icer_digits: digits to round ICER estimates to
        :param cost_multiplier: set to 1/1000 or 1/100,000 to represent cost in terms of
                thousands or hundred thousands unit
        :param effect_multiplier: set to 1/1000 or 1/100,000 to represent effect in terms of
                thousands or hundred thousands unit
        :param file_name: address and file name where the CEA results should be saved to
        :param directory: directory (relative to the current root) where the files should be located
            for example use 'Example' to create and save the csv file under the folder Example
        """

        # find the frontier if not calculated already
        if not self._ifFrontierIsCalculated:
            self.__find_frontier()

        table = [['Strategy', 'Cost', 'Effect', 'Incremental Cost', 'Incremental Effect', 'ICER']]
        # sort strategies in increasing order of cost
        self.strategies.sort(key=get_d_cost)

        for i, s in enumerate(self.strategies):
            row=[]
            # strategy name
            row.append(s.name)
            # strategy cost
            row.append(s.cost.get_formatted_mean_and_interval(interval_type=interval_type,
                                                              alpha=alpha,
                                                              deci=cost_digits,
                                                              form=',',
                                                              multiplier=cost_multiplier))
            # strategy effect
            row.append(s.effect.get_formatted_mean_and_interval(interval_type=interval_type,
                                                                alpha=alpha,
                                                                deci=effect_digits,
                                                                form=',',
                                                                multiplier=effect_multiplier))

            # strategy incremental cost
            if s.incCost is None:
                row.append('-')
            else:
                row.append(s.incCost.get_formatted_mean_and_interval(interval_type=interval_type,
                                                                     alpha=alpha,
                                                                     deci=cost_digits,
                                                                     form=',',
                                                                     multiplier=cost_multiplier))
            # strategy incremental effect
            if s.incEffect is None:
                row.append('-')
            else:
                row.append(s.incEffect.get_formatted_mean_and_interval(interval_type=interval_type,
                                                                       alpha=alpha,
                                                                       deci=effect_digits,
                                                                       form=',',
                                                                       multiplier=effect_multiplier))

            # ICER
            if s.ifDominated:
                row.append('Dominated')
            elif s.icer is not None:
                row.append(s.icer.get_formatted_ICER_and_interval(interval_type=interval_type,
                                                                  alpha=alpha,
                                                                  deci=icer_digits,
                                                                  form=',',
                                                                  multiplier=1,
                                                                  num_bootstrap_samples=NUM_OF_BOOTSTRAPS))
            else:
                row.append('-')

            table.append(row)

        IO.write_csv(file_name=file_name, directory=directory, rows=table, delimiter=',')

        # sort strategies back
        self.strategies.sort(key=get_index)

    def get_dCost_dEffect_cer(self,
                              interval_type='n',
                              alpha=0.05,
                              cost_digits=0, effect_digits=2, icer_digits=1,
                              cost_multiplier=1, effect_multiplier=1):
        """
        :param interval_type: (string) 'n' for no interval,
                                       'c' for confidence interval,
                                       'p' for percentile interval
        :param alpha: significance level
        :param cost_digits: digits to round cost estimates to
        :param effect_digits: digits to round effect estimate to
        :param icer_digits: digits to round ICER estimates to
        :param cost_multiplier: set to 1/1000 or 1/100,000 to represent cost in terms of
                thousands or hundred thousands unit
        :param effect_multiplier: set to 1/1000 or 1/100,000 to represent effect in terms of
                thousands or hundred thousands unit
        :return: a dictionary of additional cost, additional effect, and cost-effectiveness ratio for
                all strategies with respect to the Base strategy
        """

        dictionary_results = {}

        for s in [s for s in self.strategies if s.idx > 0]:

            d_cost_text = s.dCost.get_formatted_mean_and_interval(interval_type=interval_type,
                                                                  alpha=alpha,
                                                                  deci=cost_digits,
                                                                  form=',',
                                                                  multiplier=cost_multiplier)
            d_effect_text = s.dEffect.get_formatted_mean_and_interval(interval_type=interval_type,
                                                                      alpha=alpha,
                                                                      deci=effect_digits,
                                                                      form=',',
                                                                      multiplier=effect_multiplier)
            cer_text = s.cer.get_formatted_mean_and_interval(interval_type=interval_type,
                                                             alpha=alpha,
                                                             deci=icer_digits,
                                                             form=',',
                                                             multiplier=1)
            # add to the dictionary
            dictionary_results[s.name] = [d_cost_text, d_effect_text, cer_text]

        return dictionary_results

    def add_ce_plane_to_ax(self, ax,
                           x_range=None, y_range=None,
                           add_clouds=True, show_legend=True,
                           center_s=50, cloud_s=10, transparency=0.1,
                           cost_multiplier=1, effect_multiplier=1,
                           cost_digits=0, effect_digits=1):

        # find the frontier (x, y)'s
        frontier_d_effect = []
        frontier_d_costs = []
        for s in self.get_strategies_on_frontier():
            frontier_d_effect.append(s.dEffect.get_mean()*effect_multiplier)
            frontier_d_costs.append(s.dCost.get_mean()*cost_multiplier)

        # add all strategies
        for s in self.strategies:
            # the mean change in effect and cost
            ax.scatter(s.dEffect.get_mean()*effect_multiplier, s.dCost.get_mean()*cost_multiplier,
                       c=s.color,  # color
                       alpha=1,  # transparency
                       marker=s.marker,  # markers
                       s=center_s,  # marker size
                       label=s.name,  # name to show in the legend
                       zorder=2,
                       #edgecolors='k'
                       )

        # add the frontier line
        ax.plot(frontier_d_effect, frontier_d_costs,
                c='k',  # color
                alpha=0.6,  # transparency
                linewidth=2,  # line width
                zorder=3,
                label='Frontier', # label to show in the legend
                )

        if show_legend:
            ax.legend(fontsize='7.5')

        # and the clouds
        if add_clouds:
            # add all strategies
            for s in self.strategies:
                ax.scatter(s.dEffectObs * effect_multiplier, s.dCostObs * cost_multiplier,
                           c=s.color,  # color of dots
                           marker=s.marker, # marker
                           alpha=transparency,  # transparency of dots
                           s=cloud_s,  # size of dots
                           zorder=1
                           )

        ax.set_xlim(x_range)  # x-axis range
        ax.set_ylim(y_range)  # y-axis range

        # format x-axis
        vals_x = ax.get_xticks()
        ax.set_xticks(vals_x)
        ax.set_xticklabels(['{:,.{prec}f}'.format(x, prec=effect_digits) for x in vals_x])

        # format y-axis
        vals_y = ax.get_yticks()
        ax.set_yticks(vals_y)
        ax.set_yticklabels(['{:,.{prec}f}'.format(x, prec=cost_digits) for x in vals_y])

        ax.axhline(y=0, c='k', linestyle='--', linewidth=0.5)
        ax.axvline(x=0, c='k', linestyle='--', linewidth=0.5)

    def plot_CE_plane(self,
                      title='Cost-Effectiveness Analysis',
                      x_label='Additional Health',
                      y_label='Additional Cost',
                      x_range=None, y_range=None,
                      add_clouds=True, fig_size=(5, 5),
                      show_legend=True,
                      center_s=75, cloud_s=25, transparency=0.1,
                      cost_multiplier=1, effect_multiplier=1,
                      cost_digits=0, effect_digits=1,
                      file_name='CE.png'
                      ):

        fig, ax = plt.subplots(figsize=fig_size)

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # add the cost-effectiveness plane
        self.add_ce_plane_to_ax(ax=ax,
                                x_range=x_range, y_range=y_range,
                                add_clouds=add_clouds,
                                show_legend=show_legend,
                                center_s=center_s, cloud_s=cloud_s, transparency=transparency,
                                cost_multiplier=cost_multiplier, effect_multiplier=effect_multiplier,
                                cost_digits=cost_digits, effect_digits=effect_digits)

        fig.show()
        if file_name is not None:
            fig.savefig(file_name, dpi=300)

    def create_pairwise_ceas(self):
        """
        creates a list of list for pairwise cost-effectiveness analysis
        For example for strategies ['Base', 'A', 'B']:
        [
            ['Base wr Base',    'A wr Base',    'B wr Base'],
            ['Base wr A',       'A wr A',       'B wr A'],
            ['B wr B',          'A wr B',       'B wr B'],
        ]
        """

        # create CEA's for all pairs
        self._pairwise_ceas = []
        for s_base in self.strategies:
            list_ceas = []
            for s_new in self.strategies:#[1:]:

                # if the base and the new strategies are the same
                if s_base.name == s_new.name:
                    list_ceas.append(None)
                else:
                    list_ceas.append(CEA(strategies=[s_base, s_new],
                                         if_paired=self._ifPaired,
                                         health_measure=self._healthMeasure,
                                         if_reset_strategies=True)
                                     )
            self._pairwise_ceas.append(list_ceas)

        self._ifPairwiseCEAsAreCalculated = True

        # since the relative performance of strategies
        # (relative costs and relative effects) have changed.
        self._ifFrontierIsCalculated = False

    def print_pairwise_cea(self, interval_type='n',
                           alpha=0.05,
                           cost_digits=0, effect_digits=2, icer_digits=1,
                           cost_multiplier=1, effect_multiplier=1,
                           directory='Pairwise_CEA'):
        """
        :param interval_type: (string) 'n' for no interval,
                                       'c' for confidence interval,
                                       'p' for percentile interval
        :param alpha: significance level
        :param cost_digits: digits to round cost estimates to
        :param effect_digits: digits to round effect estimate to
        :param icer_digits: digits to round ICER estimates to
        :param cost_multiplier: set to 1/1000 or 1/100,000 to represent cost in terms of
                thousands or hundred thousands unit
        :param effect_multiplier: set to 1/1000 or 1/100,000 to represent effect in terms of
                thousands or hundred thousands unit
        :param directory: directory (relative to the current root) where the files should be located
            for example use 'Example' to create and save the csv file under the folder Example
        """

        # create the pair-wise cost-effectiveness analyses
        if not self._ifPairwiseCEAsAreCalculated:
            self.create_pairwise_ceas()

        # save the CEA tables
        for row_of_ceas in self._pairwise_ceas:
            for cea in row_of_ceas:
                if cea is not None:
                    name = cea.strategies[1].name + ' to ' + cea.strategies[0].name
                    cea.build_CE_table(interval_type=interval_type,
                                       alpha=alpha,
                                       cost_digits=cost_digits,
                                       effect_digits=effect_digits,
                                       icer_digits=icer_digits,
                                       cost_multiplier=cost_multiplier,
                                       effect_multiplier=effect_multiplier,
                                       file_name=name+'.csv',
                                       directory=directory)

    def plot_pairwise_ceas(self,
                           figure_size=None, font_size=6,
                           show_subplot_labels=False,
                           effect_label='', cost_label='',
                           center_s=50, cloud_s=25, transparency=0.2,
                           x_range=None, y_range=None,
                           cost_multiplier=1, effect_multiplier=1,
                           column_titles=None, row_titles=None,
                           file_name='pairwise_CEA.png'):

        # identify which CEA is valid
        # (i.e. comparing strategies that are on the frontier)
        # valid comparisons are marked with '*'
        valid_comparison = []
        for i in range(self._n):
            valid_comparison.append(['']*self._n)
        on_frontier = self.get_strategies_on_frontier()
        for idx in range(len(on_frontier)-1):
            i = on_frontier[idx].idx
            j = on_frontier[idx+1].idx
            valid_comparison[i][j] = '*'

        # set default properties of the figure
        plt.rc('font', size=font_size) # fontsize of texts
        plt.rc('axes', titlesize=font_size)  # fontsize of the figure title
        plt.rc('axes', titleweight='semibold')  # fontweight of the figure title

        # plot each panel
        f, axarr = plt.subplots(nrows=self._n, ncols=self._n,
                                sharex=True, sharey=True, figsize=figure_size)

        Y_LABEL_COORD_X = -0.05     # increase to move the A-B-C labels to right
        abc_idx = 0
        for i in range(self._n):
            for j in range(self._n):
                # get the current axis
                ax = axarr[i, j]

                # add the A-B-C label if needed
                if show_subplot_labels:
                    ax.text(Y_LABEL_COORD_X - 0.05, 1.05,
                            string.ascii_uppercase[abc_idx] + ')',
                            transform=ax.transAxes,
                            size=font_size + 1, weight='bold')

                # add titles for the figures_national in the first row
                if i == 0:
                    if column_titles is None:
                        ax.set_title(self.strategies[j].name)
                    else:
                        ax.set_title(column_titles[j])

                # add y_labels for the figures_national in the first column
                if j == 0:
                    if row_titles is None:
                        ax.set_ylabel(self.strategies[i].name, fontweight='bold')
                    else:
                        ax.set_ylabel(row_titles[i], fontweight='bold')

                # specify ranges of x- and y-axis
                if x_range is not None:
                    ax.set_xlim(x_range)
                if y_range is not None:
                    ax.set_ylim(y_range)

                # CEA of these 2 strategies
                cea = CEA(strategies=[self.strategies[i], self.strategies[j]],
                          if_paired=self._ifPaired,
                          health_measure=self._healthMeasure,
                          if_reset_strategies=True)

                # add the CE figure to this axis
                cea.add_ce_plane_to_ax(ax=ax, show_legend=False,
                                       center_s=center_s,
                                       cloud_s=cloud_s,
                                       transparency=transparency,
                                       cost_multiplier=cost_multiplier,
                                       effect_multiplier=effect_multiplier)

                # add ICER to the figure
                # could be 'Dominated', 'Cost-Saving' or 'estimated ICER'
                text = ''
                if i != j and cea.strategies[1].ifDominated:
                    text = 'Dominated'
                elif cea.strategies[1].dCost.get_mean() < 0 and cea.strategies[1].dEffect.get_mean() > 0:
                    text = 'Cost-Saving' + valid_comparison[i][j]
                elif cea.strategies[1].icer is not None:
                    text = F.format_number(cea.strategies[1].icer.get_ICER(), deci=1, format='$') + valid_comparison[i][j]
                # add the text of the ICER to the figure
                ax.text(0.95, 0.95, text, transform=ax.transAxes, fontsize=6,
                        va='top', ha='right')

                abc_idx += 1

        # add the common effect and cost labels
        f.text(0.55, 0, effect_label, ha='center', va='center', fontweight='bold')
        f.text(0.99, 0.5, cost_label, va='center', rotation=-90, fontweight='bold')

        # f.show()
        f.savefig(file_name, bbox_inches='tight', dpi=300)

        # since the relative performance of strategies 
        # (relative costs and relative effects) have changed.
        self._ifFrontierIsCalculated = False



    def __find_frontier(self):

        # apply criteria 1 (strict dominance)
        # if a strategy i yields less health than strategy j but costs more, it is dominated
        # sort by effect with respect to base
        self.strategies.sort(key=get_d_effect)
        for i in range(self._n):
            for j in range(i+1, self._n):
                if self.strategies[i].dCost.get_mean() >= self.strategies[j].dCost.get_mean():
                    self.strategies[i].ifDominated = True
                    break;

        # select all non-dominated strategies
        select_strategies = [s for s in self.strategies if not s.ifDominated]

        # apply criteria 1 (strict dominance)
        # if a strategy i costs more than strategy j but yields less health, it is dominated
        # sort strategies by cost with respect to the base
        select_strategies.sort(key=get_d_cost, reverse=True)
        for i in range(len(select_strategies)):
            for j in range(i + 1, len(select_strategies)):
                if select_strategies[i].dEffect.get_mean() <= select_strategies[j].dEffect.get_mean():
                    select_strategies[i].ifDominated = True
                    break;

        # apply criteria 2 (weak dominance)
        # select all non-dominated strategies
        select_strategies = [s for s in self.strategies if not s.ifDominated]

        for i in range(len(select_strategies)):
            for j in range(i+1, len(select_strategies)):
                # cost and effect of strategy i
                d_cost_i = select_strategies[i].dCost.get_mean()
                d_effect_i = select_strategies[i].dEffect.get_mean()
                # cost and effect of strategy j
                d_cost_j = select_strategies[j].dCost.get_mean()
                d_effect_j = select_strategies[j].dEffect.get_mean()
                # vector connecting strategy i to j
                v_i_to_j = np.array([d_effect_j - d_effect_i, d_cost_j - d_cost_i])

                # find strategies with dEffect between i and j
                s_between_i_and_j = []
                for s in select_strategies:
                    if d_effect_i < s.dEffect.get_mean() < d_effect_j:
                        s_between_i_and_j.append(s)

                # if the dEffect of no strategy is between the effects of strategies i and j
                if len(s_between_i_and_j) == 0:
                    continue  # to the next j
                else:

                    for inner_s in s_between_i_and_j:
                        # vector from i to inner_s
                        v_i_to_inner = np.array([inner_s.dEffect.get_mean() - d_effect_i,
                                                 inner_s.dCost.get_mean() - d_cost_i])

                        # cross products of vector i to j and the vectors i to the inner point
                        cross_product = v_i_to_j[0] * v_i_to_inner[1] - v_i_to_j[1] * v_i_to_inner[0]

                        # if cross_product > 0 the point is above the line
                        # (because the point are sorted vertically)
                        # ref: How to tell whether a point is to the right or left side of a line
                        # https://stackoverflow.com/questions/1560492
                        if cross_product > 0:
                            inner_s.ifDominated = True

        # sort strategies by effect with respect to the base
        self.strategies.sort(key=get_d_effect)

        # find strategies on the frontier
        self._strategies_on_frontier = [s for s in self.strategies if not s.ifDominated]

        # sort back
        self.strategies.sort(key=get_index)

        # frontier is calculated
        self._ifFrontierIsCalculated = True

        # calcualte the incremental outcomes
        self.__calculate_incremental_outcomes()

    def __calculate_incremental_outcomes(self):

        if self._ifPaired:

            for i, s in enumerate(self._strategies_on_frontier):
                if i > 0:
                    s_before = self._strategies_on_frontier[i-1]

                    s.incCostObs = s.costObs - s_before.costObs
                    s.incCost = Stat.DifferenceStatPaired(name='Incremental cost',
                                                          x=s.costObs,
                                                          y_ref=s_before.costObs)
                    # if health measure is utility
                    if self._healthMeasure == 'u':
                        s.incEffectObs = s.effectObs - self._strategies_on_frontier[i-1].effectObs
                        s.incEffect = Stat.DifferenceStatPaired(name='Effect with respect to base',
                                                                x=s.effectObs,
                                                                y_ref=s_before.effectObs)

                    else:  # if health measure is disutility
                        s.incEffectObs = self._strategies_on_frontier[i-1].effectObs - s.effectObs
                        s.incEffect = Stat.DifferenceStatPaired(name='Effect with respect to base',
                                                                x=s_before.effectObs,
                                                                y_ref=s.effectObs)
                    # ICER
                    s.icer = ICER_Paired(name='ICER of {} relative to {}'.format(s.name, s_before.name),
                                         costs_new=s.costObs,
                                         effects_new=s.effectObs,
                                         costs_base=s_before.costObs,
                                         effects_base=s_before.effectObs,
                                         health_measure=self._healthMeasure)

        else:  # if not paired

            for i, s in enumerate(self._strategies_on_frontier):

                if i > 0:
                    s_before = self._strategies_on_frontier[i - 1]

                    # get average cost and effect of the strategy i - 1
                    ave_cost_i_1 =s_before.cost.get_mean()
                    ave_effect_i_1 = s_before.effect.get_mean()

                    s.incCostObs = s.costObs - ave_cost_i_1
                    s.incCost = Stat.DifferenceStatIndp(name='Cost with respect to base',
                                                        x=s.costObs,
                                                        y_ref=s_before.costObs)
                    if self._healthMeasure == 'u':
                        s.incEffectObs = s.effectObs - ave_effect_i_1
                        s.incEffect = Stat.DifferenceStatIndp(name='Effect with respect to base',
                                                              x=s.effectObs,
                                                              y_ref=s_before.effectObs)

                    else:  # if health measure is disutility
                        s.incEffectObs = ave_effect_i_1 - s.effectObs
                        s.incEffect = Stat.DifferenceStatIndp(name='Effect with respect to base',
                                                              x=s_before.effectObs,
                                                              y_ref=s.effectObs)

                    # ICER
                    s.icer = ICER_Indp(name='ICER of {} relative to {}'.format(s.name, s_before.name),
                                       costs_new=s.costObs,
                                       effects_new=s.effectObs,
                                       costs_base=s_before.costObs,
                                       effects_base=s_before.effectObs,
                                       health_measure=self._healthMeasure)


class CBA(_EconEval):
    """ class for doing cost-benefit analysis """

    def __init__(self, strategies, wtp_range, if_paired, health_measure='u', n_of_wtp_values=200):
        """
        :param strategies: the list of strategies (assumes that the first strategy represents the "base" strategy)
        :param wtp_range: ([l, u]) range of willingness-to-pay values over which the NMB analysis should be done
        :param if_paired: indicate whether the strategies are paired
        :param health_measure: (string) choose 'u' if higher "effect" implies better health
        (e.g. when QALY is used) and set to 'd' if higher "effect" implies worse health
        (e.g. when DALYS is used)
        :param n_of_wtp_values: number of willingness-to-pay values to construct net monetary benefit curves
        """
        _EconEval.__init__(self, strategies=strategies,
                           if_paired=if_paired,
                           health_measure=health_measure)

        self.inmbCurves = []  # list of incremental NMB curves with respect to the base
        self.acceptabilityCurves = []  # the list of acceptability curves

        # use net monetary benefit for utility by default
        if health_measure == 'u':
            self.utility = inmb_u
        else:
            self.utility = inmb_d

        # wtp values (includes the specified minimum and maximum wtp value)
        self.wtp_values = np.linspace(wtp_range[0], wtp_range[1],
                                      num=n_of_wtp_values, endpoint=True)

        # index of strategy with the highest
        # expected net-monetary benefit over the wtp range
        self.idxHighestExpNMB = []

    def build_inmb_curves(self, interval_type='n'):
        """
        prepares the information needed to plot the incremental net-monetary benefit
        with respect to the first strategy (base)
        :param interval_type: (string) 'n' for no interval,
                                       'c' for confidence interval,
                                       'p' for percentile interval):
        """

        self.inmbCurves = []  # list of incremental NMB curves

        # create the NMB curves
        for s in self.strategies:

            if self._ifPaired:
                # create a paired NMB object
                inmb = INMB_Paired(name=s.name,
                                   costs_new=s.costObs,
                                   effects_new=s.effectObs,
                                   costs_base=self.strategies[0].costObs,
                                   effects_base=self.strategies[0].effectObs,
                                   health_measure=self._healthMeasure)

            else:
                # create an independent NMB object
                inmb = INMB_Indp(name=s.name,
                                 costs_new=s.costObs,
                                 effects_new=s.effectObs,
                                 costs_base=self.strategies[0].costObs,
                                 effects_base=self.strategies[0].effectObs,
                                 health_measure=self._healthMeasure)

            # make a NMB curve
            self.inmbCurves.append(INMBCurve(label=s.name,
                                             color=s.color,
                                             wtp_values=self.wtp_values,
                                             inmb_stat=inmb,
                                             interval_type=interval_type)
                                   )

        self.__find_strategies_with_highest_einmb()

    def build_acceptability_curves(self):
        """
        prepares the information needed to plot the cost-effectiveness acceptability curves
        """

        if not self._ifPaired:
            raise ValueError('Calculating the acceptability curves when outcomes are not paired'
                             'across strategies is not implemented.')

        # initialize acceptability curves
        self.acceptabilityCurves = []
        for s in self.strategies:
            self.acceptabilityCurves.append(AcceptabilityCurve(label=s.name,
                                                               color=s.color,
                                                               wtp_values=self.wtp_values))

        n_obs = len(self.strategies[0].costObs)

        # for each WTP value, calculate the number of times that
        # each strategy has the highest NMB value
        for w in self.wtp_values:

            # number of times that each strategy is optimal
            count_maximum = np.zeros(self._n)

            for obs_idx in range(n_obs):

                # find which strategy has the maximum:
                max_nmb = float('-inf')
                max_s_i = 0  # index of the optimal strategy for this observation
                for s_i, s in enumerate(self.strategies):
                    d_effect = (s.effectObs[obs_idx] - self.strategies[0].effectObs[obs_idx]) * self._u_or_d
                    d_cost = s.costObs[obs_idx] - self.strategies[0].costObs[obs_idx]
                    nmb = w * d_effect - d_cost
                    if nmb > max_nmb:
                        max_nmb = nmb
                        max_s_i = s_i

                count_maximum[max_s_i] += 1

            # calculate probabilities that each strategy has been optimal
            prob_maximum = count_maximum / n_obs

            # max_prob = 0
            # max_s_i = 0
            for i in range(self._n):
                self.acceptabilityCurves[i].xs.append(w)
                self.acceptabilityCurves[i].ys.append(prob_maximum[i])
                #
                # if prob_maximum[i] > max_prob:
                #     max_prob = prob_maximum[i]
                #     max_s_i = i

        if len(self.idxHighestExpNMB) == 0:
            self.__find_strategies_with_highest_einmb()

        # find the optimal strategy for each wtp value
        for wtp_idx, wtp in enumerate(self.wtp_values):
            opt_idx = self.idxHighestExpNMB[wtp_idx]
            self.acceptabilityCurves[opt_idx].optXs.append(wtp)
            self.acceptabilityCurves[opt_idx].optYs.append(
                self.acceptabilityCurves[opt_idx].ys[wtp_idx])

        for c in self.acceptabilityCurves:
            c.convert_lists_to_arrays()

    # def get_wtp_ranges_with_highest_exp_nmb(self):
    #     """
    #     :return: dictionary (with strategy names as keys) of wtp ranges over which
    #     each strategy has the highest expected incremental net monetary benefit.
    #     """
    #
    #     dict = {}
    #     for curve in self.inmbCurves:
    #         dict[curve.label] = curve.rangeWTPHighestValue
    #     return dict
    #
    # def get_wtp_range_with_highest_prob_of_optimal(self):
    #     """
    #     :return: dictionary (with strategy names as keys) of wtp ranges over which
    #     each strategy has the highest probability of being optimal.
    #     """
    #
    #     dict = {}
    #     for curve in self.acceptabilityCurves:
    #         dict[curve.label] = curve.rangeWTPHighestValue
    #     return dict

    def __find_strategies_with_highest_einmb(self):
        """ find strategies with the highest expected incremental net monetary benefit.
            It records the indices of these strategies over the range of wtp values
         """

        # find the optimal strategy for each wtp value
        for wtp_idx, wtp in enumerate(self.wtp_values):

            max_value = float('-inf')
            max_idx = 0
            for s_idx in range(len(self.inmbCurves)):
                if self.inmbCurves[s_idx].ys[wtp_idx] > max_value:
                    max_value = self.inmbCurves[s_idx].ys[wtp_idx]
                    max_idx = s_idx

            # store the index of the optimal strategy
            self.idxHighestExpNMB.append(max_idx)
            # self.inmbCurves[max_idx].update_range_with_highest_value(x=wtp)
            self.inmbCurves[max_idx].optXs.append(wtp)
            self.inmbCurves[max_idx].optYs.append(max_value)

        for curve in self.inmbCurves:
            curve.convert_lists_to_arrays()

    def find_optimal_switching_wtp_values(self, interval_type='n', deci=0):

        w_stars = []  # wtp values to switch between strategies
        w_star_intervals = [] # confidence or projection intervals of optimal wtp values
        s_stars = []  # indices of optimal strategies between wtp values
        s_star_names = []   # names of optimal strategies

        # working wtp value
        w = self.wtp_values[0]

        # find the optimal strategy at wtp = 0
        max_nmb = float('-inf')
        s_star = 0
        for s in self.strategies:
            u = self.utility(d_effect=s.dEffect.get_mean(),
                             d_cost=s.dCost.get_mean())
            u_value = u(w)
            if u_value > max_nmb:
                max_nmb = u_value
                s_star = s.idx

        # record the information of the optimal strategy at wtp = 0
        w_stars.append(w)
        w_star_intervals.append([w, w])
        s_stars.append(s_star)
        s_star_names.append(self.strategies[s_star].name)

        # find the optimal switching wtp values
        while w <= self.wtp_values[-1] and len(s_stars) < len(self.strategies):

            # find the intersect of the current strategy with other
            w_min = float('inf')
            for s in self.strategies:
                if s.idx not in s_stars:

                    line = Line(x1=self.strategies[s_star].dEffect.get_mean(),
                                x2=s.dEffect.get_mean(),
                                y1=self.strategies[s_star].dCost.get_mean(),
                                y2=s.dCost.get_mean())
                    w_star = line.slope

                    if w_star is not math.nan and w_star >= w:
                        if w_star < w_min:
                            w_min = w_star
                            s_star = s.idx

            w = w_min
            if w != float('inf'):
                w_stars.append(w)
                s_stars.append(s_star)
                s_star_names.append(self.strategies[s_star].name)

        # find and store the confidence or projection intervals of
        # switching wtp values
        for i, s_index in enumerate(s_stars[0:-1]):
            next_s_index = s_stars[i+1]
            if self._ifPaired:
                inmb = INMB_Paired(
                    name='',
                    costs_new=self.strategies[next_s_index].costObs,
                    effects_new=self.strategies[next_s_index].effectObs,
                    costs_base=self.strategies[s_index].costObs,
                    effects_base=self.strategies[s_index].effectObs,
                    health_measure=self._healthMeasure
                )
            else:
                inmb = INMB_Indp(
                    name='',
                    costs_new=self.strategies[next_s_index].costObs,
                    effects_new=self.strategies[next_s_index].effectObs,
                    costs_base=self.strategies[s_index].costObs,
                    effects_base=self.strategies[s_index].effectObs,
                    health_measure=self._healthMeasure
                )

            w_star, interval = inmb.get_switch_wtp_and_interval(
                wtp_range=[self.wtp_values[0], self.wtp_values[-1]],
                interval_type=interval_type
            )
            w_star_intervals.append(interval)

        # update status of strategies
        i = 0
        for s in self.strategies:
            if s.idx in s_stars:
                s.ifDominated = False
                s.switchingWTP = w_stars[i]
                s.switchingWTPInterval = w_star_intervals[i]
                i += 1
            else:
                s.ifDominated = True

        # populate the results to report back
        result = [['Strategy', 'ID', 'WTP', 'Interval']]
        for i in range(len(s_stars)):
            result.append(
                [
                    s_star_names[i],
                    s_stars[i],
                    F.format_number(w_stars[i], deci=deci, format=','),
                    F.format_interval(w_star_intervals[i], deci=deci, format=',')
                ]
            )

        return result

    def plot_incremental_nmbs(self,
                              title='Incremental Net Monetary Benefit',
                              x_label='Willingness-To-Pay Threshold',
                              y_label='Expected Incremental Net Monetary Benefit',
                              y_range=None,
                              y_axis_multiplier=1,
                              y_axis_decimal=1,
                              interval_type='c',
                              delta_wtp=None,
                              transparency_lines=0.5,
                              transparency_intervals=0.2,
                              show_legend=True,
                              figure_size=(5, 5),
                              file_name=None):
        """
        plots the incremental net-monetary benefit of each strategy
                with respect to the base (the first strategy)
        :param title: title
        :param x_label: x-axis label
        :param y_label: y-axis label
        :param y_range: (list) range of y-axis
        :param y_axis_multiplier: (float) multiplier to scale the y-axis
            (e.g. 0.001 for thousands)
        :param y_axis_decimal: (float) decimal of y_axis numbers
        :param interval_type: (string) 'n' for no interval,
                                       'c' for confidence interval,
                                       'p' for percentile interval
        :param delta_wtp: distance between the labels of WTP values shown on the x-axis
        :param transparency_lines: transparency of net monetary benefit lines (0.0 transparent through 1.0 opaque)
        :param transparency_intervals: transparency of intervals (0.0 transparent through 1.0 opaque)
        :param show_legend: set true to show legend
        :param figure_size: (tuple) size of the figure (e.g. (2, 3)
        :param file_name: (string) filename to save the figure as
        """

        # make incremental NMB curves
        self.build_inmb_curves(interval_type=interval_type)

        # initialize plot
        fig, ax = plt.subplots(figsize=figure_size)

        # add the incremental NMB curves
        self._add_curves_to_ax(ax=ax, curves=self.inmbCurves, x_values=self.wtp_values,
                               title=title, x_label=x_label,
                               y_label=y_label, y_range=y_range, delta_x=delta_wtp,
                               y_axis_multiplier=y_axis_multiplier,
                               y_axis_decimal=y_axis_decimal,
                               transparency_lines=transparency_lines,
                               transparency_intervals=transparency_intervals,
                               show_legend=show_legend)

        fig.tight_layout()

        if file_name is None:
            fig.show()
        else:
            fig.savefig(file_name, dpi=300)

    def add_inmb_curves_to_ax(self, ax, title, x_label, y_label, y_range=None,
                              y_axis_multiplier=1, delta_wtp=None,
                              transparency_lines=0.4,
                              transparency_intervals=0.2,
                              show_legend=True,
                              show_frontier=True):

        self._add_curves_to_ax(ax=ax, curves=self.inmbCurves, x_values=self.wtp_values,
                               title=title, x_label=x_label,
                               y_label=y_label, y_range=y_range, delta_x=delta_wtp,
                               y_axis_multiplier=y_axis_multiplier,
                               transparency_lines=transparency_lines,
                               transparency_intervals=transparency_intervals,
                               show_legend=show_legend,
                               show_frontier=show_frontier)

    def plot_acceptability_curves(self,
                                  title=None,
                                  x_label='Willingness-To-Pay Threshold',
                                  y_label='Probability of Being the Optimal Strategy',
                                  y_range=None,
                                  delta_wtp=None,
                                  show_legend=True, fig_size=(5, 5),
                                  legends=None,
                                  file_name=None):
        """
        plots the acceptibility curves
        :param title: title
        :param x_label: x-axis label
        :param y_label: y-axis label
        :param y_range: (list) range of y-axis
        :param delta_wtp: distance between the labels of WTP values shown on the x-axis
        :param show_legend: set true to show legend
        :param legends: (list) of legends to display on the figure
        :param fig_size: (tuple) size of the figure (e.g. (2, 3)
        :param file_name: file name
        """

        # make the NMB curves
        self.build_inmb_curves(interval_type='n')

        # make the acceptability curves
        self.build_acceptability_curves()

        # initialize plot
        fig, ax = plt.subplots(figsize=fig_size)

        # add the incremental NMB curves
        self.add_acceptability_curves_to_ax(ax=ax,
                                            wtp_delta=delta_wtp,
                                            y_range=y_range,
                                            show_legend=show_legend,
                                            legends=legends)

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        if file_name is None:
            fig.show()
        else:
            fig.savefig(file_name, bbox_inches='tight', dpi=300)

    def add_acceptability_curves_to_ax(self,
                                       ax, wtp_delta=None,
                                       y_range=None, show_legend=True, legends=None):

        for i, curve in enumerate(self.acceptabilityCurves):
            # plot line
            if legends is None:
                ax.plot(curve.xs, curve.ys, c=curve.color, alpha=1, label=curve.label)
            else:
                ax.plot(curve.xs, curve.ys, c=curve.color, alpha=1, label=legends[i])
            ax.plot(curve.optXs, curve.optYs, c=curve.color, linewidth=4)

        self._format_ax(ax=ax, y_range=y_range,
                        min_x=self.wtp_values[0],
                        max_x=self.wtp_values[-1],
                        delta_x=wtp_delta,
                        if_y_axis_prob=True)

        if show_legend:
            ax.legend(fontsize=LEGEND_FONT_SIZE)  # xx-small, x-small, small, medium, large, x-large, xx-large

    def get_w_starts(self):

        w_stars = []
        s_stars = []
        s_star_names = []

        w = self.wtp_values[0]

        # at initial w
        max_nmb = float('-inf')
        s_star = 0
        for s in self.strategies:
            u = self.utility(d_effect=s.dEffect.get_mean(),
                             d_cost=s.dCost.get_mean())
            u_value = u(w)
            if u_value > max_nmb:
                max_nmb = u_value
                s_star = s.idx

        w_stars.append(w)
        s_stars.append(s_star)
        s_star_names.append(self.strategies[s_star].name)

        while w <= self.wtp_values[-1] and len(s_stars) < len(self.strategies):

            # find the intersect of the current strategy with other
            w_min = float('inf')
            for s in self.strategies:
                if s.idx not in s_stars:

                    w_star = find_intersecting_wtp(
                        w0=w,
                        u_new=self.utility(d_effect=s.dEffect.get_mean(),
                                           d_cost=s.dCost.get_mean()),
                        u_base=self.utility(d_effect=self.strategies[s_star].dEffect.get_mean(),
                                            d_cost=self.strategies[s_star].dCost.get_mean()))

                    if w_star is not None:
                        if w_star < w_min:
                            w_min = w_star
                            s_star = s.idx

            w = w_min
            if w != float('inf'):
                w_stars.append(w)
                s_stars.append(s_star)
                s_star_names.append(self.strategies[s_star].name)

        return w_stars, s_star_names, s_stars

    def calculate_exp_incremental_nmbs(self, wtp_random_variate, n_samples, rnd):
        """ create summary statistics of incremental NMB of all strategies given the
        provided probability distribution of WTP value.
        :param wtp_random_variate: random variate generator for the probability distribution of wtp value
        :param n_samples: number of monte carlo samples
        :param rnd: the random number generator
        """

        for s in self.strategies[1:]:
            s.eIncNMB = utility_sample_stat(
                utility=self.utility,
                d_cost_samples=s.dCostObs,
                d_effect_samples=s.dEffectObs,
                wtp_random_variate=wtp_random_variate,
                n_samples=n_samples,
                rnd=rnd
            )

    def report_exp_incremental_nmb(self, interval='c', deci=0):

        report = []
        for s in self.strategies[1:]:

            mean_and_interval = s.eIncNMB.get_formatted_mean_and_interval(
                     interval_type=interval,
                     deci=deci,
                     form=','
                 )

            report.append([s.name, mean_and_interval])

        return report

    def plot_exp_incremental_nmb(self,
                                 title=None,
                                 y_label='Expected Net Monetary Benefit',
                                 y_range=None,
                                 figure_size=(5, 5),
                                 if_show_conf_interval=True,
                                 file_name=None):

        # initialize plot
        fig, ax = plt.subplots(figsize=figure_size)

        ax.set_title(title)
        ax.set_ylabel(y_label)

        for s in self.strategies[1:]:
            ax.scatter(x=s.idx, y=s.eIncNMB.get_mean(),
                       c=s.color, label=s.name)

            interval = s.eIncNMB.get_interval(interval_type='p')
            y = s.eIncNMB.get_mean()
            err = [[y-interval[0]], [interval[1]-y]]

            ax.errorbar(x=s.idx, y=y,
                        yerr=err,
                        c=s.color)

            if if_show_conf_interval:
                width = 0.1
                interval = s.eIncNMB.get_interval(interval_type='c')
                xy = [
                    [s.idx, interval[0]],
                    [s.idx - width, y],
                    [s.idx, interval[1]],
                    [s.idx + width, y]
                ]
                ax.add_patch(patches.Polygon(xy=xy, fill=False, edgecolor=s.color))

        ax.set_xticks([])
        ax.set_xlim([0.5, len(self.strategies)-0.5])
        ax.axhline(y=0, c='k', ls='--', lw=1)

        # format y-axis
        ax.set_ylim(y_range)
        vals_y = ax.get_yticks()
        ax.set_yticks(vals_y)
        ax.set_yticklabels(['{:,.{prec}f}'.format(x, prec=0) for x in vals_y])

        ax.legend()
        if file_name is None:
            fig.show()
        else:
            fig.savefig(file_name, bbox_inches='tight', dpi=300)


class HealthMaxSubjectToBudget(_EconEval):
    """ a class for selecting the alternative with the highest expected health outcomes
    subject to a budget constraint """

    def __init__(self, strategies, budget_range, if_paired, health_measure='u',
                 n_of_budget_values=200, epsilon=None):
        """
        :param strategies: the list of strategies (assumes that the first strategy represents the "base" strategy)
        :param budget_range: ([l, u]) range of budget values over which the analysis should be done
        :param if_paired: indicate whether the strategies are paired
        :param health_measure: (string) choose 'u' if higher "effect" implies better health
        (e.g. when QALY is used) and set to 'd' if higher "effect" implies worse health
        (e.g. when DALYS is used)
        :param n_of_budget_values: number of budget values to construct curves of optimal strategies
        :param epsilon: (float, 0 <= epsilon <= 1)
                        decision maker's tolerance in violating the budget constraint.
                        (i.e. epsilon in Prob{DCost_i > B} <= epsilon)
                        If set to None, this constraint will be considered: E[DCost_i] <= B.
        """
        _EconEval.__init__(self, strategies=strategies,
                           if_paired=if_paired,
                           health_measure=health_measure)

        # shift the strategies
        self._find_shifted_strategies()

        # list of cost values which the budget should be below
        # if epsilon = None, it takes the expected dCost of strategies,
        # otherwise, it takes the upper percentile of dCost of strategies.
        self.dCostUp = []
        # list of expected delta effect
        self.dEffect = []
        # list of expected effect curves
        self.effectCurves = []

        # determine budget values
        self.budget_values = np.linspace(budget_range[0], budget_range[1],
                                         num=n_of_budget_values, endpoint=True)

        # set up curves
        for s in strategies:
            if epsilon is None:
                self.dCostUp.append(s.dCost.get_mean())
            else:
                self.dCostUp.append(s.dCost.get_percentile(q=(1 - epsilon) * 100))
            self.dEffect.append(s.dEffect.get_mean())
            self.effectCurves.append(
                ExpHealthCurve(
                    label=s.name,
                    color=s.color,
                    effect_stat=s.dEffect,
                    interval_type='c')
            )

        for b in self.budget_values:
            max_effect = -float('inf')
            max_s_i = None
            for s_i, s in enumerate(self.strategies):
                # if this strategy is feasible
                if self.dCostUp[s_i] <= b:
                    self.effectCurves[s_i].update_feasibility(b=b)
                    if self.dEffect[s_i] > max_effect:
                        max_effect = self.dEffect[s_i]
                        max_s_i = s_i

            # self.effectCurves[max_s_i].update_range_with_highest_value(x=b)
            self.effectCurves[max_s_i].optXs.append(b)
            self.effectCurves[max_s_i].optYs.append(max_effect)

        # convert lists to arrays
        for c in self.effectCurves:
            c.convert_lists_to_arrays()

    def plot(self,
             title='Expected Increase in Effect',
             x_label='Budget',
             y_label='Expected Increase in Effect',
             y_range=None,
             y_axis_multiplier=1,
             delta_budget=None,
             transparency_lines=0.5,
             transparency_intervals=0.2,
             show_legend=True,
             figure_size=(5, 5),
             file_name='Budget.png'):

        # initialize plot
        fig, ax = plt.subplots(figsize=figure_size)

        # add plot to the ax
        self._add_curves_to_ax(ax=ax,
                               curves=self.effectCurves,
                               x_values=self.budget_values,
                               title=title, x_label=x_label,
                               y_label=y_label, y_range=y_range, delta_x=delta_budget,
                               y_axis_multiplier=y_axis_multiplier,
                               transparency_lines=transparency_lines,
                               transparency_intervals=transparency_intervals,
                               show_legend=show_legend,
                               if_format_y_numbers=False)

        fig.show()
        if file_name is not None:
            fig.savefig(file_name, dpi=300)

    def add_plot_to_ax(self, ax, delta_budget=None,
                       title=None, x_label=None, y_label=None,
                       y_range=None, y_axis_multiplier=1,
                       transparency_lines=0.4,
                       transparency_intervals=0.2,
                       show_legend=True,
                       show_frontier=True,
                       ):

        self._add_curves_to_ax(ax=ax,
                               curves=self.effectCurves,
                               x_values=self.budget_values,
                               delta_x=delta_budget,
                               title=title, x_label=x_label, y_label=y_label,
                               y_range=y_range, y_axis_multiplier=y_axis_multiplier,
                               transparency_lines=transparency_lines,
                               transparency_intervals=transparency_intervals,
                               show_legend=show_legend,
                               show_frontier=show_frontier,
                               if_format_y_numbers=False)


class _ComparativeEconMeasure:
    def __init__(self, name, costs_new, effects_new, costs_base, effects_base, health_measure='u'):
        """
        :param costs_new: (list or numpy.array) cost data for the new strategy
        :param effects_new: (list or numpy.array) effect data for the new strategy
        :param costs_base: (list or numpy.array) cost data for the base line
        :param effects_base: (list or numpy.array) effect data for the base line
        :param health_measure: (string) choose 'u' if higher "effect" implies better health
        (e.g. when QALY is used) and set to 'd' if higher "effect" implies worse health
        (e.g. when DALYS is used)
        """

        if health_measure not in ['u', 'd']:
            raise ValueError("health_measure can be either 'u' (for utility) or 'd' (for disutility).")

        self.name = name
        self._costsNew = costs_new  # cost data for the new strategy
        self._effectsNew = effects_new  # effect data for the new strategy
        self._costsBase = costs_base  # cost data for teh base line
        self._effectsBase = effects_base  # effect data for the base line
        # if QALY or DALY is being used
        self._effect_multiplier = 1 if health_measure == 'u' else -1

        # convert input data to numpy.array if needed
        self._costsNew = assert_np_list(costs_new, "cost_new should be list or np.array.")
        self._effectsNew = assert_np_list(effects_new, "effects_new should be list or np.array.")
        self._costsBase = assert_np_list(costs_base, "costs_base should be list or np.array.")
        self._effectsBase = assert_np_list(effects_base, "effects_base should be list or np.array.")

        # calculate the difference in average cost
        self._delta_ave_cost = np.average(self._costsNew) - np.average(self._costsBase)
        # change in effect: DALY averted or QALY gained
        self._delta_ave_effect = (np.average(self._effectsNew) - np.average(self._effectsBase)) \
                                 * self._effect_multiplier

    def get_ave_d_cost(self):
        """
        :return: average incremental cost
        """
        return self._delta_ave_cost

    def get_ave_d_effect(self):
        """
        :return: average incremental effect
        """
        return self._delta_ave_effect


class _ICER(_ComparativeEconMeasure):
    def __init__(self, name, costs_new, effects_new, costs_base, effects_base, health_measure='u'):
        """
        :param costs_new: (list or numpy.array) cost data for the new strategy
        :param effects_new: (list or numpy.array) effect data for the new strategy
        :param costs_base: (list or numpy.array) cost data for the base line
        :param effects_base: (list or numpy.array) effect data for the base line
        :param health_measure: (string) choose 'u' if higher "effect" implies better health
        (e.g. when QALY is used) and set to 'd' if higher "effect" implies worse health
        (e.g. when DALYS is used)
        """

        self._isDefined = True  # if ICER cannot be computed, this will change to False

        # initialize the base class
        _ComparativeEconMeasure.__init__(self, name, costs_new, effects_new, costs_base, effects_base, health_measure)

        # calculate ICER
        if self._delta_ave_effect == 0:
            warnings.warn(self.name + ': Mean incremental effect is 0. ICER is not computable.')
            self._isDefined = False
            self._ICER = math.nan
        else:
            # $ per DALY averted or $ per QALY gained
            self._ICER = self._delta_ave_cost / self._delta_ave_effect

    def get_ICER(self):
        """ return ICER """
        return self._ICER

    def get_CI(self, alpha=0.05, num_bootstrap_samples=1000, rng=None):
        """
        :param alpha: significance level, a value from [0, 1]
        :param num_bootstrap_samples: number of bootstrap samples
        :param rng: random number generator
        :return: confidence interval in the format of list [l, u]
        """
        # abstract method to be overridden in derived classes to process an event
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")

    def get_PI(self, alpha=0.05):
        """
        :param alpha: significance level, a value from [0, 1]
        :return: percentile interval in the format of list [l, u]
        """
        # abstract method to be overridden in derived classes to process an event
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")

    def get_formatted_ICER_and_interval(self, interval_type='c',
                                        alpha=0.05, deci=0, form=None,
                                        multiplier=1, num_bootstrap_samples=1000):
        """
        :param interval_type: (string) 'n' for no interval
                                       'c' or 'cb' for bootstrap confidence interval, and
                                       'p' for percentile interval
        :param alpha: significance level
        :param deci: digits to round the numbers to
        :param form: ',' to format as number, '%' to format as percentage, and '$' to format as currency
        :param multiplier: to multiply the estimate and the interval by the provided value
        :param num_bootstrap_samples: number of bootstrap samples to calculate confidence interval
        :return: (string) estimate of ICER and interval formatted as specified
        """

        estimate = self.get_ICER() * multiplier

        if interval_type == 'c' or interval_type == 'cb':
            interval = self.get_CI(alpha=alpha, num_bootstrap_samples=num_bootstrap_samples)
        elif interval_type == 'p':
            interval = self.get_PI(alpha=alpha)
        else:
            interval = None

        adj_interval = [v * multiplier for v in interval] if interval is not None else None

        return F.format_estimate_interval(estimate=estimate,
                                          interval=adj_interval,
                                          deci=deci,
                                          format=form)


class ICER_Paired(_ICER):

    def __init__(self, name, costs_new, effects_new, costs_base, effects_base, health_measure='u'):
        """
        :param costs_new: (list or numpy.array) cost data for the new strategy
        :param effects_new: (list or numpy.array) health data for the new strategy
        :param costs_base: (list or numpy.array) cost data for the base line
        :param effects_base: (list or numpy.array) health data for the base line
        :param health_measure: (string) choose 'u' if higher "effect" implies better health
        (e.g. when QALY is used) and set to 'd' if higher "effect" implies worse health
        (e.g. when DALYS is used)
        """

        # all cost and effects should have the same length
        if not (len(costs_new) == len(effects_new) == len(costs_base) == len(effects_base)):
            raise ValueError('Paired ICER assume the same number of observations for all cost and health outcomes.')

        # initialize the base class
        _ICER.__init__(self, name, costs_new, effects_new, costs_base, effects_base, health_measure)

        # incremental observations
        self._deltaCosts = self._costsNew - self._costsBase
        self._deltaEffects = (self._effectsNew - self._effectsBase) * self._effect_multiplier

        # check if ICER is computable
        if min(self._deltaEffects) < 0:
            self._isDefined = False
            warnings.warn('\nFor "' + name + '" one of ICERs is not computable'
                                             '\nbecause at least one incremental effect is negative.')

        # calculate ICERs
        if self._isDefined:
            self._icers = np.divide(self._deltaCosts, self._deltaEffects)

    def get_CI(self, alpha=0.05, num_bootstrap_samples=1000, rng=None):
        """
        :param alpha: significance level, a value from [0, 1]
        :param num_bootstrap_samples: number of bootstrap samples
        :param rng: random number generator
        :return: confidence interval in the format of list [l, u]
        """

        if not self._isDefined:
            return [math.nan, math.nan]

        # create a new random number generator if one is not provided.
        if rng is None:
            rng = RVG.RNG(seed=1)
        assert type(rng) is RVG.RNG, "rng should be of type RandomVariateGenerators.RNG"

        # check if ICER is computable
        if min(self._deltaEffects) == 0 and max(self._deltaEffects) == 0:
            warnings.warn(self.name + ': Incremental health observations are 0, ICER is not computable')
            return [math.nan, math.nan]
        else:
            # bootstrap algorithm
            icer_bootstrap_means = np.zeros(num_bootstrap_samples)
            for i in range(num_bootstrap_samples):
                # because cost and health are paired as one observation,
                # so do delta cost and delta health, should sample them together
                indices = rng.choice(range(len(self._deltaCosts)), size=len(self._deltaCosts), replace=True)
                sampled_delta_costs = self._deltaCosts[indices]
                sampled_delta_effects = self._deltaEffects[indices]

                ave_delta_cost = np.average(sampled_delta_costs)
                ave_delta_effect = np.average(sampled_delta_effects)

                # assert all the means should not be 0
                if np.average(ave_delta_effect) == 0:
                    warnings.warn(
                        self.name + ': Mean incremental health is 0 for one bootstrap sample, ICER is not computable')

                icer_bootstrap_means[i] = ave_delta_cost / ave_delta_effect - self._ICER

        # return the bootstrap interval
        return self._ICER - np.percentile(icer_bootstrap_means, [100 * (1 - alpha / 2.0), 100 * alpha / 2.0])

    def get_PI(self, alpha=0.05):
        """
        :param alpha: significance level, a value from [0, 1]
        :return: prediction interval in the format of list [l, u]
        """
        if not self._isDefined:
            return [math.nan, math.nan]

        return np.percentile(self._icers, [100 * alpha / 2.0, 100 * (1 - alpha / 2.0)])


class ICER_Indp(_ICER):

    def __init__(self, name, costs_new, effects_new, costs_base, effects_base, health_measure='u'):
        """
        :param costs_new: (list or numpy.array) cost data for the new strategy
        :param effects_new: (list or numpy.array) health data for the new strategy
        :param costs_base: (list or numpy.array) cost data for the base line
        :param effects_base: (list or numpy.array) health data for the base line
        :param health_measure: (string) choose 'u' if higher "effect" implies better health
        (e.g. when QALY is used) and set to 'd' if higher "effect" implies worse health
        (e.g. when DALYS is used)
        """

        # all cost and effects should have the same length for each alternative
        if not (len(costs_new) == len(effects_new) and len(costs_base) == len(effects_base)):
            raise ValueError(
                'ICER assume the same number of observations for the cost and health outcome of each alternative.')

        # initialize the base class
        _ICER.__init__(self, name, costs_new, effects_new, costs_base, effects_base, health_measure)

    def get_CI(self, alpha=0.05, num_bootstrap_samples=1000, rng=None):
        """
        :param alpha: significance level, a value from [0, 1]
        :param num_bootstrap_samples: number of bootstrap samples
        :param rng: random number generator
        :return: bootstrap confidence interval in the format of list [l, u]
        """

        # create a new random number generator if one is not provided.
        if rng is None:
            rng = RVG.RNG(seed=1)
        assert type(rng) is RVG.RNG, "rng should be of type RandomVariateGenerators.RNG"

        # vector to store bootstrap ICERs
        icer_bootstrap_means = np.zeros(num_bootstrap_samples)

        n_obs_new = len(self._costsNew)
        n_obs_base = len(self._costsBase)

        # get bootstrap samples
        for i in range(num_bootstrap_samples):
            # for the new alternative
            indices_new = rng.choice(range(n_obs_new), size=n_obs_new, replace=True)
            costs_new = self._costsNew[indices_new]
            effects_new = self._effectsNew[indices_new]

            # for the base alternative
            indices_base = np.random.choice(range(n_obs_base), size=n_obs_base, replace=True)
            costs_base = self._costsBase[indices_base]
            effects_base = self._effectsBase[indices_base]

            # calculate this bootstrap ICER
            mean_costs_new = np.mean(costs_new)
            mean_costs_base = np.mean(costs_base)
            mean_effects_new = np.mean(effects_new)
            mean_effects_base = np.mean(effects_base)

            # calculate this bootstrap ICER
            if (mean_effects_new - mean_effects_base) * self._effect_multiplier <= 0:
                self._isDefined = False
                warnings.warn('\nConfidence intervals for one of ICERs is not computable'
                              '\nbecause at least one of bootstrap incremental effect is negative.'
                              '\nIncreasing the number of cost and effect observations might resolve the issue.')
                break

            else:
                icer_bootstrap_means[i] = \
                    (mean_costs_new - mean_costs_base) / (mean_effects_new - mean_effects_base) \
                    * self._effect_multiplier

        if self._isDefined:
            return np.percentile(icer_bootstrap_means, [100 * alpha / 2.0, 100 * (1 - alpha / 2.0)])
        else:
            return [math.nan, math.nan]

    def get_PI(self, alpha=0.05, num_bootstrap_samples=1000, rng=None):
        """
        :param alpha: significance level, a value from [0, 1]
        :param num_bootstrap_samples: number of bootstrap samples
        :param rng: random number generator
        :return: prediction interval in the format of list [l, u]
        """

        # create a new random number generator if one is not provided.
        if rng is None:
            rng = RVG.RNG(seed=1)
        assert type(rng) is RVG.RNG, "rng should be of type RandomVariateGenerators.RNG"

        if num_bootstrap_samples == 0:
            num_bootstrap_samples = max(len(self._costsNew), len(self._costsBase))

        # calculate element-wise ratio as sample of ICER
        indices_new = rng.choice(range(num_bootstrap_samples), size=num_bootstrap_samples, replace=True)
        costs_new = self._costsNew[indices_new]
        effects_new = self._effectsNew[indices_new]

        indices_base = rng.choice(range(num_bootstrap_samples), size=num_bootstrap_samples, replace=True)
        costs_base = self._costsBase[indices_base]
        effects_base = self._effectsBase[indices_base]

        if min((effects_new - effects_base) * self._effect_multiplier) <= 0:
            self._isDefined = False
        else:
            sample_icers = np.divide(
                (costs_new - costs_base),
                (effects_new - effects_base) * self._effect_multiplier)

        if self._isDefined:
            return np.percentile(sample_icers, [100 * alpha / 2.0, 100 * (1 - alpha / 2.0)])
        else:
            return [math.nan, math.nan]


class _INMB(_ComparativeEconMeasure):
    # incremental net monetary benefit
    def __init__(self, name, costs_new, effects_new, costs_base, effects_base, health_measure='u'):
        """
        :param costs_new: (list or numpy.array) cost data for the new strategy
        :param effects_new: (list or numpy.array) effect data for the new strategy
        :param costs_base: (list or numpy.array) cost data for the base line
        :param effects_base: (list or numpy.array) effect data for the base line
        :param health_measure: (string) choose 'u' if higher "effect" implies better health
        (e.g. when QALY is used) and set to 'd' if higher "effect" implies worse health
        (e.g. when DALYS is used)
        """
        # initialize the base class
        _ComparativeEconMeasure.__init__(self, name, costs_new, effects_new, costs_base, effects_base, health_measure)

    def get_INMB(self, wtp):
        """
        :param wtp: willingness-to-pay ($ for QALY gained or $ for DALY averted)
        :returns: the incremental net monetary benefit at the provided willingness-to-pay value
        """
        return wtp * self._delta_ave_effect - self._delta_ave_cost

    def get_CI(self, wtp, alpha=0.05):
        """
        :param wtp: willingness-to-pay value ($ for QALY gained or $ for DALY averted)
        :param alpha: significance level, a value from [0, 1]
        :return: confidence interval in the format of list [l, u]
        """
        # abstract method to be overridden in derived classes
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")

    def get_PI(self, wtp, alpha=0.05):
        """
        :param wtp: willingness-to-pay value ($ for QALY gained or $ for DALY averted)
        :param alpha: significance level, a value from [0, 1]
        :return: percentile interval in the format of list [l, u]
        """
        # abstract method to be overridden in derived classes
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")

    def get_switch_wtp(self):

        try:
            wtp = self.get_ave_d_cost() / self.get_ave_d_effect()
        except ValueError:
            wtp = math.nan

        return wtp

    def get_switch_wtp_and_interval(self, wtp_range, interval_type='n'):

        wtp = self.get_switch_wtp()

        if interval_type == 'n':
            return wtp, None
        elif interval_type == 'c':
            interval_at_min_wtp = self.get_CI(wtp=wtp_range[0])
            interval_at_max_wtp = self.get_CI(wtp=wtp_range[1])
        elif interval_type == 'p':
            interval_at_min_wtp = self.get_PI(wtp=wtp_range[0])
            interval_at_max_wtp = self.get_PI(wtp=wtp_range[1])
        else:
            raise ValueError('Invalid value for interval_type.')

        line_lower_err = S.Line(x1=wtp_range[0],
                                x2=wtp_range[1],
                                y1=interval_at_min_wtp[0],
                                y2=interval_at_max_wtp[0])
        line_upper_err = S.Line(x1=wtp_range[0],
                                x2=wtp_range[1],
                                y1=interval_at_min_wtp[1],
                                y2=interval_at_max_wtp[1])

        if self.get_ave_d_effect() >= 0:
            interval = [line_upper_err.get_intercept_with_x_axis(),
                        line_lower_err.get_intercept_with_x_axis()]
        else:
            interval = [line_lower_err.get_intercept_with_x_axis(),
                        line_upper_err.get_intercept_with_x_axis()]

        return wtp, interval


class INMB_Paired(_INMB):

    def __init__(self, name, costs_new, effects_new, costs_base, effects_base, health_measure='u'):
        """
        :param costs_new: (list or numpy.array) cost data for the new strategy
        :param effects_new: (list or numpy.array) health data for the new strategy
        :param costs_base: (list or numpy.array) cost data for the base line
        :param effects_base: (list or numpy.array) health data for the base line
        :param health_measure: (string) choose 'u' if higher "effect" implies better health
        (e.g. when QALY is used) and set to 'd' if higher "effect" implies worse health
        (e.g. when DALYS is used)
        """
        _INMB.__init__(self, name, costs_new, effects_new, costs_base, effects_base, health_measure)

        # incremental observations
        self._deltaCost = self._costsNew - self._costsBase
        self._deltaHealth = (self._effectsNew - self._effectsBase) * self._effect_multiplier

    def get_CI(self, wtp, alpha=0.05):
        # create a summary statistics
        stat = Stat.SummaryStat(self.name, wtp * self._deltaHealth - self._deltaCost)
        return stat.get_t_CI(alpha)

    def get_PI(self, wtp, alpha=0.05):
        # create a summary statistics
        stat = Stat.SummaryStat(self.name, wtp * self._deltaHealth - self._deltaCost)
        return stat.get_PI(alpha)


class INMB_Indp(_INMB):

    def __init__(self, name, costs_new, effects_new, costs_base, effects_base, health_measure='u'):
        """
        :param costs_new: (list or numpy.array) cost data for the new strategy
        :param effects_new: (list or numpy.array) health data for the new strategy
        :param costs_base: (list or numpy.array) cost data for the base line
        :param effects_base: (list or numpy.array) health data for the base line
        :param health_measure: (string) choose 'u' if higher "effect" implies better health
        (e.g. when QALY is used) and set to 'd' if higher "effect" implies worse health
        (e.g. when DALYS is used)
        """
        _INMB.__init__(self, name, costs_new, effects_new, costs_base, effects_base, health_measure)

    def get_CI(self, wtp, alpha=0.05):
        # NMB observations of two alternatives
        stat_new = wtp * self._effectsNew * self._effect_multiplier - self._costsNew
        stat_base = wtp * self._effectsBase * self._effect_multiplier - self._costsBase

        # to get CI for stat_new - stat_base
        diff_stat = Stat.DifferenceStatIndp(self.name, stat_new, stat_base)
        return diff_stat.get_t_CI(alpha)

    def get_PI(self, wtp, alpha=0.05):
        # NMB observations of two alternatives
        stat_new = wtp * self._effectsNew * self._effect_multiplier - self._costsNew
        stat_base = wtp * self._effectsBase * self._effect_multiplier - self._costsBase

        # to get PI for stat_new - stat_base
        diff_stat = Stat.DifferenceStatIndp(self.name, stat_new, stat_base)
        return diff_stat.get_PI(alpha)


from enum import Enum
import warnings
import math as math
import numpy as np
import SimPy.StatisticalClasses as Stat
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from SimPy import FigureSupport as Fig
from SimPy import FormatFunctions as FormatFunc
from SimPy import RandomVariantGenerators as RVG


NUM_WTPS_FOR_NMB_CURVES = 100       # number of wtp values to use to make net-monetary benefit curves


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


def get_an_interval(data, interval_type, alpha=0.05):
    """
    :param data: (list or numpy.array) data
    :param interval_type: (string) 'c' for t-based confidence interval,
                                   'cb' for bootstrap confidence interval, and
                                   'p' for percentile interval
    :param alpha: significance level
    :return: a list [L, U]
    """

    assert type(data) is list or type(data) is np.ndarray, \
        "Data should be a list or a np.array but {} provided.".format(type(data))

    sum_stat = Stat.SummaryStat('', data)
    return sum_stat.get_interval(interval_type=interval_type, alpha=alpha)


class HealthMeasure(Enum):
    UTILITY = 0         # as in QALYS
    DISUTILITY = 1      # as in DALY


class Strategy:
    def __init__(self, name, cost_obs, effect_obs, color=None):
        """
        :param name: name of the strategy
        :param cost_obs: list or numpy.array of cost observations
        :param effect_obs: list or numpy.array of effect observations
        :param color: (string) color code
                (https://www.webucator.com/blog/2015/03/python-color-constants-module/)
        """

        assert type(cost_obs) is list or type(cost_obs) is np.ndarray, \
            "cost_obs should be list or np.array."
        assert type(effect_obs) is list or type(effect_obs) is np.ndarray, \
            "effect_obs should be list or np.array."
        assert color is None or type(color) is str, "color argument should be a string."

        self.name = name
        if type(cost_obs) is list:
            self.costObs = np.array(cost_obs)
        else:
            self.costObs = cost_obs
        if type(effect_obs) is list:
            self.effectObs = np.array(effect_obs)
        else:
            self.effectObs = effect_obs

        self.aveCost = np.average(self.costObs)
        self.aveEffect = np.average(self.effectObs)
        self.ifDominated = False
        self.color = color

    def get_cost_interval(self, interval_type, alpha):
        """
        :param interval_type: (string) 'c' for t-based confidence interval,
                                       'cb' for bootstrap confidence interval, and
                                       'p' for percentile interval
        :param alpha: significance level
        :return: list [L, U] for a confidence or prediction interval of cost observations
        """
        return get_an_interval(self.costObs, interval_type, alpha)

    def get_effect_interval(self, interval_type, alpha):
        """
        :param interval_type: (string) 'c' for t-based confidence interval,
                               'cb' for bootstrap confidence interval, and
                               'p' for percentile interval
        :param alpha: significance level
        :return: list [L, U] for a confidence or prediction interval of effect observations
        """
        return get_an_interval(self.effectObs, interval_type, alpha)

    def get_cost_err_interval(self, interval_type, alpha):
        """
        :param interval_type: (string) 'c' for t-based confidence interval,
                                       'cb' for bootstrap confidence interval, and
                                       'p' for percentile interval
        :param alpha: significance level
        :return: list [err_l, err_u] for the lower and upper error length
                of confidence or prediction intervals of cost observations.
                NOTE: [err_l, err_u] = [mean - L, mean + U], where [L, U] is the confidence or prediction interval

        """
        interval = self.get_cost_interval(interval_type, alpha)
        return [self.aveCost - interval[0], interval[1] - self.aveCost]

    def get_effect_err_interval(self, interval_type, alpha):
        """
        :param interval_type: (string) 'c' for t-based confidence interval,
                                       'cb' for bootstrap confidence interval, and
                                       'p' for percentile interval
        :param alpha: significance level
        :return: list [err_l, err_u] for the lower and upper error length
                of confidence or prediction intervals of effect observations.
                NOTE: [err_l, err_u] = [mean - L, mean + U], where [L, U] is the confidence or prediction interval

        """
        interval = self.get_effect_interval(interval_type, alpha)
        return [self.aveEffect - interval[0], interval[1] - self.aveEffect]


class _EconEval:
    """ master class for cost-effective analysis (CEA) and cost-benefit analysis (CBA) """

    def __init__(self, strategies, if_paired, health_measure=HealthMeasure.UTILITY):
        """
        :param strategies: the list of strategies (assumes that the first strategy represents the "base" strategy)
        :param if_paired: set to true to indicate that the strategies are paired
        :param health_measure: set to HealthMeasure.UTILITY if higher "effect" implies better health
        (e.g. when QALY is used) and set to HealthMeasure.DISUTILITY if higher "effect" implies worse health
        (e.g. when DALYS is used)
        """
        self._n = len(strategies)  # number of strategies
        self._ifPaired = if_paired
        self._healthMeasure = health_measure
        self._utility_or_disutility = 1 if health_measure == HealthMeasure.UTILITY else -1

        self._strategies = strategies  # list of strategies
        self._strategiesOnFrontier = []  # list of strategies on the frontier
        self._strategiesNotOnFrontier = []  # list of strategies not on the frontier
        self._shiftedStrategies = [] # list of shifted strategies
        self._shiftedStrategiesOnFrontier = []  # list of shifted strategies on the frontier
        self._shiftedStrategiesNotOnFrontier = []  # list of shifted strategies not on the frontier
        self._ifFrontierIsCalculated = False # CE frontier is not calculated yet

        # shift the strategies
        self.__find_shifted_strategies()

    def __find_shifted_strategies(self):
        """ find shifted strategies.
        In calculating the change in effect, it accounts for whether QALY or DALY is used.
        """

        # shift all strategies such that the base strategy (first in the list) lies on the origin
        # if observations are paired across strategies
        if self._ifPaired:
            for i in range(self._n):
                shifted_strategy = Strategy(
                    name=self._strategies[i].name,
                    cost_obs=self._strategies[i].costObs - self._strategies[0].costObs,
                    effect_obs=(self._strategies[i].effectObs - self._strategies[0].effectObs) * self._utility_or_disutility,
                    color=self._strategies[i].color
                )
                self._shiftedStrategies.append(shifted_strategy)

        else:  # if not paired
            base_ave_cost = self._strategies[0].aveCost  # average cost of the base strategy
            base_ave_effect = self._strategies[0].aveEffect  # average effect of the base strategy
            for i in range(self._n):
                shifted_strategy = Strategy(
                    name=self._strategies[i].name,
                    cost_obs=self._strategies[i].costObs - base_ave_cost,
                    effect_obs=(self._strategies[i].effectObs - base_ave_effect) * self._utility_or_disutility,
                    color=self._strategies[i].color
                )
                self._shiftedStrategies.append(shifted_strategy)

    def get_shifted_strategies(self):
        """
        :return: the list of strategies after being shifted so that the first strategy lies on the origin of
            the cost-effectiveness plane
        """
        if len(self._shiftedStrategies) == 0:
            warnings.warn('The list of shifted strategies are empty.')

        return self._shiftedStrategies


class CEA(_EconEval):
    """ class for conducting cost-effectiveness analysis """

    def __init__(self, strategies, if_paired, health_measure=HealthMeasure.UTILITY):
        """
        :param strategies: list of strategies (assumes that the first strategy represents the "base" strategy)
        :param if_paired: set to true to indicate that the strategies are paired
        :param health_measure: set to HealthMeasure.UTILITY if higher "effect" implies better health
        (e.g. when QALY is used) and set to HealthMeasure.DISUTILITY if higher "effect" implies worse health
        (e.g. when DALYS is used)
        """
        _EconEval.__init__(self, strategies, if_paired, health_measure)

        self._dfStrategies = None  # data frame to store the CE table
        self._dfShiftedStrategies = None  # data frame to build CE figure

    def get_strategies_on_frontier(self):
        """ :return list of strategies on the frontier"""

        if not self._ifFrontierIsCalculated:
            self._find_frontier()

        return self._strategiesOnFrontier

    def get_strategies_not_on_frontier(self):
        """ :return list of strategies that are not on the frontier """

        if not self._ifFrontierIsCalculated:
            self._find_frontier()

        return self._strategiesNotOnFrontier

    def get_shifted_strategies_on_frontier(self):
        """ :return list of shifted strategies on the frontier"""

        if not self._ifFrontierIsCalculated:
            self._find_frontier()

        return self._shiftedStrategiesOnFrontier

    def get_shifted_strategies_not_on_frontier(self):
        """ :return list of shifted strategies not on the frontier"""

        if not self._ifFrontierIsCalculated:
            self._find_frontier()

        return self._shiftedStrategiesNotOnFrontier

    def _find_frontier(self):
        """ find the cost-effectiveness frontier """

        # populate the data frame for CE table
        self._dfStrategies = pd.DataFrame(
            index=range(self._n),
            columns=['Name', 'E[Cost]', 'E[Effect]', 'Dominated'])
        for j in range(self._n):
            self._dfStrategies.loc[j, 'Name'] = self._strategies[j].name
            self._dfStrategies.loc[j, 'E[Cost]'] = self._strategies[j].aveCost
            self._dfStrategies.loc[j, 'E[Effect]'] = self._strategies[j].aveEffect
            self._dfStrategies.loc[j, 'Dominated'] = self._strategies[j].ifDominated
            self._dfStrategies.loc[j, 'Color'] = "k"  # not Dominated black, Dominated blue

        # create a data frame for shifted strategies
        self._dfShiftedStrategies = pd.DataFrame(
            index=range(self._n),
            columns=['Name', 'E[Cost]', 'E[Effect]', 'Dominated', 'Color'])
        for j in range(self._n):
            self._dfShiftedStrategies.loc[j, 'Name'] = self._shiftedStrategies[j].name
            self._dfShiftedStrategies.loc[j, 'E[Cost]'] = self._shiftedStrategies[j].aveCost
            self._dfShiftedStrategies.loc[j, 'E[Effect]'] = self._shiftedStrategies[j].aveEffect
            self._dfShiftedStrategies.loc[j, 'Dominated'] = self._shiftedStrategies[j].ifDominated
            self._dfShiftedStrategies.loc[j, 'Color'] = "k"  # not Dominated black, Dominated blue

        # sort shifted strategies by cost in an ascending order
        df_shifted_sorted = self._dfShiftedStrategies.sort_values('E[Cost]')

        # apply criteria 1 (strict dominance)
        for i in range(self._n):
            # strategies with higher cost and lower Effect are dominated
            df_shifted_sorted.loc[
                (df_shifted_sorted['E[Cost]'] > df_shifted_sorted['E[Cost]'][i]) &
                (df_shifted_sorted['E[Effect]'] <= df_shifted_sorted['E[Effect]'][i]),
                'Dominated'] = True
        # change the color of dominated strategies to blue
        df_shifted_sorted.loc[df_shifted_sorted['Dominated'] == True, 'Color'] = 'blue'

        # apply criteria 2 (weak dominance)
        # select all non-dominated strategies
        df2 = df_shifted_sorted.loc[df_shifted_sorted['Dominated']==False]
        n2 = len(df2['E[Cost]'])

        for i in range(0, n2): # can't decide for first and last point
            for j in range(i+1, n2):
                # cost and effect of strategy i
                effect_i = df2['E[Effect]'].iloc[i]
                cost_i = df2['E[Cost]'].iloc[i]
                # cost and effect of strategy j
                effect_j = df2['E[Effect]'].iloc[j]
                cost_j = df2['E[Cost]'].iloc[j]
                # vector connecting strategy i to j
                v_i_to_j = np.array([effect_j-effect_i, cost_j-cost_i])

                # if the effect of no strategy is between the effects of strategies i and j
                if not ((df2['E[Effect]'] > effect_i) & (df2['E[Effect]'] < effect_j)).any():
                    continue    # to the next j
                else:
                    # get all the strategies with effect between strategies i and j
                    inner_points = df2.loc[(df2['E[Effect]'] > effect_i) & (df2['E[Effect]'] < effect_j)]
                    # difference in effect of inner points and strategy i
                    v2_x = inner_points['E[Effect]'] - effect_i
                    # difference in cost of inner points and strategy i
                    v2_y = inner_points['E[Cost]']-cost_i

                    # cross products of vector i to j and the vectors i to all inner points
                    cross_product = v_i_to_j[0] * np.array(v2_y) - v_i_to_j[1] * np.array(v2_x)

                    # if cross_product > 0 the point is above the line
                    # (because the point are sorted vertically)
                    # ref: How to tell whether a point is to the right or left side of a line
                    # https://stackoverflow.com/questions/1560492
                    dominated_index = inner_points[cross_product > 0].index
                    df_shifted_sorted.loc[list(dominated_index), 'Dominated'] = True
                    df_shifted_sorted.loc[list(dominated_index), 'Color'] = 'blue'

        # update the not shifted strategies
        for i in range(self._n):
            for j in range(self._n):
                if self._dfStrategies['Name'].iloc[i] == df_shifted_sorted['Name'].iloc[j]:
                    self._dfStrategies['Dominated'].iloc[i] = df_shifted_sorted['Dominated'].iloc[j]

        # sort the not shifted strategies
        self._dfStrategies = self._dfStrategies.sort_values('E[Cost]')

        # create list of strategies on frontier
        on_frontier_index = df_shifted_sorted[df_shifted_sorted['Dominated'] == False].index
        for i in on_frontier_index:
            self._strategies[i].ifDominated = False
            self._shiftedStrategies[i].ifDominated = False
            self._strategiesOnFrontier.append(self._strategies[i])
            self._shiftedStrategiesOnFrontier.append(self._shiftedStrategies[i])

        # create list of strategies not on frontier
        not_on_frontier_index = df_shifted_sorted[df_shifted_sorted['Dominated'] == True].index
        for j in not_on_frontier_index:
            self._strategies[j].ifDominated = True
            self._shiftedStrategies[j].ifDominated = True
            self._strategiesNotOnFrontier.append(self._strategies[j])
            self._shiftedStrategiesNotOnFrontier.append(self._shiftedStrategies[j])

        # update the data frame of the shifted strategies
        self._dfShiftedStrategies = df_shifted_sorted

        # frontier is calculated
        self._ifFrontierIsCalculated = True

    def show_CE_plane(self, title, x_label, y_label,
                      show_names=False, show_clouds=False, transparency=0.4,
                      show_legend=False, figure_size=6, x_range=None, y_range=None):
        """
        :param title: title of the figure
        :param x_label: (string) x-axis label
        :param y_label: (string) y-axis label
        :param show_names: logical, show strategy names
        :param show_clouds: logical, show true sample observation of strategies
        :param transparency: transparency of clouds (0.0 transparent through 1.0 opaque)
        :param show_legend: shows the legend of strategies, would only be used when show_clouds is true
        :param figure_size: int, specify the figure size
        :param x_range: list, range of x axis
        :param y_range: list, range of y axis
        """

        # find the frontier if not calculated already
        if not self._ifFrontierIsCalculated:
            self._find_frontier()

        # plots
        # operate on local variable data rather than self attribute
        df_shifted_strategies = self._dfShiftedStrategies
        # find the dominated strategies
        df_shifted_strategies["Dominated"] = self._dfStrategies["Dominated"]

        # re-sorted according to Effect to draw line
        frontier_plot = df_shifted_strategies.loc[df_shifted_strategies["Dominated"] == False]\
            .sort_values('E[Effect]')

        # draw the frontier
        plt.figure(figsize=(figure_size, figure_size))
        plt.plot(frontier_plot['E[Effect]'], frontier_plot['E[Cost]'], c='k', alpha=0.6, linewidth=2,
                 label="Frontier")
        plt.axhline(y=0, c='k', linewidth=0.5)
        plt.axvline(x=0, c='k', linewidth=0.5)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # show observation clouds for strategies
        if show_clouds:

            rainbow_colors = cm.rainbow(np.linspace(0, 1, self._n))
            colors = []
            i = 0
            for s in self._shiftedStrategies:
                if s.color:
                    colors.append(s.color)
                else:
                    colors.append(rainbow_colors[i])
                i += 1

            for strategy_i, color in zip(self._shiftedStrategies, colors):
                x_values = strategy_i.effectObs
                y_values = strategy_i.costObs
                # plot clouds
                plt.scatter(x_values, y_values, c=[color], alpha=transparency, s=25, label=strategy_i.name)
            if show_legend:
                handles, labels = plt.gca().get_legend_handles_labels()
                order = np.append(range(len(handles))[1:], 0)
                plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
                # to customize legend: loc='lower right', numpoints=1, ncol=3, fontsize=8)
            plt.scatter(df_shifted_strategies['E[Effect]'],
                        df_shifted_strategies['E[Cost]'],
                        marker='x', c='k', s=50, linewidths=2)

        else:
            plt.scatter(df_shifted_strategies['E[Effect]'],
                        df_shifted_strategies['E[Cost]'],
                        c=list(df_shifted_strategies['Color']),
                        s=50)

        if not (x_range is None):
            plt.xlim(x_range)

        if not (y_range is None):
            plt.ylim(y_range)

        vals_y, labs_y = plt.yticks()
        vals_x, labs_x = plt.xticks()

        # get ranges of x, y axis
        Lx = np.ptp(vals_x)
        Ly = np.ptp(vals_y)

        # if range Lx or Ly <= 10, use the default format for ticks
        # else, format them with 0 decimal
        if Ly > 10:
            plt.yticks(vals_y, ['{:,.{prec}f}'.format(x, prec=0) for x in vals_y])
        if Lx > 10:
            plt.xticks(vals_x, ['{:,.{prec}f}'.format(x, prec=0) for x in vals_x])

        # show names of strategies
        if show_names:
            if not show_clouds:
                for label, x, y, if_dominated in zip(
                        df_shifted_strategies['Name'],
                        df_shifted_strategies['E[Effect]'],
                        df_shifted_strategies['E[Cost]'],
                        df_shifted_strategies['Dominated']):
                    if if_dominated:
                        dx = - 0.05 * Lx
                        dy = + 0.03 * Ly
                    else:
                        dx = + 0.05 * Lx
                        dy = - 0.03 * Ly
                    plt.annotate(
                        label, xy=(x, y), xycoords='data', xytext=(x+dx, y+dy),
                        textcoords='data', weight='bold', ha='center')

            elif show_clouds:
                for label, x, y in zip(
                        df_shifted_strategies['Name'],
                        df_shifted_strategies['E[Effect]'],
                        df_shifted_strategies['E[Cost]']):
                    plt.annotate(
                        label, ha='center',
                        xy=(x, y), xycoords='data', xytext=(x - 0.05 * Lx, y + 0.04 * Ly), textcoords='data',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3', shrinkA=0, shrinkB=2),
                        weight='bold', bbox=dict(pad=0, facecolor="none", edgecolor="none"))

        # show the figure
        Fig.output_figure(plt, output_type='show', filename=title)

    def build_CE_table(self,
                       interval_type='n',
                       alpha=0.05,
                       cost_digits=0, effect_digits=2, icer_digits=1,
                       cost_multiplier=1, effect_multiplier=1,
                       file_name='CETable'):
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
        :return: output csv file called in local environment
        """

        # find the frontier if not calculated already
        if not self._ifFrontierIsCalculated:
            self._find_frontier()

        # add CEA columns to the strategies data frame
        self._dfStrategies['E[dCost]'] = "-"
        self._dfStrategies['E[dEffect]'] = "-"
        self._dfStrategies['ICER'] = "Dominated"

        # get strategies on the frontier sorted by cost
        frontier_strategies = self._dfStrategies.loc[self._dfStrategies["Dominated"] == False].sort_values('E[Cost]')
        n_frontier_strategies = frontier_strategies.shape[0]    # number of strategies

        list_incr_costs = []      # list of incremental costs
        list_incr_effects = []    # list of incremental effects
        list_ICERs = []           # list of ICER estimates

        # decide about the ICER of the first strategy on the frontier
        if frontier_strategies["E[Cost]"].iloc[0] < 0: ### *** this condition is not correct ****
            self._dfStrategies.loc[frontier_strategies.index[0], 'ICER'] = 'Cost-Saving'
        else:
            self._dfStrategies.loc[frontier_strategies.index[0], 'ICER'] = '-'

        # calculate incremental costs, incremental effects and ICER
        if n_frontier_strategies > 1:
            for i in range(1, n_frontier_strategies):

                # incremental cost
                incremental_cost = frontier_strategies["E[Cost]"].iloc[i]-frontier_strategies["E[Cost]"].iloc[i-1]
                list_incr_costs = np.append(list_incr_costs, incremental_cost)

                # incremental effect
                incremental_effect = self._utility_or_disutility\
                           *(frontier_strategies["E[Effect]"].iloc[i]-frontier_strategies["E[Effect]"].iloc[i-1])
                list_incr_effects = np.append(list_incr_effects, incremental_effect)

                # ICER
                try:
                    list_ICERs = np.append(list_ICERs, incremental_cost/incremental_effect)
                except ValueError:
                    warnings.warn('Incremental effect is 0 for strategy ' + frontier_strategies['Name'].iloc[i])
                    list_ICERs = np.append(list_ICERs, math.nan)

        # format estimates in the cost-effectiveness table
        self._format_nums_in_dfstrategies(frontier_strategies=frontier_strategies,
                                          list_incr_costs=list_incr_costs,
                                          list_incr_effects=list_incr_effects,
                                          list_ICERs=list_ICERs,
                                          cost_digits=cost_digits,
                                          effect_digits=effect_digits,
                                          icer_digits=icer_digits)

        # create a data frame for intervals
        df_intervals = None
        if interval_type in ['c', 'p']:
            df_intervals = pd.DataFrame(
                index=self._dfStrategies.index,
                columns=['Name', 'Cost', 'Effect', 'Dominated'])
            # initialize incremental cost and health and ICER with -
            df_intervals['dCost'] = '-'
            df_intervals['dEffect'] = '-'
            df_intervals['ICER'] = '-'

        # decide about what interval to return and create table out_intervals
        if interval_type == 'p':
            # calculate prediction intervals for cost and effect
            for i in self._dfStrategies.index:
                # populated name, dominated, cost and effect PI columns
                df_intervals.loc[i, 'Name'] = self._strategies[i].name
                df_intervals.loc[i, 'Dominated'] = self._dfStrategies.loc[i, 'Dominated']
                # prediction interval of cost
                df_intervals.loc[i, 'Cost'] = Stat.SummaryStat("", self._strategies[i].costObs).get_PI(alpha)
                # prediction interval of effect
                df_intervals.loc[i, 'Effect'] = Stat.SummaryStat("", self._strategies[i].effectObs).get_PI(alpha)

            # calculate prediction intervals for incremental cost and effect and ICER
            if self._ifPaired:
                for i in range(1, n_frontier_strategies):
                    # calculate the prediction interval of incremental cost
                    df_intervals.loc[frontier_strategies.index[i], 'dCost'] \
                        = Stat.DifferenceStatPaired("",
                                                    self._strategies[frontier_strategies.index[i]].costObs,
                                                    self._strategies[frontier_strategies.index[i-1]].costObs
                                                    ).get_PI(alpha)

                    # calculate the prediction interval of incremental effect
                    df_intervals.loc[frontier_strategies.index[i], 'dEffect'] \
                        = Stat.DifferenceStatPaired("",
                                                    self._strategies[frontier_strategies.index[i]].effectObs * self._utility_or_disutility,
                                                    self._strategies[frontier_strategies.index[i-1]].effectObs * self._utility_or_disutility
                                                    ).get_PI(alpha)

                    # calculate the prediction interval of ICER
                    PI_PairedICER = ICER_paired(
                        "",
                        self._strategies[frontier_strategies.index[i]].costObs,
                        self._strategies[frontier_strategies.index[i]].effectObs*self._utility_or_disutility,
                        self._strategies[frontier_strategies.index[i-1]].costObs,
                        self._strategies[frontier_strategies.index[i-1]].effectObs*self._utility_or_disutility)
                    # add the interval to the data frame
                    df_intervals.loc[frontier_strategies.index[i], 'ICER'] = \
                        PI_PairedICER.get_PI(alpha)

            else: # if not paired
                for i in range(1, n_frontier_strategies):
                    # calculate the prediction interval of incremental cost
                    df_intervals.loc[frontier_strategies.index[i], 'dCost'] \
                        = Stat.DifferenceStatIndp("",
                                                  self._strategies[frontier_strategies.index[i]].costObs,
                                                  self._strategies[frontier_strategies.index[i - 1]].costObs
                                                  ).get_PI(alpha)

                    # calculate the prediction interval of incremental effect
                    df_intervals.loc[frontier_strategies.index[i], 'dEffect'] \
                        = Stat.DifferenceStatIndp("",
                                                  self._strategies[frontier_strategies.index[i]].effectObs,
                                                  self._strategies[frontier_strategies.index[i - 1]].effectObs
                                                  ).get_PI(alpha)

                    # calculate the prediction interval of ICER
                    PI_indpICER = ICER_indp(
                        "",
                        self._strategies[frontier_strategies.index[i]].costObs,
                        self._strategies[frontier_strategies.index[i]].effectObs*self._utility_or_disutility,
                        self._strategies[frontier_strategies.index[i - 1]].costObs,
                        self._strategies[frontier_strategies.index[i - 1]].effectObs*self._utility_or_disutility)
                    # add the interval to the data frame
                    df_intervals.loc[frontier_strategies.index[i], 'ICER'] = PI_indpICER.get_PI(alpha)

        elif interval_type == 'c':
            # calculate confidence intervals for cost and effect
            for i in self._dfStrategies.index:
                # populated name, dominated, cost and effect CI columns
                df_intervals.loc[i, 'Name'] = self._strategies[i].name
                df_intervals.loc[i, 'Dominated'] = self._dfStrategies.loc[i, 'Dominated']
                # confidence interval of cost
                df_intervals.loc[i, 'Cost'] = Stat.SummaryStat("", self._strategies[i].costObs).get_t_CI(alpha)
                # confidence interval of effect
                df_intervals.loc[i, 'Effect'] = Stat.SummaryStat("", self._strategies[i].effectObs).get_t_CI(alpha)

            # calculate confidence intervals for incremental cost and effect and ICER
            if self._ifPaired:
                for i in range(1, n_frontier_strategies):

                    name = self._strategies[frontier_strategies.index[i]].name \
                           + " to " \
                           + self._strategies[frontier_strategies.index[i-1]].name

                    # calculate the confidence interval of incremental cost
                    df_intervals.loc[frontier_strategies.index[i], 'dCost'] \
                        = Stat.DifferenceStatPaired(name,
                                                    self._strategies[frontier_strategies.index[i]].costObs,
                                                    self._strategies[frontier_strategies.index[i-1]].costObs
                                                    ).get_t_CI(alpha)

                    # calculate the confidence interval of incremental effect
                    df_intervals.loc[frontier_strategies.index[i], 'dEffect'] \
                        = Stat.DifferenceStatPaired(name,
                                                    self._strategies[frontier_strategies.index[i]].effectObs * self._utility_or_disutility,
                                                    self._strategies[frontier_strategies.index[i-1]].effectObs * self._utility_or_disutility
                                                    ).get_t_CI(alpha)

                    # calculate the confidence interval of incremental ICER
                    CI_PairedICER = ICER_paired(
                        name,
                        self._strategies[frontier_strategies.index[i]].costObs,
                        self._strategies[frontier_strategies.index[i]].effectObs*self._utility_or_disutility,
                        self._strategies[frontier_strategies.index[i-1]].costObs,
                        self._strategies[frontier_strategies.index[i-1]].effectObs*self._utility_or_disutility)

                    # add the interval to the data frame
                    df_intervals.loc[frontier_strategies.index[i], 'ICER'] = CI_PairedICER.get_CI(alpha, 1000)

            else:  # if not paired
                for i in range(1, n_frontier_strategies):
                    # calculate the confidence interval of incremental cost
                    df_intervals.loc[frontier_strategies.index[i], 'dCost'] \
                        = Stat.DifferenceStatIndp("",
                                                  self._strategies[frontier_strategies.index[i]].costObs,
                                                  self._strategies[frontier_strategies.index[i - 1]].costObs
                                                  ).get_t_CI(alpha)

                    # calculate the confidence interval of incremental effect
                    df_intervals.loc[frontier_strategies.index[i], 'dEffect'] \
                        = Stat.DifferenceStatIndp("",
                                                  self._strategies[frontier_strategies.index[i]].effectObs * self._utility_or_disutility,
                                                  self._strategies[frontier_strategies.index[i - 1]].effectObs * self._utility_or_disutility
                                                  ).get_t_CI(alpha)

                    # calculate the confidence interval of incremental ICER
                    CI_indpICER = ICER_indp(
                        "",
                        self._strategies[frontier_strategies.index[i]].costObs,
                        self._strategies[frontier_strategies.index[i]].effectObs*self._utility_or_disutility,
                        self._strategies[frontier_strategies.index[i - 1]].costObs,
                        self._strategies[frontier_strategies.index[i - 1]].effectObs*self._utility_or_disutility)

                    # add the interval to the data frame
                    df_intervals.loc[frontier_strategies.index[i], 'ICER'] = \
                        CI_indpICER.get_CI(alpha, 1000)

        # create a cost-effective table to print
        ce_table = self._create_ce_table(
            frontier_strategies=frontier_strategies, df_intervals=df_intervals,
            cost_multiplier=cost_multiplier, effect_multiplier=effect_multiplier,
            cost_digits=cost_digits, effect_digits=effect_digits, icer_digits=icer_digits)

        # define column order and write csv
        ce_table[['Name', 'E[Cost]', 'E[Effect]', 'E[dCost]', 'E[dEffect]', 'ICER']].to_csv(
            file_name+'.csv', encoding='utf-8', index=False)

    def _format_nums_in_dfstrategies(self,
                                     frontier_strategies,
                                     list_incr_costs,
                                     list_incr_effects,
                                     list_ICERs,
                                     cost_digits,
                                     effect_digits,
                                     icer_digits):

        # format the estimates of incremental outcomes and ICER
        indices = frontier_strategies.index[1:]

        # format cost column
        if cost_digits == 0:
            self._dfStrategies['E[Cost]'] = self._dfStrategies['E[Cost]'].astype(int)
            self._dfStrategies.loc[indices, 'E[dCost]'] = list_incr_costs.astype(int)
        else:
            self._dfStrategies['E[Cost]'] = self._dfStrategies['E[Cost]'].astype(float).round(cost_digits)
            self._dfStrategies.loc[indices, 'E[dCost]'] = list_incr_costs.astype(float).round(cost_digits)

        # format effect column
        if effect_digits == 0:
            self._dfStrategies['E[Effect]'] = self._dfStrategies['E[Effect]'].astype(int)
            self._dfStrategies.loc[indices, 'E[dEffect]'] = list_incr_effects.astype(int)
        else:
            self._dfStrategies['E[Effect]'] = self._dfStrategies['E[Effect]'].astype(float).round(effect_digits)
            self._dfStrategies.loc[indices, 'E[dEffect]'] = list_incr_effects.astype(float).round(effect_digits)

        if icer_digits == 0:
            self._dfStrategies.loc[indices, 'ICER'] = list_ICERs.astype(int)
        else:
            self._dfStrategies.loc[indices, 'ICER'] = list_ICERs.astype(float).round(icer_digits)

    def _create_ce_table(self, frontier_strategies, df_intervals,
                         cost_multiplier, effect_multiplier,
                         cost_digits, effect_digits, icer_digits):

        if df_intervals is None:
            return self._dfStrategies

        # merge estimates and intervals together
        ce_table = pd.DataFrame(
            {'Name': self._dfStrategies['Name'],
             'E[Cost]': self._dfStrategies['E[Cost]'],
             'E[Effect]': self._dfStrategies['E[Effect]'],
             'E[dCost]': self._dfStrategies['E[dCost]'],
             'E[dEffect]': self._dfStrategies['E[dEffect]'],
             'ICER': self._dfStrategies['ICER']
             })

        # put estimates and intervals together
        for i in self._dfStrategies.index:
            ce_table.loc[i, 'E[Cost]'] = \
                FormatFunc.format_estimate_interval(
                    estimate=self._dfStrategies.loc[i, 'E[Cost]'] * cost_multiplier,
                    interval=[x * cost_multiplier for x in df_intervals.loc[i, 'Cost']],
                    deci=cost_digits, format=',')

            ce_table.loc[i, 'E[Effect]'] = \
                FormatFunc.format_estimate_interval(
                    estimate=self._dfStrategies.loc[i, 'E[Effect]'] * effect_multiplier,
                    interval=[x * effect_multiplier for x in df_intervals.loc[i, 'Effect']],
                    deci=effect_digits, format=',')

        # add the incremental and ICER estimates and intervals
        for i in range(1, frontier_strategies.shape[0]):
            ce_table.loc[frontier_strategies.index[i], 'E[dCost]'] = \
                FormatFunc.format_estimate_interval(
                    estimate=self._dfStrategies.loc[frontier_strategies.index[i], 'E[dCost]'] * cost_multiplier,
                    interval=[x * cost_multiplier for x in df_intervals.loc[frontier_strategies.index[i], 'dCost']],
                    deci=cost_digits, format=',')

            ce_table.loc[frontier_strategies.index[i], 'E[dEffect]'] = \
                FormatFunc.format_estimate_interval(
                    estimate=self._dfStrategies.loc[frontier_strategies.index[i], 'E[dEffect]'] * effect_multiplier,
                    interval=[x * effect_multiplier for x in
                              df_intervals.loc[frontier_strategies.index[i], 'dEffect']],
                    deci=effect_digits, format=',')

            ce_table.loc[frontier_strategies.index[i], 'ICER'] = \
                FormatFunc.format_estimate_interval(
                    estimate=self._dfStrategies.loc[frontier_strategies.index[i], 'ICER'],
                    interval=df_intervals.loc[frontier_strategies.index[i], 'ICER'],
                    deci=icer_digits, format=',')

        return ce_table

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
                all strategies
        """

        dictionary_results = {}

        for s in self._shiftedStrategies:

            interval = s.get_cost_interval(interval_type=interval_type, alpha=alpha)
            d_cost_text = FormatFunc.format_estimate_interval(
                estimate=s.aveCost * cost_multiplier,
                interval=[x*cost_multiplier for x in interval],
                deci=cost_digits,
                format=','
            )
            interval = s.get_effect_interval(interval_type=interval_type, alpha=alpha)
            d_effect_text = FormatFunc.format_estimate_interval(
                estimate=s.aveEffect*effect_multiplier,
                interval=[x*effect_multiplier for x in interval],
                deci=effect_digits,
                format=','
            )

            # calculate cost-effectiveness ratio
            if self._ifPaired:
                ratio_stat = Stat.RatioStatPaired(name='', x=s.costObs, y_ref=s.effectObs)
            else:
                ratio_stat = Stat.RatioStatIndp(name='', x=s.costObs, y_ref=s.effectObs)
            cer_text = FormatFunc.format_estimate_interval(
                estimate=ratio_stat.get_mean(),
                interval=ratio_stat.get_interval(interval_type=interval_type, alpha=alpha),
                deci=icer_digits,
                format=',')

            # add to the dictionary
            dictionary_results[s.name] = [d_cost_text, d_effect_text, cer_text]

        return dictionary_results


class NMBCurve:

    def __init__(self, label, color, wtps, ys, l_errs, u_errs):
        self.label = label
        self.color = color
        self.wtps = wtps
        self.ys = ys
        self.l_errs = l_errs
        self.u_errs = u_errs


class CBA(_EconEval):
    """ class for doing cost-benefit analysis """

    def __init__(self, strategies, if_paired, health_measure=HealthMeasure.UTILITY):
        """
        :param strategies: the list of strategies (assumes that the first strategy represents the "base" strategy)
        :param if_paired: indicate whether the strategies are paired
        :param health_measure: set to HealthMeasure.UTILITY if higher "effect" implies better health
        (e.g. when QALY is used) and set to HealthMeasure.DISUTILITY if higher "effect" implies worse health
        (e.g. when DALYS is used)
        """
        _EconEval.__init__(self, strategies, if_paired, health_measure)

        self.nmbCurves = []  # list of NMB curves

    def make_nmb_curves(self, min_wtp, max_wtp, interval_type='n'):
        """
        prepares the information needed to plot the incremental net-monetary benefit
        compared to the first strategy (base)
        :param min_wtp: minimum willingness-to-pay (or cost-effectiveness threshold) on the x-axis
        :param max_wtp: maximum willingness-to-pay (or cost-effectiveness threshold) on the x-axis
        :param interval_type: (string) 'n' for no interval,
                                       'c' for confidence interval,
                                       'p' for percentile interval):
        """

        self.nmbCurves = []  # empty the list of NMB curves

        # wtp values at which NMB should be evaluated
        x_values = np.linspace(min_wtp, max_wtp, num=NUM_WTPS_FOR_NMB_CURVES, endpoint=True)

        # decide about the color of each curve
        rainbow_colors = cm.rainbow(np.linspace(0, 1, self._n - 1))
        colors = []
        i = 0
        for s in self._strategies[1:]:
            if s.color:
                colors.append(s.color)
            else:
                colors.append(rainbow_colors[i])
            i += 1

        # create the NMB curves
        for strategy_i, color in zip(self._strategies[1:], colors):

            if self._ifPaired:
                # create a paired NMB object
                paired_nmb = NMB_paired(name=strategy_i.name,
                                        costs_new=strategy_i.costObs,
                                        effects_new=strategy_i.effectObs,
                                        costs_base=self._strategies[0].costObs,
                                        effects_base=self._strategies[0].effectObs,
                                        health_measure=self._healthMeasure)

                # get the NMB values for each wtp
                y_values, l_err, u_err = self.__get_ys_lerrs_uerrs(
                    nmb=paired_nmb, wtps=x_values, interval_type=interval_type
                )

            else:
                # create an indp NMB object
                ind_nmb = NMB_indp(name=strategy_i.name,
                                   costs_new=strategy_i.costObs,
                                   effects_new=strategy_i.effectObs,
                                   costs_base=self._strategies[0].costObs,
                                   effects_base=self._strategies[0].effectObs,
                                   health_measure=self._healthMeasure)

                # get the NMB values for each wtp
                y_values, l_err, u_err = self.__get_ys_lerrs_uerrs(
                    nmb=ind_nmb, wtps=x_values, interval_type=interval_type
                )

            # make a NMB curve
            self.nmbCurves.append(NMBCurve(label=strategy_i.name,
                                           color=color,
                                           wtps=x_values,
                                           ys=y_values,
                                           l_errs=l_err,
                                           u_errs=u_err)
                                  )

    def graph_incremental_NMBs(self, min_wtp, max_wtp,
                               title, x_label, y_label,
                               interval_type='n', transparency=0.4,
                               show_legend=False, figure_size=(6, 6)):
        """
        plots the incremental net-monetary benefit compared to the first strategy (base)
        :param min_wtp: minimum willingness-to-pay (or cost-effectiveness threshold) on the x-axis
        :param max_wtp: maximum willingness-to-pay (or cost-effectiveness threshold) on the x-axis
        :param title: title
        :param x_label: x-axis label
        :param y_label: y-axis label
        :param interval_type: (string) 'n' for no interval,
                                       'c' for confidence interval,
                                       'p' for percentile interval
        :param transparency: transparency of intervals (0.0 transparent through 1.0 opaque)
        :param show_legend: set true to show legend
        :param figure_size: (tuple) size of the figure (e.g. (2, 3)
        """

        # make the NMB curves
        self.make_nmb_curves(min_wtp=min_wtp,
                             max_wtp=max_wtp,
                             interval_type=interval_type)

        # initialize plot
        plt.figure(figsize=figure_size)

        for curve in self.nmbCurves:
            # plot line
            plt.plot(curve.wtps, curve.ys, c=curve.color, alpha=1, label=curve.label)
            # plot intervals
            plt.fill_between(curve.wtps, curve.ys - curve.l_errs, curve.ys + curve.u_errs,
                             color=curve.color, alpha=transparency)

        if show_legend:
            plt.legend()

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        vals_y, labs_y = plt.yticks()
        vals_x, labs_x = plt.xticks()
        plt.yticks(vals_y, ['{:,.{prec}f}'.format(x, prec=0) for x in vals_y])
        plt.xticks(vals_x, ['{:,.{prec}f}'.format(x, prec=0) for x in vals_x])

        d = (max_wtp - min_wtp) / NUM_WTPS_FOR_NMB_CURVES
        plt.xlim([min_wtp-d, max_wtp+d])

        plt.axhline(y=0, c='k', ls='--', linewidth=0.5)
        #plt.axvline(x=0, c='k', ls='--', linewidth=0.5)

        plt.show()

    def __get_ys_lerrs_uerrs(self, nmb, wtps, interval_type='c'):

        # get the NMB values for each wtp
        y_values = [nmb.get_NMB(x) for x in wtps]

        if interval_type == 'c':
            y_ci = [nmb.get_CI(x, alpha=0.05) for x in wtps]
        elif interval_type == 'p':
            y_ci = [nmb.get_PI(x, alpha=0.05) for x in wtps]
        else:
            raise ValueError('Invalid value for internal_type.')

        # reshape confidence interval to plot
        u_err = np.array([p[1] for p in y_ci]) - y_values
        l_err = y_values - np.array([p[0] for p in y_ci])

        return y_values, l_err, u_err


class _ComparativeEconMeasure:
    def __init__(self, name, costs_new, effects_new, costs_base, effects_base, health_measure=HealthMeasure.UTILITY):
        """
        :param costs_new: (list or numpy.array) cost data for the new strategy
        :param effects_new: (list or numpy.array) effect data for the new strategy
        :param costs_base: (list or numpy.array) cost data for the base line
        :param effects_base: (list or numpy.array) effect data for the base line
        :param health_measure: set to HealthMeasure.UTILITY if higher "effect" implies better health
        (e.g. when QALY is used) and set to HealthMeasure.DISUTILITY if higher "effect" implies worse health
        (e.g. when DALYS is used)
        """

        assert type(costs_new) is list or type(costs_new) is np.ndarray, \
            "cost_new should be list or np.array."
        assert type(effects_new) is list or type(effects_new) is np.ndarray, \
            "effect_new should be list or np.array."
        assert type(costs_base) is list or type(costs_base) is np.ndarray, \
            "cost_base should be list or np.array."
        assert type(effects_base) is list or type(effects_base) is np.ndarray, \
            "effect_base should be list or np.array."
        assert type(health_measure) is HealthMeasure, \
            "health_measure should be of type HealthMeasure."

        self.name = name
        self._costsNew = costs_new          # cost data for the new strategy
        self._effectsNew = effects_new      # effect data for the new strategy
        self._costsBase = costs_base        # cost data for teh base line
        self._effectsBase = effects_base    # effect data for the base line
        # if QALY or DALY is being used
        self._effect_multiplier = 1 if health_measure == HealthMeasure.UTILITY else -1

        # convert input data to numpy.array if needed
        if type(costs_new) == list:
            self._costsNew = np.array(costs_new)
        if type(effects_new) == list:
            self._effectsNew = np.array(effects_new)
        if type(costs_base) == list:
            self._costsBase = np.array(costs_base)
        if type(effects_base) == list:
            self._effectsBase = np.array(effects_base)

        # calculate the difference in average cost and effect
        self._delta_ave_cost = np.average(self._costsNew) - np.average(self._costsBase)
        # change in effect: DALY averted or QALY gained
        self._delta_ave_effect = (np.average(self._effectsNew) - np.average(self._effectsBase)) \
                                 * self._effect_multiplier


class _ICER(_ComparativeEconMeasure):
    def __init__(self, name, costs_new, effects_new, costs_base, effects_base, health_measure=HealthMeasure.UTILITY):
        """
        :param costs_new: (list or numpy.array) cost data for the new strategy
        :param effects_new: (list or numpy.array) effect data for the new strategy
        :param costs_base: (list or numpy.array) cost data for the base line
        :param effects_base: (list or numpy.array) effect data for the base line
        :param health_measure: set to HealthMeasure.UTILITY if higher "effect" implies better health
        (e.g. when QALY is used) and set to HealthMeasure.DISUTILITY if higher "effect" implies worse health
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
            self._ICER = self._delta_ave_cost/self._delta_ave_effect

    def get_ICER(self):
        """ return ICER """
        return self._ICER

    def get_CI(self, alpha, num_bootstrap_samples, rng=None):
        """
        :param alpha: significance level, a value from [0, 1]
        :param num_bootstrap_samples: number of bootstrap samples
        :param rng: random number generator
        :return: confidence interval in the format of list [l, u]
        """
        # abstract method to be overridden in derived classes to process an event
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")

    def get_PI(self, alpha):
        """
        :param alpha: significance level, a value from [0, 1]
        :return: percentile interval in the format of list [l, u]
        """
        # abstract method to be overridden in derived classes to process an event
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")


class ICER_paired(_ICER):

    def __init__(self, name, costs_new, effects_new, costs_base, effects_base, health_measure=HealthMeasure.UTILITY):
        """
        :param costs_new: (list or numpy.array) cost data for the new strategy
        :param effects_new: (list or numpy.array) health data for the new strategy
        :param costs_base: (list or numpy.array) cost data for the base line
        :param effects_base: (list or numpy.array) health data for the base line
        :param health_measure: set to HealthMeasure.UTILITY if higher "effect" implies better health
        (e.g. when QALY is used) and set to HealthMeasure.DISUTILITY if higher "effect" implies worse health
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
            warnings.warn('\nConfidence intervals for one of ICERs is not computable'
                          '\nbecause at least one of bootstrap incremental effect is negative.'
                          '\nIncreasing the number of cost and effect observations might resolve the issue.')

        # calculate ICERs
        if self._isDefined:
            self._icers = np.divide(self._deltaCosts, self._deltaEffects)

    def get_CI(self, alpha, num_bootstrap_samples, rng=None):
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

                icer_bootstrap_means[i] = ave_delta_cost/ave_delta_effect - self._ICER

        # return the bootstrap interval
        return self._ICER - np.percentile(icer_bootstrap_means, [100*(1-alpha/2.0), 100*alpha/2.0])

    def get_PI(self, alpha):
        """
        :param alpha: significance level, a value from [0, 1]
        :return: prediction interval in the format of list [l, u]
        """
        if not self._isDefined:
            return [math.nan, math.nan]

        return np.percentile(self._icers, [100 * alpha / 2.0, 100 * (1 - alpha / 2.0)])


class ICER_indp(_ICER):

    def __init__(self, name, costs_new, effects_new, costs_base, effects_base, health_measure=HealthMeasure.UTILITY):
        """
        :param costs_new: (list or numpy.array) cost data for the new strategy
        :param effects_new: (list or numpy.array) health data for the new strategy
        :param costs_base: (list or numpy.array) cost data for the base line
        :param effects_base: (list or numpy.array) health data for the base line
        :param health_measure: set to HealthMeasure.UTILITY if higher "effect" implies better health
        (e.g. when QALY is used) and set to HealthMeasure.DISUTILITY if higher "effect" implies worse health
        (e.g. when DALYS is used)
        """

        # all cost and effects should have the same length for each alternative
        if not (len(costs_new) == len(effects_new) and len(costs_base) == len(effects_base)):
            raise ValueError(
                'ICER assume the same number of observations for the cost and health outcome of each alternative.')

        # initialize the base class
        _ICER.__init__(self, name, costs_new, effects_new, costs_base, effects_base, health_measure)

    def get_CI(self, alpha, num_bootstrap_samples, rng=None):
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
                    (mean_costs_new - mean_costs_base)/(mean_effects_new - mean_effects_base) \
                    * self._effect_multiplier

        if self._isDefined:
            return np.percentile(icer_bootstrap_means, [100*alpha/2.0, 100*(1-alpha/2.0)])
        else:
            return [math.nan, math.nan]

    def get_PI(self, alpha, num_bootstrap_samples=0, rng=None):
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

        if min((effects_new - effects_base)*self._effect_multiplier) <= 0:
            self._isDefined = False
        else:
            sample_icers = np.divide(
                (costs_new - costs_base),
                (effects_new - effects_base)*self._effect_multiplier)

        if self._isDefined:
            return np.percentile(sample_icers, [100*alpha/2.0, 100*(1-alpha/2.0)])
        else:
            return [math.nan, math.nan]


class _NMB(_ComparativeEconMeasure):
    def __init__(self, name, costs_new, effects_new, costs_base, effects_base, health_measure=HealthMeasure.UTILITY):
        """
        :param costs_new: (list or numpy.array) cost data for the new strategy
        :param effects_new: (list or numpy.array) health data for the new strategy
        :param costs_base: (list or numpy.array) cost data for the base line
        :param effects_base: (list or numpy.array) health data for the base line
        :param health_measure: set to HealthMeasure.UTILITY if higher "effect" implies better health
        (e.g. when QALY is used) and set to HealthMeasure.DISUTILITY if higher "effect" implies worse health
        (e.g. when DALYS is used)
        """
        # initialize the base class
        _ComparativeEconMeasure.__init__(self, name, costs_new, effects_new, costs_base, effects_base, health_measure)

    def get_NMB(self, wtp):
        """
        :param wtp: willingness-to-pay ($ for QALY gained or $ for DALY averted)
        :returns: the net monetary benefit at the provided willingness-to-pay value
        """
        return wtp * self._delta_ave_effect - self._delta_ave_cost

    def get_CI(self, wtp, alpha):
        """
        :param wtp: willingness-to-pay value ($ for QALY gained or $ for DALY averted)
        :param alpha: significance level, a value from [0, 1]
        :return: confidence interval in the format of list [l, u]
        """
        # abstract method to be overridden in derived classes
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")

    def get_PI(self, wtp, alpha):
        """
        :param wtp: willingness-to-pay value ($ for QALY gained or $ for DALY averted)
        :param alpha: significance level, a value from [0, 1]
        :return: percentile interval in the format of list [l, u]
        """
        # abstract method to be overridden in derived classes
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")


class NMB_paired(_NMB):

    def __init__(self, name, costs_new, effects_new, costs_base, effects_base, health_measure=HealthMeasure.UTILITY):
        """
        :param costs_new: (list or numpy.array) cost data for the new strategy
        :param effects_new: (list or numpy.array) health data for the new strategy
        :param costs_base: (list or numpy.array) cost data for the base line
        :param effects_base: (list or numpy.array) health data for the base line
        :param health_measure: set to HealthMeasure.UTILITY if higher "effect" implies better health
        (e.g. when QALY is used) and set to HealthMeasure.DISUTILITY if higher "effect" implies worse health
        (e.g. when DALYS is used)
        """
        _NMB.__init__(self, name, costs_new, effects_new, costs_base, effects_base, health_measure)

        # incremental observations
        self._deltaCost = self._costsNew - self._costsBase
        self._deltaHealth = (self._effectsNew - self._effectsBase) * self._effect_multiplier

    def get_CI(self, wtp, alpha):

        # create a summary statistics
        stat = Stat.SummaryStat(self.name, wtp * self._deltaHealth - self._deltaCost)
        return stat.get_t_CI(alpha)

    def get_PI(self, wtp, alpha):

        # create a summary statistics
        stat = Stat.SummaryStat(self.name, wtp * self._deltaHealth - self._deltaCost)
        return stat.get_PI(alpha)


class NMB_indp(_NMB):

    def __init__(self, name, costs_new, effects_new, costs_base, effects_base, health_measure=HealthMeasure.UTILITY):
        """
        :param costs_new: (list or numpy.array) cost data for the new strategy
        :param effects_new: (list or numpy.array) health data for the new strategy
        :param costs_base: (list or numpy.array) cost data for the base line
        :param effects_base: (list or numpy.array) health data for the base line
        :param health_measure: set to HealthMeasure.UTILITY if higher "effect" implies better health
        (e.g. when QALY is used) and set to HealthMeasure.DISUTILITY if higher "effect" implies worse health
        (e.g. when DALYS is used)
        """
        _NMB.__init__(self, name, costs_new, effects_new, costs_base, effects_base, health_measure)

    def get_CI(self, wtp, alpha):

        # NMB observations of two alternatives
        stat_new = wtp * self._effectsNew * self._effect_multiplier - self._costsNew
        stat_base = wtp * self._effectsBase * self._effect_multiplier - self._costsBase

        # to get CI for stat_new - stat_base
        diff_stat = Stat.DifferenceStatIndp(self.name, stat_new, stat_base)
        return diff_stat.get_t_CI(alpha)

    def get_PI(self, wtp, alpha):

        # NMB observations of two alternatives
        stat_new = wtp * self._effectsNew * self._effect_multiplier - self._costsNew
        stat_base = wtp * self._effectsBase * self._effect_multiplier - self._costsBase

        # to get PI for stat_new - stat_base
        diff_stat = Stat.DifferenceStatIndp(self.name, stat_new, stat_base)
        return diff_stat.get_PI(alpha)

from enum import Enum
import numpy as np
import SimPy.StatisticalClasses as Stat
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from SimPy import FigureSupport as Fig
from SimPy import FormatFunctions as FormatFunc


def pv(payment, discount_rate, discount_period):
    """ calculates the present value of a payment
    :param payment: payment to calculate the present value for
    :param discount_rate: discount rate (per period)
    :param discount_period: number of periods to discount the payment
    :return: payment/(1+discount_rate)^discount_period    """
    return payment * pow(1 + discount_rate, -discount_period)


class Interval(Enum):
    NO_INTERVAL = 0
    CONFIDENCE = 1
    PREDICTION = 2


class HealthMeasure(Enum):
    UTILITY = 0
    DISUTILITY = 1


def get_an_interval(data, interval, alpha):
    sum_stat = Stat.SummaryStat('', data)
    if interval == Interval.CONFIDENCE:
        return sum_stat.get_t_CI(alpha)
    elif interval == Interval.PREDICTION:
        return sum_stat.get_PI(alpha)
    else:
        return None


class Strategy:
    def __init__(self, name, cost_obs, effect_obs):
        """
        :param name: name of the strategy
        :param cost_obs: list or numpy.array of cost observations
        :param effect_obs: list or numpy.array of effect observations
        """
        self.name = name
        if type(cost_obs) == list:
            self.costObs = np.array(cost_obs)
        else:
            self.costObs = cost_obs
        if type(effect_obs) == list:
            self.effectObs = np.array(effect_obs)
        else:
            self.effectObs = effect_obs
        self.aveCost = np.average(self.costObs)
        self.aveEffect = np.average(self.effectObs)
        self.ifDominated = False

    def get_cost_interval(self, interval, alpha):
        return get_an_interval(self.costObs, interval, alpha)

    def get_effect_interval(self, interval, alpha):
        return get_an_interval(self.effectObs, interval, alpha)


class _EconEval:
    """ master class for cost-effective analysis (CEA) and cost-benefit analysis (CBA) """

    def __init__(self, strategies, if_paired, health_measure=HealthMeasure.UTILITY):
        """
        :param strategies: the list of strategies (assumes that the first strategy represents the "base" strategy)
        :param if_paired: indicate whether the strategies are paired
        :param health_measure: set to HealthMeasure.UTILITY if higher "effect" implies better health
        (e.g. when QALY is used) and set to HealthMeasure.DISUTILITY if higher "effect" implies worse health
        (e.g. when DALYS is used)
        """
        self._n = len(strategies)  # number of strategies
        self._strategies = strategies  # list of strategies
        self._strategiesOnFrontier = []  # list of strategies on the frontier
        self._strategiesNotOnFrontier = []  # list of strategies not on the frontier
        self._shifted_strategiesOnFrontier = []  # list of shifted strategies on the frontier
        self._shifted_strategiesNotOnFrontier = []  # list of shifted strategies not on the frontier
        self._ifPaired = if_paired
        self._effect_multiplier = (-1, 1)[health_measure == HealthMeasure.UTILITY]

        # create a data frame for all strategies' expected outcomes
        self._dfStrategies = pd.DataFrame(
            index=range(self._n),
            columns=['Name', 'E[Cost]', 'E[Effect]', 'Dominated'])

        # populate the data frame
        for j in range(self._n):
            self._dfStrategies.loc[j, 'Name'] = strategies[j].name
            self._dfStrategies.loc[j, 'E[Cost]'] = strategies[j].aveCost
            self._dfStrategies.loc[j, 'E[Effect]'] = strategies[j].aveEffect
            self._dfStrategies.loc[j, 'Dominated'] = strategies[j].ifDominated
            self._dfStrategies.loc[j, 'Color'] = "k"  # not Dominated black, Dominated blue

        # now shift all strategies such that the base strategy (first in the list) lies on the origin
        # all the following data analysis are based on the shifted data
        shifted_strategies = []
        # if observations are paired across strategies
        if if_paired:
            for i in range(self._n):
                shifted_strategy = Strategy(strategies[i].name,
                                            strategies[i].costObs - strategies[0].costObs,
                                            (strategies[i].effectObs - strategies[0].effectObs)*self._effect_multiplier)

                shifted_strategies.append(shifted_strategy)

        else:  # if not paired
            e_cost = strategies[0].aveCost
            e_effect = strategies[0].aveEffect
            for i in range(self._n):
                shifted_strategy = Strategy(strategies[i].name,
                                            strategies[i].costObs - e_cost,
                                            (strategies[i].effectObs - e_effect)*self._effect_multiplier)
                shifted_strategies.append(shifted_strategy)
        self._shifted_strategies = shifted_strategies  # list of shifted strategies

        # create a data frame for all strategies' shifted expected outcomes
        self._dfStrategies_shifted = pd.DataFrame(
            index=range(self._n),
            columns=['Name', 'E[Cost]', 'E[Effect]', 'Dominated', 'Color'])

        # populate the data frame
        for j in range(self._n):
            self._dfStrategies_shifted.loc[j, 'Name'] = shifted_strategies[j].name
            self._dfStrategies_shifted.loc[j, 'E[Cost]'] = shifted_strategies[j].aveCost
            self._dfStrategies_shifted.loc[j, 'E[Effect]'] = shifted_strategies[j].aveEffect
            self._dfStrategies_shifted.loc[j, 'Dominated'] = shifted_strategies[j].ifDominated
            self._dfStrategies_shifted.loc[j, 'Color'] = "k"  # not Dominated black, Dominated blue

    def get_shifted_strategies(self):
        """
        :return: the list of strategies after being shifted so that the first strategy falls on the origin
        """
        return self._shifted_strategies


class CEA(_EconEval):
    """ class for doing cost-effectiveness analysis """

    def __init__(self, strategies, if_paired, if_find_frontier=True, health_measure=HealthMeasure.UTILITY):
        """
        :param strategies: the list of strategies (assumes that the first strategy represents the "base" strategy)
        :param if_paired: indicate whether the strategies are paired
        :param if_find_frontier: if the cost-effectiveness frontier should be calculated
        :param health_measure: set to HealthMeasure.UTILITY if higher "effect" implies better health
        (e.g. when QALY is used) and set to HealthMeasure.DISUTILITY if higher "effect" implies worse health
        (e.g. when DALYS is used)
        """
        _EconEval.__init__(self, strategies, if_paired, health_measure)

        # find the CE frontier
        if if_find_frontier:
            self.__find_frontier()

    def get_strategies_on_frontier(self):
        """ :return list of strategies on the frontier"""
        return self._strategiesOnFrontier

    def get_strategies_not_on_frontier(self):
        """ :return list of strategies that are not on the frontier """
        return self._strategiesNotOnFrontier

    def get_shifted_strategies_on_frontier(self):
        """ :return list of shifted strategies on the frontier"""
        return self._shifted_strategiesOnFrontier

    def get_shifted_strategies_not_on_frontier(self):
        """ :return list of shifted strategies not on the frontier"""
        return self._shifted_strategiesNotOnFrontier

    def __find_frontier(self):
        """ find the cost-effectiveness frontier """

        # sort shifted strategies by cost, ascending
        # operate on local variable data rather than self attribute
        df_shifted_sorted = self._dfStrategies_shifted.sort_values('E[Cost]')

        # apply criteria 1
        for i in range(self._n):
            # strategies with higher cost and lower Effect are dominated
            df_shifted_sorted.loc[
                (df_shifted_sorted['E[Cost]'] > df_shifted_sorted['E[Cost]'][i]) &
                (df_shifted_sorted['E[Effect]'] <= df_shifted_sorted['E[Effect]'][i]),
                'Dominated'] = True
        # change the color of dominated strategies to blue
        df_shifted_sorted.loc[df_shifted_sorted['Dominated'] == True, 'Color'] = 'blue'

        # apply criteria 2
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

        # update the unshifted strategies
        for i in range(self._n):
            for j in range(self._n):
                if self._dfStrategies['Name'].iloc[i] == df_shifted_sorted['Name'].iloc[j]:
                    self._dfStrategies['Dominated'].iloc[i] = df_shifted_sorted['Dominated'].iloc[j]

        # sort the unshifted strategies
        self._dfStrategies = self._dfStrategies.sort_values('E[Cost]')

        # create list of strategies on frontier
        on_frontier_index = df_shifted_sorted[df_shifted_sorted['Dominated'] == False].index
        for i in on_frontier_index:
            self._strategiesOnFrontier.append(self._strategies[i])
            self._shifted_strategiesOnFrontier.append(self._shifted_strategies[i])

        # create list of strategies not on frontier
        not_on_frontier_index = df_shifted_sorted[df_shifted_sorted['Dominated'] == True].index
        for j in not_on_frontier_index:
            self._strategiesNotOnFrontier.append(self._strategies[j])
            self._shifted_strategiesNotOnFrontier.append(self._shifted_strategies[j])

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
        # plots
        # operate on local variable data rather than self attribute
        df_shifted_strategies = self._dfStrategies_shifted
        # find the dominated strategies
        df_shifted_strategies["Dominated_result"] = self._dfStrategies["Dominated"]

        # re-sorted according to Effect to draw line
        frontier_plot = df_shifted_strategies.loc[df_shifted_strategies["Dominated_result"] == False]\
            .sort_values('E[Effect]')

        # draw the fontier
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
            for strategy_i, color in zip(self._shifted_strategies, cm.rainbow(np.linspace(0, 1, self._n))):
                x_values = strategy_i.effectObs
                y_values = strategy_i.costObs
                # plot clouds
                plt.scatter(x_values, y_values, c=color, alpha=transparency, s=25, label=strategy_i.name)
            if show_legend:
                handles, labels = plt.gca().get_legend_handles_labels()
                order = np.append(range(len(handles))[1:], 0)
                plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
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
                for label, x, y in zip(
                        df_shifted_strategies['Name'],
                        df_shifted_strategies['E[Effect]'],
                        df_shifted_strategies['E[Cost]']):
                    plt.annotate(
                        label, xy=(x, y), xycoords='data', xytext=(x - 0.05 * Lx, y + 0.03 * Ly),
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
        Fig.output_figure(plt, Fig.OutType.SHOW, title)

    def build_CE_table(self,
                       interval=Interval.NO_INTERVAL,
                       alpha=0.05,
                       cost_digits=0, effect_digits=2, icer_digits=1,
                       file_name='CETable'):
        """
        :param interval: type of interval to report for the cost, effect and ICER estimates,
                        can take values from
                        Interval.NO_INTERVAL, Interval.CONFIDENCE, Interval.PREDICTION
        :param alpha: significance level
        :param cost_digits: digits to round cost estimates to
        :param effect_digits: digits to round effect estimate to
        :param icer_digits: digits to round ICER estimates to
        :param file_name: address and file name where the CEA results should be saved to
        :return: output csv file called in local environment
        """

        # initialize the table
        #dfStrategies = self._dfStrategies
        self._dfStrategies['E[dCost]'] = "-"
        self._dfStrategies['E[dEffect]'] = "-"
        self._dfStrategies['ICER'] = "Dominated"

        # get strategies on the frontier
        frontier_strategies = self._dfStrategies.loc[self._dfStrategies["Dominated"] == False].sort_values('E[Cost]')
        # number of strategies on the frontier
        n_frontier_strategies = frontier_strategies.shape[0]

        incr_cost = []      # list of incremental costs
        incr_effect = []    # list of incremental effects
        ICER = []           # list of ICER estimates

        # calculate incremental costs, incremental effects and ICER
        if n_frontier_strategies > 1:
            for i in range(1, n_frontier_strategies):
                # incremental cost
                d_cost = frontier_strategies["E[Cost]"].iloc[i]-frontier_strategies["E[Cost]"].iloc[i-1]
                incr_cost = np.append(incr_cost, d_cost)
                # incremental effect
                d_effect = self._effect_multiplier\
                           *(frontier_strategies["E[Effect]"].iloc[i]-frontier_strategies["E[Effect]"].iloc[i-1])
                if d_effect == 0:
                    raise ValueError('invalid value of E[dEffect], the ratio is not computable')
                incr_effect = np.append(incr_effect, d_effect)
                # ICER
                ICER = np.append(ICER, d_cost/d_effect)

            # format the numbers
            ind_change = frontier_strategies.index[1:]
            self._dfStrategies.loc[ind_change, 'E[dCost]'] = incr_cost.astype(float).round(cost_digits)
            self._dfStrategies.loc[ind_change, 'E[dEffect]'] = incr_effect.astype(float).round(effect_digits)
            self._dfStrategies.loc[ind_change, 'ICER'] = ICER.astype(float).round(icer_digits)

        # put - for the ICER of the first strategy on the frontier
        self._dfStrategies.loc[frontier_strategies.index[0], 'ICER'] = '-'

        # format cost column
        if cost_digits == 0:
            output_cost = self._dfStrategies['E[Cost]'].astype(int)
        else:
            output_cost = self._dfStrategies['E[Cost]'].astype(float).round(cost_digits)

        # format effect column
        if effect_digits == 0:
            output_effect = self._dfStrategies['E[Effect]'].astype(int)
        else:
            output_effect = self._dfStrategies['E[Effect]'].astype(float).round(effect_digits)

        # create output dataframe
        # dataframe of estimates (without intervals)
        output_estimates = pd.DataFrame(
            {'Name': self._dfStrategies['Name'],
             'E[Cost]': output_cost,
             'E[Effect]': output_effect,
             'E[dCost]': self._dfStrategies['E[dCost]'],
             'E[dEffect]': self._dfStrategies['E[dEffect]'],
             'ICER': self._dfStrategies['ICER']
             })

        # decide about what interval to return and create table out_intervals
        if interval == Interval.PREDICTION:
            # create the dataframe
            out_intervals_PI = pd.DataFrame(index=self._dfStrategies.index,
                columns=['Name', 'Cost_I', 'Effect_I', 'Dominated'])
            # initialize incremental cost and health and ICER with -
            out_intervals_PI['dCost_I'] = '-'
            out_intervals_PI['dEffect_I'] = '-'
            out_intervals_PI['ICER_I'] = '-'

            # calculate prediction intervals for cost and effect
            for i in self._dfStrategies.index:
                # populated name, dominated, cost and effect PI columns
                out_intervals_PI.loc[i, 'Name'] = self._strategies[i].name
                out_intervals_PI.loc[i, 'Dominated'] = self._dfStrategies.loc[i, 'Dominated']

                # prediction interval of cost
                temp_c = Stat.SummaryStat("",self._strategies[i].costObs).get_PI(alpha)
                out_intervals_PI.loc[i, 'Cost_I'] = temp_c

                # prediction interval of effect
                temp_e = Stat.SummaryStat("",self._strategies[i].effectObs).get_PI(alpha)
                out_intervals_PI.loc[i, 'Effect_I'] = temp_e

            # calculate prediction intervals for incremental cost and effect and ICER
            if self._ifPaired:
                # paired: populated dcost, deffect and ICER PI columns
                for i in range(1, n_frontier_strategies):
                    # calculate the prediction interval of incremental cost
                    PI_PariedDiffCost = Stat.DifferenceStatPaired("",
                        self._strategies[frontier_strategies.index[i]].costObs,
                        self._strategies[frontier_strategies.index[i-1]].costObs).get_PI(alpha)
                    # add the interval to the data frame
                    out_intervals_PI.loc[frontier_strategies.index[i], 'dCost_I'] = \
                        PI_PariedDiffCost

                    # calculate the prediction interval of incremental effect
                    PI_PariedDiffEffect = Stat.DifferenceStatPaired("",
                        self._strategies[frontier_strategies.index[i]].effectObs,
                        self._strategies[frontier_strategies.index[i-1]].effectObs).get_PI(alpha)
                    # add the interval to the data frame
                    out_intervals_PI.loc[frontier_strategies.index[i], 'dEffect_I'] = \
                        PI_PariedDiffEffect

                    # calculate the prediction interval of ICER
                    PI_PairedICER = ICER_paired("", self._strategies[frontier_strategies.index[i]].costObs,
                                            self._strategies[frontier_strategies.index[i]].effectObs,
                                            self._strategies[frontier_strategies.index[i-1]].costObs,
                                            self._strategies[frontier_strategies.index[i-1]].effectObs)
                    # add the interval to the data frame
                    out_intervals_PI.loc[frontier_strategies.index[i], 'ICER_I'] = \
                        PI_PairedICER.get_PI(alpha)

            else: # if not paired
                # indp: populated dcost, deffect and ICER PI columns
                for i in range(1, n_frontier_strategies):
                    # calculate the prediction interval of incremental cost
                    PI_indpDiffCost = Stat.DifferenceStatIndp("",
                        self._strategies[frontier_strategies.index[i]].costObs,
                        self._strategies[frontier_strategies.index[i - 1]].costObs).get_PI(alpha)
                    # add the interval to the data frame
                    out_intervals_PI.loc[frontier_strategies.index[i], 'dCost_I'] = \
                        PI_indpDiffCost

                    # calculate the prediction interval of incremental effect
                    PI_indpDiffEffect = Stat.DifferenceStatIndp("",
                        self._strategies[frontier_strategies.index[i]].effectObs,
                        self._strategies[frontier_strategies.index[i - 1]].effectObs).get_PI(alpha)
                    # add the interval to the data frame
                    out_intervals_PI.loc[frontier_strategies.index[i], 'dEffect_I'] = \
                        PI_indpDiffEffect

                    # calculate the prediction interval of ICER
                    PI_indpICER = ICER_indp("", self._strategies[frontier_strategies.index[i]].costObs,
                                            self._strategies[frontier_strategies.index[i]].effectObs,
                                            self._strategies[frontier_strategies.index[i - 1]].costObs,
                                            self._strategies[frontier_strategies.index[i - 1]].effectObs)
                    # add the interval to the data frame
                    out_intervals_PI.loc[frontier_strategies.index[i], 'ICER_I'] = \
                        PI_indpICER.get_PI(alpha)

            out_intervals = out_intervals_PI[['Name', 'Cost_I', 'Effect_I', 'dCost_I',
                                                   'dEffect_I', 'ICER_I']]

        elif interval == Interval.CONFIDENCE:
            # create the datafram
            out_intervals_CI = pd.DataFrame(index=self._dfStrategies.index,
                columns=['Name', 'Cost_I', 'Effect_I', 'Dominated'])
            out_intervals_CI['dCost_I'] = '-'
            out_intervals_CI['dEffect_I'] = '-'
            out_intervals_CI['ICER_I'] = '-'

            # calculate confidence intervals for cost and effect
            for i in self._dfStrategies.index:
                # populated name, dominated, cost and effect CI columns
                out_intervals_CI.loc[i, 'Name'] = self._strategies[i].name
                out_intervals_CI.loc[i, 'Dominated'] = self._dfStrategies.loc[i, 'Dominated']

                # confidence interval of cost
                temp_c = Stat.SummaryStat("",self._strategies[i].costObs).get_t_CI(alpha)
                out_intervals_CI.loc[i, 'Cost_I'] = temp_c

                # confidence interval of effect
                temp_e = Stat.SummaryStat("",self._strategies[i].effectObs).get_t_CI(alpha)
                out_intervals_CI.loc[i, 'Effect_I'] = temp_e

            # calculate confidence intervals for incremental cost and effect and ICER
            if self._ifPaired:
                # paired: populated dcost, deffect and ICER CI columns
                # for difference statistics, and CI are using t_CI
                # for ratio, using bootstrap with 1000 number of samples
                for i in range(1, n_frontier_strategies):
                    # calculate the confidence interval of incremental cost
                    CI_PariedDiffCost = Stat.SummaryStat("",
                        self._strategies[frontier_strategies.index[i]].costObs -
                        self._strategies[frontier_strategies.index[i-1]].costObs).get_t_CI(alpha)
                    # add the interval to the data frame
                    out_intervals_CI.loc[frontier_strategies.index[i], 'dCost_I'] = \
                        CI_PariedDiffCost

                    # calculate the confidence interval of incremental effect
                    CI_PariedDiffEffect = Stat.SummaryStat("",
                        self._strategies[frontier_strategies.index[i]].effectObs -
                        self._strategies[frontier_strategies.index[i-1]].effectObs).get_t_CI(alpha)

                    # add the interval to the data frame
                    out_intervals_CI.loc[frontier_strategies.index[i], 'dEffect_I'] = \
                        CI_PariedDiffEffect

                    # calculate the confidence interval of incremental ICER
                    CI_PairedICER = ICER_paired("", self._strategies[frontier_strategies.index[i]].costObs,
                                            self._strategies[frontier_strategies.index[i]].effectObs,
                                            self._strategies[frontier_strategies.index[i-1]].costObs,
                                            self._strategies[frontier_strategies.index[i-1]].effectObs)

                    # add the interval to the data frame
                    out_intervals_CI.loc[frontier_strategies.index[i], 'ICER_I'] = \
                        CI_PairedICER.get_CI(alpha, 1000)

            else: # if not paired
                # indp: populated dcost, deffect and ICER CI columns
                for i in range(1, n_frontier_strategies):
                    # calculate the confidence interval of incremental cost
                    CI_indpDiffCost = Stat.DifferenceStatIndp("",
                        self._strategies[frontier_strategies.index[i]].costObs,
                        self._strategies[frontier_strategies.index[i - 1]].costObs).get_t_CI(alpha)

                    # add the interval to the data frame
                    out_intervals_CI.loc[frontier_strategies.index[i], 'dCost_I'] = \
                        CI_indpDiffCost

                    # calculate the confidence interval of incremental effect
                    CI_indpDiffEffect = Stat.DifferenceStatIndp("",
                        self._strategies[frontier_strategies.index[i]].effectObs,
                        self._strategies[frontier_strategies.index[i - 1]].effectObs).get_t_CI(alpha)

                    # add the interval to the data frame
                    out_intervals_CI.loc[frontier_strategies.index[i], 'dEffect_I'] = \
                        CI_indpDiffEffect

                    # calculate the confidence interval of incremental ICER
                    CI_indpICER = ICER_indp("", self._strategies[frontier_strategies.index[i]].costObs,
                                            self._strategies[frontier_strategies.index[i]].effectObs,
                                            self._strategies[frontier_strategies.index[i - 1]].costObs,
                                            self._strategies[frontier_strategies.index[i - 1]].effectObs)

                    # add the interval to the data frame
                    out_intervals_CI.loc[frontier_strategies.index[i], 'ICER_I'] = \
                        CI_indpICER.get_CI(alpha, 1000)

            out_intervals = out_intervals_CI[['Name', 'Cost_I', 'Effect_I', 'dCost_I',
                                                   'dEffect_I', 'ICER_I']]

        else:
            # define column order and write csv
            output_estimates[['Name', 'E[Cost]', 'E[Effect]', 'E[dCost]', 'E[dEffect]', 'ICER']].\
                to_csv("CETable.csv", encoding='utf-8', index=False)
            # no need to calculate intervals, break of function
            return None

        # merge estimates and intervals together
        out_table = pd.DataFrame(
            {'Name': self._dfStrategies['Name'],
             'E[Cost]': output_estimates['E[Cost]'],
             'E[Effect]': output_estimates['E[Effect]'],
             'E[dCost]': output_estimates['E[dCost]'],
             'E[dEffect]': output_estimates['E[dEffect]'],
             'ICER': output_estimates['ICER']
             })

        # put estimates and intervals together
        for i in self._dfStrategies.index:
            out_table.loc[i, 'E[Cost]'] = \
                FormatFunc.format_estimate_interval(output_estimates.loc[i, 'E[Cost]'],
                                                    out_intervals.loc[i,'Cost_I'],
                                                    cost_digits, format=FormatFunc.FormatNumber.NUMBER)
            out_table.loc[i, 'E[Effect]'] = \
                FormatFunc.format_estimate_interval(output_estimates.loc[i, 'E[Effect]'],
                                                    out_intervals.loc[i,'Effect_I'],
                                                    effect_digits, format=FormatFunc.FormatNumber.NUMBER)

        # add the incremental and ICER estimates and intervals
        for i in range(1, n_frontier_strategies):

            out_table.loc[frontier_strategies.index[i], 'E[dCost]'] = \
                FormatFunc.format_estimate_interval(output_estimates.loc[frontier_strategies.index[i], 'E[dCost]'],
                                                    out_intervals.loc[frontier_strategies.index[i],'dCost_I'],
                                                    cost_digits, format=FormatFunc.FormatNumber.NUMBER)

            out_table.loc[frontier_strategies.index[i], 'E[dEffect]'] = \
                FormatFunc.format_estimate_interval(output_estimates.loc[frontier_strategies.index[i], 'E[dEffect]'],
                                                    out_intervals.loc[frontier_strategies.index[i],'dEffect_I'],
                                                    effect_digits, format=FormatFunc.FormatNumber.NUMBER)

            out_table.loc[frontier_strategies.index[i], 'ICER'] = \
                FormatFunc.format_estimate_interval(output_estimates.loc[frontier_strategies.index[i], 'ICER'],
                                                    out_intervals.loc[frontier_strategies.index[i],'ICER_I'],
                                                    icer_digits, format=FormatFunc.FormatNumber.NUMBER)

        # define column order and write csv
        out_table[['Name', 'E[Cost]', 'E[Effect]', 'E[dCost]', 'E[dEffect]', 'ICER']].to_csv(
            file_name+'.csv', encoding='utf-8', index=False)


class CBA(_EconEval):
    """ class for doing cost-benefit analysis """

    def __init__(self, strategies, if_paired):
        """
        :param strategies: the list of strategies (assumes that the first strategy represents the "base" strategy)
        :param if_paired: indicate whether the strategies are paired
        """
        _EconEval.__init__(self, strategies, if_paired)

    def graph_deltaNMB_lines(self, min_wtp, max_wtp, title,
                        x_label, y_label, interval=Interval.NO_INTERVAL, transparency=0.4,
                        show_legend=False, figure_size=6):
        """
        :param min_wtp: minimum willingness-to-pay (or cost-effectiveness threshold) on the x-axis
        :param max_wtp: maximum willingness-to-pay (or cost-effectiveness threshold) on the x-axis
        :param title: title
        :param x_label: x-axis label
        :param y_label: y-axis label
        :param interval: type of interval to show. Can take values from
                Interval.NO_INTERVAL, Interval.CONFIDENCE, Interval.PREDICTION
        :param transparency: transparency of intervals (0.0 transparent through 1.0 opaque)
        :param show_legend: set true to show legend
        :param figure_size: size of the figure
        """
        # set x-axis
        x_values = np.arange(min_wtp, max_wtp, (max_wtp-min_wtp)/100.0)

        # initialize plot
        plt.figure(figsize=(figure_size, figure_size))

        # if paired
        if self._ifPaired:
            # nmb_paired = []
            for strategy_i, color in zip(self._strategies[1:], cm.rainbow(np.linspace(0, 1, self._n-1))):
                # create NMB_paired objects
                nmbi = NMB_paired(strategy_i.name, strategy_i.costObs,
                                  strategy_i.effectObs, self._strategies[0].costObs,
                                  self._strategies[0].effectObs)

                # nmb_paired.append(nmbi)

                # get the NMB values for each wtp
                y_values = [nmbi.get_NMB(x) for x in x_values]
                # plot line
                plt.plot(x_values, y_values, c=color, alpha=transparency, label=strategy_i.name)

                # get confidence interval and plot
                if interval == Interval.CONFIDENCE:
                    y_ci = [nmbi.get_CI(x, alpha=0.05) for x in x_values]
                    # reshape confidence interval to plot
                    xerr = np.array([p[1] for p in y_ci]) - y_values
                    yerr = y_values - np.array([p[0] for p in y_ci])
                    plt.fill_between(x_values, y_values - yerr, y_values + xerr, color="grey", alpha=transparency)
                    #plt.errorbar(x_values, y_values, np.array([xerr, yerr]), color=color,
                    #             alpha=transparency, label=strategy_i.name)

                # get prediction interval and plot
                if interval == Interval.PREDICTION:
                    y_ci = [nmbi.get_PI(x, alpha=0.05) for x in x_values]
                    # reshape confidence interval to plot
                    xerr = np.array([p[1] for p in y_ci]) - y_values
                    yerr = y_values - np.array([p[0] for p in y_ci])
                    plt.fill_between(x_values, y_values - yerr, y_values + xerr, color="grey", alpha=transparency)

        # if unpaired
        elif self._ifPaired==False:
            # nmb_indp = []
            for strategy_i, color in zip(self._strategies[1:], cm.rainbow(np.linspace(0, 1, self._n-1))):
                # create NMB_indp objects
                nmbi = NMB_indp(strategy_i.name, strategy_i.costObs,
                                  strategy_i.effectObs, self._strategies[0].costObs,
                                  self._strategies[0].effectObs)

                # nmb_indp.append(nmbi)

                # get the NMB values for each wtp
                y_values = [nmbi.get_NMB(x) for x in x_values]
                # plot line
                plt.plot(x_values, y_values, c=color, alpha=transparency, label=strategy_i.name)

                # get confidence interval and plot
                if interval == Interval.CONFIDENCE:
                    y_ci = [nmbi.get_CI(x, alpha=0.05) for x in x_values]
                    # reshape confidence interval to plot
                    xerr = np.array([p[1] for p in y_ci]) - y_values
                    yerr = y_values - np.array([p[0] for p in y_ci])
                    plt.fill_between(x_values, y_values - yerr, y_values + xerr, color="grey", alpha=transparency)

                # get prediction interval and plot
                if interval == Interval.PREDICTION:
                    y_ci = [nmbi.get_PI(x, alpha=0.05) for x in x_values]
                    # reshape confidence interval to plot
                    xerr = np.array([p[1] for p in y_ci]) - y_values
                    yerr = y_values - np.array([p[0] for p in y_ci])
                    plt.fill_between(x_values, y_values - yerr, y_values + xerr, color="grey", alpha=transparency)

        if show_legend:
            plt.legend()

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim([min_wtp, max_wtp])

        vals_y, labs_y = plt.yticks()
        vals_x, labs_x = plt.xticks()
        plt.yticks(vals_y, ['{:,.{prec}f}'.format(x, prec=0) for x in vals_y])
        plt.xticks(vals_x, ['{:,.{prec}f}'.format(x, prec=0) for x in vals_x])

        plt.axhline(y=0, c='k', ls='--', linewidth=0.5)
        plt.axvline(x=0, c='k', ls='--', linewidth=0.5)

        plt.show()


class ComparativeEconMeasure:
    def __init__(self, name, cost_new, health_new, cost_base, health_base):
        """
        :param cost_new: (list or numpy.array) cost data for the new strategy
        :param health_new: (list or numpy.array) health data for the new strategy
        :param cost_base: (list or numpy.array) cost data for the base line
        :param health_base: (list or numpy.array) health data for the base line
        """
        np.random.seed(1)

        self._name = name
        self._costNew = cost_new          # cost data for the new strategy
        self._healthNew = health_new      # health data for the new strategy
        self._costBase = cost_base        # cost data for teh base line
        self._healthBase = health_base    # health data for the base line

        # convert input data to numpy.array if needed
        if type(cost_new) == list:
            self._costNew = np.array(cost_new)
        if type(health_new) == list:
            self._healthNew = np.array(health_new)
        if type(cost_base) == list:
            self._costBase = np.array(cost_base)
        if type(health_base) == list:
            self._healthBase = np.array(health_base)

        # calculate the difference in average cost and health
        self._delta_ave_cost = np.average(self._costNew) - np.average(self._costBase)
        self._delta_ave_health = np.average(self._healthNew) - np.average(self._healthBase)


class _ICER(ComparativeEconMeasure):
    def __init__(self, name, cost_new, health_new, cost_base, health_base):
        """
        :param cost_new: (list or numpy.array) cost data for the new strategy
        :param health_new: (list or numpy.array) health data for the new strategy
        :param cost_base: (list or numpy.array) cost data for the base line
        :param health_base: (list or numpy.array) health data for the base line
        """
        # initialize the base class
        ComparativeEconMeasure.__init__(self, name, cost_new, health_new, cost_base, health_base)
        # calcualte ICER
        self._ICER = self._delta_ave_cost/self._delta_ave_health

    def get_ICER(self):
        """ return ICER """
        return self._ICER

    def get_CI(self, alpha, num_bootstrap_samples):
        """
        :param alpha: significance level, a value from [0, 1]
        :param num_bootstrap_samples: number of bootstrap samples
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

    def __init__(self, name, cost_new, health_new, cost_base, health_base):
        """
        :param cost_new: (list or numpy.array) cost data for the new strategy
        :param health_new: (list or numpy.array) health data for the new strategy
        :param cost_base: (list or numpy.array) cost data for the base line
        :param health_base: (list or numpy.array) health data for the base line
        """
        # initialize the base class
        _ICER.__init__(self, name, cost_new, health_new, cost_base, health_base)

        # incremental observations
        self._deltaCost = self._costNew - self._costBase
        self._deltaHealth = self._healthNew - self._healthBase

        # create a ratio stat
        self._ratio_stat = np.divide(self._deltaCost, self._deltaHealth)

    def get_CI(self, alpha, num_bootstrap_samples):

        # bootstrap algorithm
        ICERs = np.zeros(num_bootstrap_samples)
        for i in range(num_bootstrap_samples):
            # because cost and health are paired as one observation in natural,
            # so do delta cost and delta health, should sample them together
            index = np.random.choice(range(len(self._deltaCost)), size=len(self._deltaCost), replace=True)
            d_cost = self._deltaCost[index]
            d_health = self._deltaHealth[index]

            ave_d_cost = np.average(d_cost)
            ave_d_health = np.average(d_health)

            # assert all the means should not be 0
            if np.average(ave_d_health) == 0:
                raise ValueError('invalid value of mean of y, the ratio is not computable')

            ICERs[i] = ave_d_cost/ave_d_health - self._ICER

        return self._ICER - np.percentile(ICERs, [100 * (1 - alpha / 2.0), 100 * alpha / 2.0])

    def get_PI(self, alpha):
        return np.percentile(self._ratio_stat, [100*alpha/2.0, 100*(1-alpha/2.0)])


class ICER_indp(_ICER):

    def __init__(self, name, cost_new, health_new, cost_base, health_base):
        """
        :param cost_new: (list or numpy.array) cost data for the new strategy
        :param health_new: (list or numpy.array) health data for the new strategy
        :param cost_base: (list or numpy.array) cost data for the base line
        :param health_base: (list or numpy.array) health data for the base line
        """
        # initialize the base class
        _ICER.__init__(self, name, cost_new, health_new, cost_base, health_base)

    def get_CI(self, alpha, num_bootstrap_samples):
        """
        :param alpha: significance level, a value from [0, 1]
        :param num_bootstrap_samples: number of bootstrap samples
        :return: confidence interval in the format of list [l, u]
        """
        ICERs = np.zeros(num_bootstrap_samples)

        n_new = len(self._costNew)
        n_base = len(self._costBase)
        for i in range(num_bootstrap_samples):
            index_new_i = np.random.choice(range(n_new), size=n_new, replace=True)
            cost_new_i = self._costNew[index_new_i]
            health_new_i = self._healthNew[index_new_i]

            index_base_i = np.random.choice(range(n_base), size=n_base, replace=True)
            cost_base_i = self._costBase[index_base_i]
            health_base_i = self._healthBase[index_base_i]

            # for each random sample of (c2,h2), (c1,h1)
            # calculate ICER = (E(c2)-E(c1))/(E(h2)-E(h1))
            r_temp = np.mean(cost_new_i - cost_base_i)/np.mean(health_new_i - health_base_i)
            ICERs[i] = np.mean(r_temp)

        return np.percentile(ICERs, [100*alpha/2.0, 100*(1-alpha/2.0)])

    def get_PI(self, alpha):
        """
        :param alpha: significance level, a value from [0, 1]
        :return: prediction interval in the format of list [l, u]
        """

        n = max(len(self._costNew), len(self._costBase))

        # calculate element-wise ratio as sample of ICER
        index_new_0 = np.random.choice(range(n), size=n, replace=True)
        cost_new_0 = self._costNew[index_new_0]
        health_new_0 = self._healthNew[index_new_0]

        index_base_0 = np.random.choice(range(n), size=n, replace=True)
        cost_base_0 = self._costBase[index_base_0]
        health_base_0 = self._healthBase[index_base_0]

        sum_stat_sample_ratio = np.divide((cost_new_0 - cost_base_0), (health_new_0 - health_base_0))

        return np.percentile(sum_stat_sample_ratio, [100*alpha/2.0, 100*(1-alpha/2.0)])


class _NMB(ComparativeEconMeasure):
    def __init__(self, name, cost_new, health_new, cost_base, health_base):
        """
        :param cost_new: (list or numpy.array) cost data for the new strategy
        :param health_new: (list or numpy.array) health data for the new strategy
        :param cost_base: (list or numpy.array) cost data for the base line
        :param health_base: (list or numpy.array) health data for the base line
        """
        # initialize the base class
        ComparativeEconMeasure.__init__(self, name, cost_new, health_new, cost_base, health_base)

    def get_NMB(self, wtp):
        """
        :param wtp: willingness-to-pay
        :returns: the net monetary benefit at the provided willingness-to-pay value
        """
        return wtp * self._delta_ave_health - self._delta_ave_cost

    def get_CI(self, wtp, alpha):
        """
        :param wtp: willingness-to-pay value
        :param alpha: significance level, a value from [0, 1]
        :return: confidence interval in the format of list [l, u]
        """
        # abstract method to be overridden in derived classes to process an event
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")

    def get_PI(self, wtp, alpha):
        """
        :param wtp: willingness-to-pay value
        :param alpha: significance level, a value from [0, 1]
        :return: percentile interval in the format of list [l, u]
        """
        # abstract method to be overridden in derived classes to process an event
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")


class NMB_paired(_NMB):

    def __init__(self, name, cost_new, health_new, cost_base, health_base):
        """
        :param cost_new: (list or numpy.array) cost data for the new strategy
        :param health_new: (list or numpy.array) health data for the new strategy
        :param cost_base: (list or numpy.array) cost data for the base line
        :param health_base: (list or numpy.array) health data for the base line
        """
        _NMB.__init__(self, name, cost_new, health_new, cost_base, health_base)

        # incremental observations
        self._deltaCost = self._costNew - self._costBase
        self._deltaHealth = self._healthNew - self._healthBase

    def get_CI(self, wtp, alpha):

        # create a summary statistics
        stat = Stat.SummaryStat(self._name, wtp * self._deltaHealth - self._deltaCost)
        return stat.get_t_CI(alpha)

    def get_PI(self, wtp, alpha):

        # create a summary statistics
        stat = Stat.SummaryStat(self._name, wtp * self._deltaHealth - self._deltaCost)
        return stat.get_PI(alpha)


class NMB_indp(_NMB):

    def __init__(self, name, cost_new, health_new, cost_base, health_base):
        """
        :param cost_new: (list or numpy.array) cost data for the new strategy
        :param health_new: (list or numpy.array) health data for the new strategy
        :param cost_base: (list or numpy.array) cost data for the base line
        :param health_base: (list or numpy.array) health data for the base line
        """
        _NMB.__init__(self, name, cost_new, health_new, cost_base, health_base)

    def get_CI(self, wtp, alpha):
        # reform 2 independent variables to pass in DifferenceStatIndp class
        stat_new = wtp * self._healthNew - self._costNew
        stat_base = wtp * self._healthBase - self._costBase
        # to get CI for stat_new - stat_base
        diff_stat = Stat.DifferenceStatIndp(self._name, stat_new, stat_base)
        return diff_stat.get_t_CI(alpha)

    def get_PI(self, wtp, alpha):
        # reform 2 independent variables to pass in DifferenceStatIndp class
        stat_new = wtp * self._healthNew - self._costNew
        stat_base = wtp * self._healthBase - self._costBase

        # to get PI for stat_new - stat_base
        diff_stat = Stat.DifferenceStatIndp(self._name, stat_new, stat_base)
        return diff_stat.get_PI(alpha)
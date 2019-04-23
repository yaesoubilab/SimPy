import string

import SimPy.InOutFunctions as IO
from SimPy.EconEvalClasses import *

NUM_OF_BOOTSTRAPS = 1000  # number of bootstrap samples to calculate confidence intervals for ICER


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

        self.idx = 0        # index of the strategy
        self.name = name
        self.ifDominated = False
        self.color = color

        self.costObs = None     # (list) cost observations
        self.dCostObs = None    # (list) cost observations with respect to base
        self.incCostObs = None  # (list) incremental cost observations
        self.cost = None        # summary statistics for cost
        self.dCost = None       # summary statistics for cost with respect to base
        self.incCost = None     # summary statistics for incremental cost

        self.effectObs = None       # (list) effect observations
        self.dEffectObs = None      # (list) effect observations with respect to base
        self.incEffectObs = None    # (list) incremental effect observations
        self.effectObs = None       # effect observations
        self.effect = None          # summary statistics for effect
        self.dEffect = None         # summary statistics for effect with respect to base
        self.incEffect = None       # summary statistics for incremental effect

        self.cer = None         # cost-effectiveness ratio with respect to base
        self.icer = None        # icer summary statistics

        if type(cost_obs) is list:
            self.costObs = np.array(cost_obs)
        else:
            self.costObs = cost_obs
        if type(effect_obs) is list:
            self.effectObs = np.array(effect_obs)
        else:
            self.effectObs = effect_obs

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

    def get_cost_err_interval(self, interval_type, alpha, multiplier=1):
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
        self.__find_shifted_strategies()

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
                all strategies
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

    def add_ce_plane_to_ax(self, ax, add_clouds=True, show_legend=True,
                           center_s=75, cloud_s=25, transparency=0.1,
                           cost_multiplier=1, effect_multiplier=1):

        # find the frontier (x, y)'s
        frontier_d_effect = []
        frontier_d_costs = []
        for s in self.get_strategies_on_frontier():
            frontier_d_effect.append(s.dEffect.get_mean()*effect_multiplier)
            frontier_d_costs.append(s.dCost.get_mean()*cost_multiplier)

        # add the frontier line
        ax.plot(frontier_d_effect, frontier_d_costs,
                c='k',  # color
                alpha=0.6,  # transparency
                linewidth=2,  # line width
                label="Frontier")  # label to show in the legend

        for s in self.strategies:
            ax.scatter(s.dEffect.get_mean()*effect_multiplier, s.dCost.get_mean()*cost_multiplier,
                       c=s.color,  # color
                       alpha=1,  # transparency
                       marker='o',  # markers
                       s=center_s,  # marker size
                       label=s.name  # name to show in the legend
                       )
            if add_clouds: # idx >= 2:
                ax.scatter(s.dEffectObs*effect_multiplier, s.dCostObs*cost_multiplier,
                           c=s.color,  # color of dots
                           alpha=transparency,  # transparency of dots
                           s=cloud_s,  # size of dots
                           zorder=1
                           )

        if show_legend:
            ax.legend()
        ax.axhline(y=0, c='k', linewidth=0.5)
        ax.axvline(x=0, c='k', linewidth=0.5)

    def show_CE_plane(self, add_clouds=True):

        fig, ax = plt.subplots()

        # add the cost-effectiveness plane
        self.add_ce_plane_to_ax(ax=ax, add_clouds=add_clouds)

        fig.show()

    def create_pairwise_ceas(self):
        """
        creates a list of list for pairwise cost-effectiveness analysis
        """

        # create CEA's for all pairs

        self._pairwise_ceas = []
        for s_base in self.strategies:
            list_ceas = []
            for s_new in self.strategies[1:]:

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

        # since the relative performance of strategies (relative costs and relative effects) have changed.
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

    def plot_pairwise_ceas(self, figure_size=None, font_size=6, show_subplot_labels=False,
                           effect_label='', cost_label='',
                           center_s=50, cloud_s=25, transparency=0.2,
                           x_range=None, y_range=None,
                           cost_multiplier=1, effect_multiplier=1,
                           column_titles=None, row_titles=None,
                           file_name='pairwise_CEA.png'):

        # set default properties
        plt.rc('font', size=font_size) # fontsize of texts
        plt.rc('axes', titlesize=font_size)  # fontsize of the figure title
        plt.rc('axes', titleweight='semibold')  # fontweight of the figure title

        # plot each panel
        f, axarr = plt.subplots(nrows=self._n, ncols=self._n-1,
                                sharex=True, sharey=True, figsize=figure_size)

        # show subplot labels
        Y_LABEL_COORD_X = -0.05     # increase to move right
        # if show_subplot_labels:
        #     axs = axarr.flat
        #     for n, ax in enumerate(axs):
        #         ax.text(Y_LABEL_COORD_X-0.05, 1.05, string.ascii_uppercase[n]+')', transform=ax.transAxes,
        #                 size=DEFAULT_FONT_SIZE+1, weight='bold')
        n = 0
        for i in range(self._n):
            for j in range(self._n-1):
                # get the current axis
                ax = axarr[i, j]

                # if i == j + 1:
                #     pass
                #     ax.axis('off')
                # else:
                if show_subplot_labels:
                    ax.text(Y_LABEL_COORD_X - 0.05, 1.05, string.ascii_uppercase[n] + ')',
                            transform=ax.transAxes,
                            size=font_size + 1, weight='bold')

                # add titles for the figures in the first row
                if i == 0:
                    if column_titles is None:
                        ax.set_title(self.strategies[j+1].name)
                    else:
                        ax.set_title(column_titles[j])

                # add y_labels for the figures in the first column
                if j == 0:
                    if row_titles is None:
                        ax.set_ylabel(self.strategies[i].name, fontweight='bold')
                    else:
                        ax.set_ylabel(row_titles[i], fontweight='bold')

                if x_range is not None:
                    ax.set_xlim(x_range)
                if y_range is not None:
                    ax.set_ylim(y_range)

                cea = CEA(strategies=[self.strategies[i], self.strategies[j+1]],
                          if_paired=self._ifPaired,
                          health_measure=self._healthMeasure,
                          if_reset_strategies=True)

                cea.add_ce_plane_to_ax(ax=ax, show_legend=False,
                                       center_s=center_s,
                                       cloud_s=cloud_s,
                                       transparency=transparency,
                                       cost_multiplier=cost_multiplier,
                                       effect_multiplier=effect_multiplier)

                # add ICER
                text = ''
                if i != j+1 and cea.strategies[1].ifDominated:
                    text = 'Dominated'
                elif cea.strategies[1].dCost.get_mean() < 0 and cea.strategies[1].dEffect.get_mean() > 0:
                    text = 'Cost-saving'
                elif cea.strategies[1].icer is not None:
                    text = F.format_number(cea.strategies[1].icer.get_ICER(), deci=1, format='$')
                ax.text(0.95, 0.95, text, transform=ax.transAxes, fontsize=6,
                        va='top', ha='right')

                n += 1
                # remove unnecessary labels for shared axis
                # if i < self._n - 1:
                #     ax.set(xlabel='')
                if j > 0:
                    ax.set(ylabel='')

        f.text(0.55, 0, effect_label, ha='center', va='center', fontweight='bold')
        f.text(0.99, 0.5, cost_label, va='center', rotation=-90, fontweight='bold')
        f.show()
        f.savefig(file_name, bbox_inches='tight', dpi=300)

        # since the relative performance of strategies (relative costs and relative effects) have changed.
        self._ifFrontierIsCalculated = False

    def __find_shifted_strategies(self):
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
                    s.icer = ICER_paired(name='ICER of {} relative to {}'.format(s.name, s_before.name),
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
                    s.icer = ICER_indp(name='ICER of {} relative to {}'.format(s.name, s_before.name),
                                       costs_new=s.costObs,
                                       effects_new=s.effectObs,
                                       costs_base=s_before.costObs,
                                       effects_base=s_before.effectObs,
                                       health_measure=self._healthMeasure)


class _Curve:
    def __init__(self, label, color, wtps):
        self.label = label
        self.color = color
        self.wtps = wtps


class NMBCurve(_Curve):

    def __init__(self, label, color, wtps, ys, l_errs, u_errs):

        _Curve.__init__(self, label, color, wtps)
        self.ys = ys            # net monetary benefits over a range of wtp values
        self.l_errs = l_errs    # lower error length of NMB over a range of wtp values
        self.u_errs = u_errs    # upper error length of NMB over a range of wtp values


class AcceptabilityCurve(_Curve):

    def __init__(self, label, color, wtps):

        _Curve.__init__(self, label, color, wtps)
        self.probs = []      # probability that this strategy is optimal over a range of wtp values
        self.optProbs = []
        self.optWTPs = []


class CBA(_EconEval):
    """ class for doing cost-benefit analysis """

    def __init__(self, strategies, if_paired, health_measure='u'):
        """
        :param strategies: the list of strategies (assumes that the first strategy represents the "base" strategy)
        :param if_paired: indicate whether the strategies are paired
        :param health_measure: (string) choose 'u' if higher "effect" implies better health
        (e.g. when QALY is used) and set to 'd' if higher "effect" implies worse health
        (e.g. when DALYS is used)
        """
        _EconEval.__init__(self, strategies=strategies,
                           if_paired=if_paired,
                           health_measure=health_measure)

        self.nmbCurves = []  # list of NMB curves
        self.optStrategyIdx = []  # index of strategy with highest net-monetary benefit over the wtp range
        self.acceptabilityCurves = []  # the list of acceptability curves

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

        self.nmbCurves = []  # list of NMB curves

        # wtp values at which NMB should be evaluated
        wtp_values = np.linspace(min_wtp, max_wtp, num=NUM_WTPS_FOR_NMB_CURVES, endpoint=True)

        # decide about the color of each curve
        rainbow_colors = cm.rainbow(np.linspace(0, 1, self._n - 1))
        colors = []
        for i, s in enumerate(self.strategies[1:]):
            if s.color:
                colors.append(s.color)
            else:
                colors.append(rainbow_colors[i])

        # create the NMB curves
        for strategy_i, color in zip(self.strategies[1:], colors):

            if self._ifPaired:
                # create a paired NMB object
                paired_nmb = NMB_paired(name=strategy_i.name,
                                        costs_new=strategy_i.costObs,
                                        effects_new=strategy_i.effectObs,
                                        costs_base=self.strategies[0].costObs,
                                        effects_base=self.strategies[0].effectObs,
                                        health_measure=self._healthMeasure)

                # get the NMB values for each wtp
                y_values, l_err, u_err = self.__get_ys_lerrs_uerrs(
                    nmb=paired_nmb, wtps=wtp_values, interval_type=interval_type
                )

            else:
                # create an indp NMB object
                ind_nmb = NMB_indp(name=strategy_i.name,
                                   costs_new=strategy_i.costObs,
                                   effects_new=strategy_i.effectObs,
                                   costs_base=self.strategies[0].costObs,
                                   effects_base=self.strategies[0].effectObs,
                                   health_measure=self._healthMeasure)

                # get the NMB values for each wtp
                y_values, l_err, u_err = self.__get_ys_lerrs_uerrs(
                    nmb=ind_nmb, wtps=wtp_values, interval_type=interval_type
                )

            # make a NMB curve
            self.nmbCurves.append(NMBCurve(label=strategy_i.name,
                                           color=color,
                                           wtps=wtp_values,
                                           ys=y_values,
                                           l_errs=l_err,
                                           u_errs=u_err)
                                  )

        # find the optimal strategy for each wtp value
        for wtp_idx, wtp in enumerate(wtp_values):

            max_value = float('-inf')
            max_idx = 0
            for s_idx in range(len(self.nmbCurves)):
                if self.nmbCurves[s_idx].ys[wtp_idx] > max_value:
                    max_value = self.nmbCurves[s_idx].ys[wtp_idx]
                    max_idx = s_idx

            # store the optimal the index of the strategy
            if max_value > 0:
                self.optStrategyIdx.append(max_idx + 1)
            else:
                self.optStrategyIdx.append(0)

    def make_acceptability_curves(self, min_wtp, max_wtp):
        """
        prepares the information needed to plot the cost-effectiveness acceptability curves
        :param min_wtp: minimum willingness-to-pay (or cost-effectiveness threshold) on the x-axis
        :param max_wtp: maximum willingness-to-pay (or cost-effectiveness threshold) on the x-axis
        """

        if not self._ifPaired:
            raise ValueError('Calculating the acceptability curves when outcomes are not paried'
                             'across strategies is not implemented.')

        # wtp values at which NMB should be evaluated
        wtp_values = np.linspace(min_wtp, max_wtp, num=NUM_WTPS_FOR_NMB_CURVES, endpoint=True)

        # initialize acceptability curves
        self.acceptabilityCurves = []
        for s in self.strategies:
            self.acceptabilityCurves.append(AcceptabilityCurve(label=s.name,
                                                               color=s.color,
                                                               wtps=wtp_values))

        n_obs = len(self.strategies[0].costObs)

        for w in wtp_values:

            countMaximum = np.zeros(self._n)

            for obs_idx in range(n_obs):

                # find which strategy has the maximum:
                max_nmb = float('-inf')
                max_idx = 0
                for idx, s in enumerate(self.strategies):
                    d_effect = (s.effectObs[obs_idx] - self.strategies[0].effectObs[obs_idx])*self._u_or_d
                    d_cost = s.costObs[obs_idx] - self.strategies[0].costObs[obs_idx]
                    nmb = w * d_effect - d_cost
                    if nmb > max_nmb:
                        max_nmb = nmb
                        max_idx = idx

                countMaximum[max_idx] += 1

            # calculate proportion maximum
            probMaximum = countMaximum/n_obs

            for i in range(self._n):
                self.acceptabilityCurves[i].probs.append(probMaximum[i])

        # find the optimal strategy for each wtp value
        for wtp_idx, wtp in enumerate(wtp_values):

            optIdx = self.optStrategyIdx[wtp_idx]
            self.acceptabilityCurves[optIdx].optWTPs.append(wtp)
            self.acceptabilityCurves[optIdx].optProbs.append(
                self.acceptabilityCurves[optIdx].probs[wtp_idx])

    def add_incremental_NMBs_to_ax(self, ax,
                                   min_wtp, max_wtp,
                                   title, x_label, y_label, y_range=None,
                                   transparency=0.4, show_legend=False):

        for curve in self.nmbCurves:
            # plot line
            ax.plot(curve.wtps, curve.ys, c=curve.color, alpha=1, label=curve.label)
            # plot intervals
            ax.fill_between(curve.wtps, curve.ys - curve.l_errs, curve.ys + curve.u_errs,
                            color=curve.color, alpha=transparency)

        if show_legend:
            ax.legend()

        # format y-axis
        vals_y = ax.get_yticks()
        ax.set_yticks(vals_y)
        ax.set_yticklabels(['{:,.{prec}f}'.format(x, prec=0) for x in vals_y])

        # do the other formatting
        self.__format_ax(ax=ax, title=title,x_label=x_label, y_label=y_label,
                         y_range=y_range, min_wtp=min_wtp, max_wtp=max_wtp)

    def add_acceptability_curves_to_ax(self, ax,
                                       min_wtp, max_wtp,
                                       title, x_label, y_label,
                                       y_range=None, show_legend=False):

        for curve in self.acceptabilityCurves:
            # plot line
            ax.plot(curve.wtps, curve.probs, c=curve.color, alpha=1, label=curve.label)
            ax.plot(curve.optWTPs, curve.optProbs, c=curve.color, linewidth=5)
        if show_legend:
            ax.legend()

    def __format_ax(self, ax, title, x_label, y_label, y_range,
                    min_wtp, max_wtp,):

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_ylim(y_range)

        # format x-axis
        vals_x = ax.get_xticks()
        ax.set_xticks(vals_x)
        ax.set_xticklabels(['{:,.{prec}f}'.format(x, prec=0) for x in vals_x])

        d = (max_wtp - min_wtp) / NUM_WTPS_FOR_NMB_CURVES
        ax.set_xlim([min_wtp - d, max_wtp + d])

        ax.axhline(y=0, c='k', ls='--', linewidth=0.5)

    def graph_incremental_NMBs(self, min_wtp, max_wtp,
                               title, x_label, y_label, y_range=None,
                               interval_type='n', transparency=0.4,
                               show_legend=True, figure_size=(6, 6)):
        """
        plots the incremental net-monetary benefit compared to the first strategy (base)
        :param min_wtp: minimum willingness-to-pay (or cost-effectiveness threshold) on the x-axis
        :param max_wtp: maximum willingness-to-pay (or cost-effectiveness threshold) on the x-axis
        :param title: title
        :param x_label: x-axis label
        :param y_label: y-axis label
        :param y_range: (list) range of y-axis
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
        fig, ax = plt.subplots(figsize=figure_size)

        # add the incremental NMB curves
        self.add_incremental_NMBs_to_ax(ax=ax,
                                        min_wtp=min_wtp, max_wtp=max_wtp,
                                        title=title, x_label=x_label, y_label=y_label, y_range=y_range,
                                        transparency=transparency, show_legend=show_legend)

        fig.show()

    def graph_acceptability_curves(self, min_wtp, max_wtp,
                                   title=None, x_label=None, y_label=None, y_range=None,
                                   show_legend=True, figure_size=(6, 6), file_name='CEAC.png'):
        """
        plots the acceptibility curves
        :param min_wtp: minimum willingness-to-pay (or cost-effectiveness threshold) on the x-axis
        :param max_wtp: maximum willingness-to-pay (or cost-effectiveness threshold) on the x-axis
        :param title: title
        :param x_label: x-axis label
        :param y_label: y-axis label
        :param y_range: (list) range of y-axis
        :param show_legend: set true to show legend
        :param figure_size: (tuple) size of the figure (e.g. (2, 3)
        :param file_name: file name
        """

        # make the acceptability curves
        self.make_acceptability_curves(min_wtp=min_wtp,
                                       max_wtp=max_wtp)

        # initialize plot
        fig, ax = plt.subplots(figsize=figure_size)

        # add the incremental NMB curves
        self.add_acceptability_curves_to_ax(ax=ax,
                                            min_wtp=min_wtp, max_wtp=max_wtp,
                                            title=title, x_label=x_label, y_label=y_label,
                                            y_range=y_range, show_legend=show_legend)

        self.__format_ax(ax=ax, title=title, x_label=x_label, y_label=y_label,
                         y_range=y_range, min_wtp=min_wtp, max_wtp=max_wtp)

        fig.show()
        fig.savefig(file_name, bbox_inches='tight', dpi=300)

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


def get_d_cost(strategy):
    return strategy.dCost.get_mean()


def get_d_effect(strategy):
    return strategy.dEffect.get_mean()


def get_index(strategy):
    return strategy.idx
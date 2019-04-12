import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import SimPy.StatisticalClasses as Stat

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
        self.cost = None  # summary statistics for cost
        self.dCost = None  # summary statistics for cost with respect to base
        self.incCost = None  # summary statistics for incremental cost

        self.effectObs = None     # (list) effect observations
        self.dEffectObs = None    # (list) effect observations with respect to base
        self.incEffectObs = None  # (list) incremental effect observations
        self.effectObs = None  # effect observations
        self.effect = None  # summary statistics for effect
        self.dEffect = None  # summary statistics for effect with respect to base
        self.incEffect = None  # summary statistics for incremental effect

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


class CEA:
    """ master class for cost-effective analysis (CEA) and cost-benefit analysis (CBA) """

    def __init__(self, strategies, if_paired, health_measure='u'):
        """
        :param strategies: the list of strategies (assumes that the first strategy represents the "base" strategy)
        :param if_paired: set to true to indicate that the strategies are paired
        :param health_measure: (string) choose 'u' if higher "effect" implies better health
        (e.g. when QALY is used) and set to 'd' if higher "effect" implies worse health
        (e.g. when DALYS is used)
        """

        if health_measure not in ['u', 'd']:
            raise ValueError("health_measure can be either 'u' (for utility) or 'd' (for disutility).")

        self.strategies = strategies  # list of strategies
        # assign the index of each strategy
        for i, s in enumerate(strategies):
            s.idx = i

        self._n = len(strategies)   # number of strategies
        self._ifPaired = if_paired  # if cost and effect outcomes are paired across strategies
        self._healthMeasure = health_measure    # utility of disutility
        self._u_or_d = 1 if health_measure == 'u' else -1
        self._ifFrontierIsCalculated = False  # CE frontier is not calculated yet

        # shift the strategies
        self.__find_shifted_strategies()

        # find the cost-effectiveness frontier
        self.__find_frontier()

    def __find_shifted_strategies(self):
        """ find shifted strategies.
        In calculating the change in effect, it accounts for whether QALY or DALY is used.
        """

        # shift all strategies such that the base strategy (first in the list) lies on the origin
        # if observations are paired across strategies
        if self._ifPaired:

            for s in self.strategies:

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

        else:  # if not paired
            # get average cost and effect of the base strategy
            base_ave_cost = self.strategies[0].cost.get_mean()
            base_ave_effect = self.strategies[0].effect.get_mean()

            for s in self.strategies:
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

        # sort back strategies
        self.strategies.sort(key=get_index)

    def get_strategies_on_frontier(self):

        # sort strategies by effect with respect to the base
        self.strategies.sort(key=get_d_effect)

        # find strategies on the frontier
        frontier_strategies = [s for s in self.strategies if not s.ifDominated]

        # sort back
        self.strategies.sort(key=get_index)

        return frontier_strategies

    def add_ce_plane_to_ax(self, ax, include_clouds=True):

        # find the frontier (x, y)'s
        frontier_d_effect = []
        frontier_d_costs = []
        for s in self.get_strategies_on_frontier():
            frontier_d_effect.append(s.dEffect.get_mean())
            frontier_d_costs.append(s.dCost.get_mean())

        # add the frontier line
        plt.plot(frontier_d_effect, frontier_d_costs,
                 c='k',  # color
                 alpha=0.6,  # transparency
                 linewidth=2,  # line width
                 label="Frontier")  # label to show in the legend

        for s in self.strategies:
            ax.scatter(s.dEffect.get_mean(), s.dCost.get_mean(),
                       c=s.color,  # color
                       alpha=1,  # transparency
                       marker='o',  # markers
                       s=75,  # marker size
                       label=s.name  # name to show in the legend
                       )
            # if include_clouds:
            #     ax.scatter(s.dEffect.get_mean(), s.dCost.get_mean(),
            #                c='black',  # color
            #                alpha=1,  # transparency
            #                marker='x',  # markers
            #                s=75,  # marker size
            #                )

        ax.legend()
        ax.axhline(y=0, c='k', linewidth=0.5)
        ax.axvline(x=0, c='k', linewidth=0.5)

    def show_CE_plane(self, include_clouds=True):

        fig, ax = plt.subplots()

        # add the cost-effectiveness plane
        self.add_ce_plane_to_ax(ax=ax, include_clouds=include_clouds)

        fig.show()

def get_d_cost(strategy):
    return strategy.dCost.get_mean()


def get_d_effect(strategy):
    return strategy.dEffect.get_mean()


def get_index(strategy):
    return strategy.idx
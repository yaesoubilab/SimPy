import numpy as np
import SimPy.Statistics as S
from SimPy.RandomVariateGenerators import Beta


class Model:

    def __init__(self, decision_rule=None, cost_sigma=0.5, action_cost=1):

        self.costSigma = cost_sigma
        self.actionCost = action_cost

        self.decisionRule = decision_rule
        self.seqCosts = []

    def simulate(self, itr):

        self._reset()
        rng = np.random.RandomState(seed=itr)

        # find the initial state
        state = rng.random_sample()

        for t in range(3):

            # make a decision
            action = self.decisionRule.get_decision(feature_values=[t, state])

            # store the reward
            cost = self._cost(state=state, action=action, rng=rng)
            self.seqCosts.append(cost)

            # next state
            if action[0] == 0:
                result = Beta.fit_mm(mean=state, st_dev=0.01)
                if result['a'] <= 0 or result['b'] <= 0:
                    state = state
                else:
                    state = rng.beta(a=result['a'], b=result['b'])
            elif action[0] == 1:
                if state < 0.5:
                    state += (0.5 - state) * rng.random_sample()
                else:
                    state -= (state - 0.5) * rng.random_sample()
            else:
                raise ValueError

    def _cost(self, state, action, rng):

        state_cost = 1000*pow(state - 0.5, 2) + rng.normal(0, self.costSigma)
        action_cost = self.actionCost if action[0] == 1 else 0

        return state_cost + action_cost

    def set_approx_decision_maker(self, approx_decision_maker):

        self.decisionRule = Dynamic(approx_decision_maker=approx_decision_maker)

    def get_seq_of_costs(self):
        return self.seqCosts

    def _reset(self):
        self.seqCosts.clear()


class MultiModel:

    def __init__(self, decision_rule=None, cost_sigma=0.5, action_cost=1):

        self.decisionRule = decision_rule
        self.costSigma = cost_sigma
        self.actionCost = action_cost

        self.cumCosts = []
        self.statCost = None

    def simulate(self, n):

        for i in range(n):
            model = Model(decision_rule=self.decisionRule,
                          cost_sigma=self.costSigma,
                          action_cost=self.actionCost)
            model.simulate(itr=i)
            self.cumCosts.append(sum(model.seqCosts))

        self.statCost = S.SummaryStat(data=self.cumCosts)


class _DecisionRule:
    def __init__(self):
        pass

    def get_decision(self, feature_values):
        raise NotImplementedError


class AlwaysOn(_DecisionRule):
    def __init__(self):
        _DecisionRule.__init__(self)

    def get_decision(self, feature_values):
        return [1]


class AlwaysOff(_DecisionRule):
    def __init__(self):
        _DecisionRule.__init__(self)

    def get_decision(self, feature_values):
        return [0]


class Myopic(_DecisionRule):
    def __init__(self):
        _DecisionRule.__init__(self)

    def get_decision(self, feature_values):
        return [0] if 0.25 <= feature_values[1] <= 0.75 else [1]


class Dynamic(_DecisionRule):

    def __init__(self, approx_decision_maker):
        _DecisionRule.__init__(self)

        self.approxDecisionMaker = approx_decision_maker

    def get_decision(self, feature_values):

        return self.approxDecisionMaker.make_a_decision(feature_values=feature_values)


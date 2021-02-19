import numpy as np
import SimPy.Statistics as S
from SimPy.RandomVariateGenerators import Beta


class Model:

    def __init__(self, decision_rule, cost_sigma=0.5, action_cost=1):

        self.costSigma = cost_sigma
        self.actionCost = action_cost

        self.decisionRule = decision_rule
        self.seqFeatures = []
        self.seqCosts = []
        self.seqActions = []

    def simulate(self, itr):

        rng = np.random.RandomState(seed=itr)

        # find the initial state
        state = rng.random_sample()

        for t in range(3):

            # store the features
            self.seqFeatures.append([state])

            # make a decision
            action = self.decisionRule.get_decision(state=state)
            self.seqActions.append(action)

            # store the reward
            cost = self._cost(state=state, action=action, rng=rng)
            self.seqCosts.append(cost)

            # next state
            if action == 0:
                result = Beta.fit_mm(mean=state, st_dev=0.01)
                if result['a'] <= 0 or result['b'] <= 0:
                    state = state
                else:
                    state = rng.beta(a=result['a'], b=result['b'])
            elif action == 1:
                if state < 0.5:
                    state += (0.5 - state) * rng.random_sample()
                else:
                    state -= (state - 0.5) * rng.random_sample()
            else:
                pass

    def _cost(self, state, action, rng):

        state_cost = 100*pow(state - 0.5, 2) + rng.normal(0, self.costSigma)
        action_cost = self.actionCost if action == 1 else 0

        return state_cost + action_cost


class MultiModel:

    def __init__(self, decision_rule, cost_sigma=0.5, action_cost=1):

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

    def get_decision(self, state):
        pass


class AlwaysOn(_DecisionRule):
    def __init__(self):
        _DecisionRule.__init__(self)

    def get_decision(self, state):
        return 1


class AlwaysOff(_DecisionRule):
    def __init__(self):
        _DecisionRule.__init__(self)

    def get_decision(self, state):
        return 0


class Myopic(_DecisionRule):
    def __init__(self):
        _DecisionRule.__init__(self)

    def get_decision(self, state):
        return 0 if 0.25 <= state <= 0.75 else 1


class Dynamic(_DecisionRule):

    def __init__(self):
        _DecisionRule.__init__(self)

        self.qFunctions = []

    def get_decision(self, state):
        return 1

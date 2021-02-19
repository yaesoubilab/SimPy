import numpy as np
import SimPy.Statistics as S
from SimPy.RandomVariateGenerators import Beta


class Model:

    def __init__(self, decision_rule, reward_sigma=0.5, action_cost=1):

        self.reward_sigma = reward_sigma
        self.actionCost = action_cost

        self.decisionMaker = DecisionMaker(decision_rule)
        self.seqRewards = []
        self.seqActions = []

    def simulate(self, itr):

        rng = np.random.RandomState(seed=itr)

        # find the initial state
        state = rng.random_sample()

        for t in range(3):

            # make a decision
            action = self.decisionMaker.make_a_decision(state=state)
            self.seqActions.append(action)

            # store the reward
            reward = self._reward(state=state, action=action, rng=rng)
            self.seqRewards.append(reward)

            # next state
            if action == 0:
                result = Beta.fit_mm(mean=state, st_dev=0.1)
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

    def _reward(self, state, action, rng):

        state_reward = -0.5*(pow(state, 2) - state) + rng.normal(0, self.reward_sigma)
        action_cost = self.actionCost if action == 1 else 0

        return state_reward - action_cost


class MultiModel:

    def __init__(self, decision_rule, reward_sigma=0.5, action_cost=1):

        self.decisionRule = decision_rule
        self.rewardSigma = reward_sigma
        self.actionCost = action_cost

        self.cumRewards = []
        self.statRewards = None

    def simulate(self, n):

        for i in range(n):
            model = Model(decision_rule=self.decisionRule, reward_sigma=self.rewardSigma, action_cost=self.actionCost)
            model.simulate(itr=i)
            self.cumRewards.append(sum(model.seqRewards))

        self.statRewards = S.SummaryStat(data=self.cumRewards)


class DecisionMaker:

    def __init__(self, decision_rule):

        self.decisionRule = decision_rule

    def make_a_decision(self, state):

        return self.decisionRule.get_decision(state)


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

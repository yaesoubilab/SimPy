from math import pow
from SimPy.Regression import PolynomialQFunction
from SimPy.Optimization.Support import *
import numpy as np


class SimModel:
    # abstract class to be overridden by the simulation model to optimize
    def __init__(self):
        pass

    def set_approx_decision_maker(self, approx_decision_maker):
        pass

    def simulate(self, itr):
        pass

    def get_seq_of_features(self):
        pass

    def get_seq_of_costs(self):
        pass

    def get_seq_of_action_combos(self):
        pass


class ADPState:

    def __init__(self, feature_values, action_combo, cost):
        """
        :param feature_values: (list) of feature values
        :param action_combo: (list) of on/off status of actions for the next period
        :param cost: (float) cost of this period
        """

        self.featureValues = feature_values
        self.actionCombo = action_combo
        self.cost = cost
        self.costToGo = 0


class ApproxDecisionMaker:

    def __init__(self, num_of_actions, exploration_rule, q_function_degree, l2_penalty):

        self.nOfActions = num_of_actions
        self.explorationRule = exploration_rule
        self.qFunctions = []
        self.nOfActionCombos = int(pow(2, self.nOfActions))
        self.rng = np.random.RandomState(seed=0)
        self.itr = 0

        # create the q-functions
        for i in range(self.nOfActionCombos):
            self.qFunctions.append(
                PolynomialQFunction(name='Q-function for '+str(action_combo_of_a_index(i)),
                                    degree=q_function_degree,
                                    l2_penalty=l2_penalty)
            )

    def make_an_epsilon_greedy_decision(self, feature_values):

        if self.rng.random_sample() < self.explorationRule.get_epsilon(itr=self.itr):
            # explore
            i = self.rng.randint(low=0, high=self.nOfActionCombos)
            return action_combo_of_a_index(i)
        else:
            # exploit
            return self.make_a_greedy_decision(feature_values=feature_values)

    def make_a_greedy_decision(self, feature_values):

        minimum = float('inf')
        opt_action_combo = None
        for i in range(self.nOfActionCombos):

            q_value = self.qFunctions[i].f(x=feature_values)

            if q_value < minimum:
                minimum = q_value
                opt_action_combo = action_combo_of_a_index(i)

        return opt_action_combo


class ApproximatePolicyIteration:
    
    def __init__(self, sim_model, num_of_actions, learning_rule, exploration_rule, discount_factor,
                 q_function_degree, l2_penalty):
        """
        :param sim_model (SimModel) simulation model
        :param num_of_actions: (int) number of possible actions to turn on or off
        :param learning_rule:
        :param exploration_rule:
        :param discount_factor: (float) is 1 / (1 + interest rate)
        """

        self.simModel = sim_model
        self.learningRule = learning_rule
        self.discountFactor = discount_factor

        self.appoxDecisionMaker = ApproxDecisionMaker(num_of_actions=num_of_actions,
                                                      q_function_degree=q_function_degree,
                                                      exploration_rule=exploration_rule,
                                                      l2_penalty=l2_penalty)

        # assign an approximate decision maker to the model
        self.simModel.set_approx_decision_maker(
            approx_decision_maker=self.appoxDecisionMaker)

    def optimize(self, n_iterations):

        for itr in range(0, n_iterations):

            # increment the iteration count
            self.appoxDecisionMaker.itr += 1

            # simulate
            self.simModel.simulate(itr=itr)

            # get sequence of actions and features
            self._back_propagate(itr=itr,
                                 seq_of_features=self.simModel.get_seq_of_features(),
                                 seq_of_action_combos=self.simModel.get_seq_of_action_combos(),
                                 seq_of_costs=self.simModel.get_seq_of_costs())

    def _back_propagate(self, itr, seq_of_features, seq_of_action_combos, seq_of_costs):

        # make adp states
        self.adpStates = []
        for i in range(len(seq_of_features)):
            self.adpStates.append(ADPState(feature_values=seq_of_features[i],
                                           action_combo=seq_of_action_combos[i],
                                           cost=seq_of_costs[i]))

        # cost of last adp state
        self.adpStates[-1].costToGo = self.adpStates[-1].cost

        # calculate discounted cost-to-go of adp states
        for i in range(len(self.adpStates)-1, 0, -1):
            self.adpStates[i].costToGo = self.adpStates[i].cost \
                                         + self.discountFactor * self.adpStates[i+1].costToGo

        # update q-functions
        for i in range(len(self.adpStates), 0, -1):
            q_index = index_of_an_action_combo(self.adpStates[i].actionCombo)
            self.appoxDecisionMaker.qFunctions[q_index].update(
                x=self.adpStates[i].featureValues,
                f=self.adpStates[i].costToGo,
                forgetting_factor=self.learningRule.get_forgetting_factor(itr=itr))

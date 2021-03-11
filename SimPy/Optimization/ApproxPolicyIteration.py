from math import pow

import matplotlib.pyplot as plt
import numpy as np

from SimPy.InOutFunctions import write_csv, read_csv_rows
from SimPy.Optimization.Support import *
from SimPy.Regression import PolynomialQFunction
from SimPy.Support.MiscFunctions import get_moving_average


class SimModel:
    # abstract class to be overridden by the simulation model to optimize
    # a simulation model should have the following class attributes

    def __init__(self):
        pass

    def set_approx_decision_maker(self, approx_decision_maker):
        """ to allow the optimization algorithm to set a decision maker for the model that makes
            approximately optimal decisions. """
        raise NotImplementedError

    def simulate(self, itr):
        """ to allow th optimization algorithm to get one replication of the simulation model
        :param itr: (int) the iteration of the optimization algorithm
        """
        raise NotImplementedError

    def get_seq_of_costs(self):
        """ to allow tht optimization algorithm to get the sequence of cost observed
            during the decision periods of the simulation """
        raise NotImplementedError


class State:
    # a state of the approximate policy iteration algorithm
    # contains (feature values, action combination over the next period, cost over the next period)

    def __init__(self, action_combo, cost, values_of_continuous_features, values_of_indicator_features=None):
        """
        :param values_of_continuous_features: (list) of values of continuous features
        :param values_of_indicator_features: (list) of values of indicator features
        :param action_combo: (list) of on/off status of actions for the next period
        :param cost: (float) cost of this period
        """

        self.valuesOfContinuousFeatures = values_of_continuous_features
        self.valuesOfIndicatorFeatures = values_of_indicator_features
        self.actionCombo = action_combo
        self.cost = cost
        self.costToGo = 0


class _ApproxDecisionMaker:
    # super class for epsilon-greedy and greedy decisions maker

    def __init__(self, num_of_actions):
        """
        :param num_of_actions: (int) number of actions with on/off switches
        """

        self.nOfActions = num_of_actions
        self.nOfActionCombos = int(pow(2, self.nOfActions))
        self.qFunctions = []

        self.seq_of_continuous_feature_values = [] # sequence of continuous feature values throughout the simulation
        self.seq_of_indicator_feature_values = []  # sequence of indicator feature values throughout the simulation
        self.seq_of_action_combos = []  # sequence of action combinations throughout the simulation

    def reset_for_new_iteration(self):
        """ clear the sequence of feature values and action combinations """

        self.seq_of_continuous_feature_values.clear()
        self.seq_of_indicator_feature_values.clear()
        self.seq_of_action_combos.clear()

    def make_a_decision(self, continuous_feature_values=None, indicator_feature_values=None):
        raise NotImplementedError

    def _make_a_greedy_decision(self, continuous_feature_values=None, indicator_feature_values=None):
        """ makes a greedy decision given the feature values
        :param continuous_feature_values: (list) of values for continuous features
        :param indicator_feature_values: (list) of values for indicator features (can take only 0 or 1)
        :returns (list of 0s and 1s) the selected combinations of actions
        """

        if continuous_feature_values is None and indicator_feature_values is None:
            raise ValueError('Values of either continuous or indicator features should be provided.')

        minimum = float('inf')
        opt_action_combo = None
        for i in range(self.nOfActionCombos):

            # if the q-function is not updated with any data yet
            if self.qFunctions[i].get_coeffs() is None:
                q_value = 0
            else:
                q_value = self.qFunctions[i].f(values_of_continuous_features=continuous_feature_values,
                                               values_of_indicator_features=indicator_feature_values)

            if q_value < minimum:
                minimum = q_value
                opt_action_combo = action_combo_of_an_index(i)

        return opt_action_combo


class GreedyApproxDecisionMaker(_ApproxDecisionMaker):
    # class to make greedy decisions

    def __init__(self, num_of_actions, q_function_degree, q_functions_csv_file):
        """
        :param num_of_actions: (int) number of actions with on/off switches
        :param q_function_degree: (float) degree of the polynomial function used for q-functions
        :param q_functions_csv_file: (string) csv filename to read the coefficient of the q-functions
        """
        _ApproxDecisionMaker.__init__(self, num_of_actions=num_of_actions)

        # read the q-functions
        rows = read_csv_rows(file_name=q_functions_csv_file, if_ignore_first_row=False, if_convert_float=True)
        # create the q-functions
        for i, row in enumerate(rows):
            q_function = PolynomialQFunction(name='Q-function for ' + str(action_combo_of_an_index(i)),
                                             degree=q_function_degree)
            # read coefficients
            # [1:-1] is to remove left and right brackets from the list of coefficients
            q_function.set_coeffs(np.fromstring(row[1][1:-1], sep=' '))
            self.qFunctions.append(q_function)

    def make_a_decision(self, continuous_feature_values=None, indicator_feature_values=None):
        """ make a greedy decision
        :returns (list of 0s and 1s) the greedy selection of actions
        """

        a = self._make_a_greedy_decision(continuous_feature_values=continuous_feature_values,
                                         indicator_feature_values=indicator_feature_values)
        self.seq_of_continuous_feature_values.append(continuous_feature_values)
        self.seq_of_indicator_feature_values.append(indicator_feature_values)
        self.seq_of_action_combos.append(a)

        return a


class EpsilonGreedyApproxDecisionMaker(_ApproxDecisionMaker):
    # class to make epsilon-greedy decisions

    def __init__(self, num_of_actions, exploration_rule, q_function_degree, l2_penalty, q_functions_csv_file):
        """
        :param num_of_actions: (int) number of actions with on/off switches
        :param exploration_rule: exploration rule
        :param q_function_degree: (float) degree of the polynomial function used for q-functions
        :param l2_penalty: (float) l2 regularization penalty
        :param q_functions_csv_file: (string) csv filename to store the coefficient of the q-functions
        """

        _ApproxDecisionMaker.__init__(self, num_of_actions=num_of_actions)

        self.explorationRule = exploration_rule
        self.rng = np.random.RandomState(seed=0)
        self.itr = 0   # iteration of the algorithm (needed to calculate exploration rate)
        self.qFunctionsCSVFile = q_functions_csv_file

        # create the q-functions
        for i in range(self.nOfActionCombos):
            self.qFunctions.append(
                PolynomialQFunction(name='Q-function for '+str(action_combo_of_an_index(i)),
                                    degree=q_function_degree,
                                    l2_penalty=l2_penalty)
            )

    def make_a_decision(self, continuous_feature_values=None, indicator_feature_values=None):
        """ makes an epsilon-greedy decision given the feature values
        :param continuous_feature_values: (list) of values for continuous features
        :param indicator_feature_values: (list) of values for indicator features (can take only 0 or 1)
        """

        if self.rng.random_sample() < self.explorationRule.get_epsilon(itr=self.itr):
            # explore
            i = self.rng.randint(low=0, high=self.nOfActionCombos)
            a = action_combo_of_an_index(i)
        else:
            # exploit
            a = self._make_a_greedy_decision(continuous_feature_values=continuous_feature_values,
                                             indicator_feature_values=indicator_feature_values)

        self.seq_of_continuous_feature_values.append(continuous_feature_values)
        self.seq_of_indicator_feature_values.append(indicator_feature_values)
        self.seq_of_action_combos.append(a)

        return a

    def export_q_functions(self):
        """ exports the coefficients of q-functions to a csv file. """

        rows = []
        for q in self.qFunctions:
            rows.append([q.name, q.get_coeffs()])

        write_csv(rows=rows, file_name=self.qFunctionsCSVFile)


class ApproximatePolicyIteration:
    
    def __init__(self, sim_model, num_of_actions, learning_rule, exploration_rule, discount_factor,
                 q_function_degree, l2_penalty, q_functions_csv_file='q-functions.csv'):
        """
        :param sim_model (SimModel) simulation model
        :param num_of_actions: (int) number of possible actions to turn on or off
        :param learning_rule: learning rule
        :param exploration_rule: exploration rule
        :param discount_factor: (float) is 1 / (1 + interest rate)
        :param q_function_degree: (int) degree of the polynomial function used for q-functions
        :param l2_penalty: (float) l2 regularization penalty
        :param q_functions_csv_file: (string) csv filename to store the coefficient of the q-functions
        """

        assert hasattr(sim_model, 'set_approx_decision_maker'), \
            'sim_model should implement the attribute set_approx_decision_maker.'
        assert hasattr(sim_model, 'simulate'), \
            'sim_model should implement the attribute simulate.'
        assert hasattr(sim_model, 'get_seq_of_costs'), \
            'sim_model should implement the attribute get_seq_of_costs.'

        self.simModel = sim_model
        self.learningRule = learning_rule
        self.discountFactor = discount_factor

        self.itr_i = []  # iteration indices
        self.itr_total_cost = []  # discounted total cost over iterations
        self.itr_error = []  # errors over iterations
        self.itr_forgetting_factor = []  # forgetting factor over iterations
        self.itr_exploration_rate = []  # exploration rate over iterations

        # create the approximate decision maker
        self.appoxDecisionMaker = EpsilonGreedyApproxDecisionMaker(num_of_actions=num_of_actions,
                                                                   q_function_degree=q_function_degree,
                                                                   exploration_rule=exploration_rule,
                                                                   l2_penalty=l2_penalty,
                                                                   q_functions_csv_file=q_functions_csv_file)

        # assign the approximate decision maker to the model
        self.simModel.set_approx_decision_maker(
            approx_decision_maker=self.appoxDecisionMaker)

    def optimize(self, n_iterations):
        """
        :param n_iterations: (int) number of iterations
        """

        for itr in range(1, n_iterations + 1):

            # store iteration, exploration rate, and forgetting factor
            self.itr_i.append(itr)
            self.itr_forgetting_factor.append(self.learningRule.get_forgetting_factor(itr=itr))
            self.itr_exploration_rate.append(self.appoxDecisionMaker.explorationRule.get_epsilon(itr=itr))

            # update the iteration of the approximate decision maker (to calculate exploration rate)
            self.appoxDecisionMaker.itr = itr
            self.appoxDecisionMaker.reset_for_new_iteration()

            # simulate
            self.simModel.simulate(itr=itr)

            # do back-propagation
            self._back_propagate(
                itr=itr,
                seq_of_continuous_feature_values=self.appoxDecisionMaker.seq_of_continuous_feature_values,
                seq_of_indicator_feature_values=self.appoxDecisionMaker.seq_of_indicator_feature_values,
                seq_of_action_combos=self.appoxDecisionMaker.seq_of_action_combos,
                seq_of_costs=self.simModel.get_seq_of_costs())

        # export q-functions:
        self.appoxDecisionMaker.export_q_functions()

    def _back_propagate(self, itr,
                        seq_of_continuous_feature_values,
                        seq_of_indicator_feature_values,
                        seq_of_action_combos,
                        seq_of_costs):

        if not (len(seq_of_continuous_feature_values)
                == len(seq_of_indicator_feature_values)
                == len(seq_of_action_combos)
                == len(seq_of_costs)):
            raise ValueError('For iteration {}, the number of past feature values, '
                             'action combinations, and costs are not equal.')
        
        # if no decisions are made, back-propagation cannot be performed
        if len(seq_of_continuous_feature_values) == 0:
            self.itr_total_cost.append(None)
            self.itr_error.append(None)
            return

        # make feature/action states
        self.states = []
        for i in range(len(seq_of_continuous_feature_values)):
            self.states.append(State(values_of_continuous_features=seq_of_continuous_feature_values[i],
                                     values_of_indicator_features=seq_of_indicator_feature_values[i],
                                     action_combo=seq_of_action_combos[i],
                                     cost=seq_of_costs[i]))

        # cost of last state in this simulation run
        self.states[-1].costToGo = self.states[-1].cost

        # calculate discounted cost-to-go of states
        i = len(self.states) - 1 - 1
        while i >= 0:
            self.states[i].costToGo = self.states[i].cost \
                                      + self.discountFactor * self.states[i + 1].costToGo
            i -= 1
        
        # store the discounted total cost
        self.itr_total_cost.append(self.states[0].costToGo)

        # store error of the first period
        q_index = index_of_an_action_combo(self.states[0].actionCombo)
        if self.appoxDecisionMaker.qFunctions[q_index].get_coeffs() is None:
            self.itr_error.append(None)
        else:
            predicted_cost = self.appoxDecisionMaker.qFunctions[q_index].f(
                values_of_continuous_features=self.states[0].valuesOfContinuousFeatures,
                values_of_indicator_features=self.states[0].valuesOfIndicatorFeatures)
            self.itr_error.append(self.states[0].costToGo - predicted_cost)

        # update q-functions
        forgetting_factor = self.learningRule.get_forgetting_factor(itr=itr)

        for s in self.states:
            q_index = index_of_an_action_combo(s.actionCombo)
            self.appoxDecisionMaker.qFunctions[q_index].update(
                values_of_continuous_features=s.valuesOfContinuousFeatures,
                values_of_indicator_features=s.valuesOfIndicatorFeatures,
                f=s.costToGo,
                forgetting_factor=forgetting_factor)

    def plot_cost_itr(self, moving_ave_window=None,
                      y_range=None,
                      y_label='Discounted total cost',
                      x_label='Iteration', fig_size=(6, 5)):
        """
        :return: a plot of cost as the algorithm iterates
        """

        fig, ax = plt.subplots(figsize=fig_size)

        self.add_cost_itr(ax=ax, moving_ave_window=moving_ave_window,
                          y_range=y_range, y_label=y_label)

        ax.set_xlabel(x_label)

        plt.show()

    def add_cost_itr(self, ax, moving_ave_window=None,
                     y_range=None, y_label=None):
        """
        :return: a plot of cost as the algorithm iterates
        """

        # discounted cost
        ax.plot(self.itr_i, self.itr_total_cost)
        # moving average of the objective function
        if moving_ave_window is not None:
            ax.plot(self.itr_i, get_moving_average(self.itr_total_cost, window=moving_ave_window),
                    'k-', markersize=1)

        if y_label is None:
            y_label = 'Discounted\ntotal cost'

        ax.set_ylim(y_range)
        ax.set_ylabel(y_label)

    def add_error_itr(self, ax, moving_ave_window=None,
                      y_range=None, y_label=None):
        """
        :return: a plot of error of first period as the algorithm iterates
        """

        # discounted cost
        ax.plot(self.itr_i, self.itr_error)
        # moving average of the objective function
        if moving_ave_window is not None:
            ax.plot(self.itr_i, get_moving_average(self.itr_error, window=moving_ave_window),
                    'k-', markersize=1)

        ax.axhline(y=0, linestyle='--', color='black', linewidth=1)

        if y_label is None:
            y_label = 'Error of\nfirst period'

        ax.set_ylim(y_range)
        ax.set_ylabel(y_label)

    def add_forgetting_factor_itr(self, ax, y_range=None, y_label=None):

        ax.plot(self.itr_i, self.itr_forgetting_factor)

        if y_label is None:
            y_label = 'Forgetting factor'
        if y_range is None:
            y_range = (0, 1)

        ax.set_ylim(y_range)
        ax.set_ylabel(y_label)

    def add_exploration_rate_itr(self, ax, y_range=None, y_label=None):

        ax.plot(self.itr_i, self.itr_exploration_rate)

        if y_label is None:
            y_label = 'Exploration'
        if y_range is None:
            y_range = (0, 1)

        ax.set_ylim(y_range)
        ax.set_ylabel(y_label)

    def plot_itr(self, moving_ave_window=None, y_ranges=None, y_labels=None, fig_size=(6, 6)):

        if y_ranges is None:
            y_ranges = [None]*4
        if y_labels is None:
            y_labels = [None]*4

        f, axarr = plt.subplots(4, 1, figsize=fig_size, sharex=True)

        # cost
        self.add_cost_itr(ax=axarr[0], moving_ave_window=moving_ave_window,
                          y_range=y_ranges[0], y_label=y_labels[0])

        # error
        self.add_error_itr(ax=axarr[1], moving_ave_window=None,
                           y_range=y_ranges[1], y_label=y_labels[1])

        # forgetting factor
        self.add_forgetting_factor_itr(ax=axarr[2], y_range=y_ranges[2], y_label=y_labels[2])

        # exploration rate
        self.add_exploration_rate_itr(ax=axarr[3], y_range=y_ranges[3], y_label=y_labels[3])

        f.tight_layout()
        f.align_ylabels()

        plt.show()

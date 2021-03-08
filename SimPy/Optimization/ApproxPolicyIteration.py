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


class _ApproxDecisionMaker:
    # super class for epsilon-greedy and greedy decisions maker

    def __init__(self, num_of_actions):
        """
        :param num_of_actions: (int) number of actions with on/off switches
        """

        self.nOfActions = num_of_actions
        self.nOfActionCombos = int(pow(2, self.nOfActions))
        self.qFunctions = []

        self.seq_of_feature_values = [] # sequence of feature values throughout the simulation
        self.seq_of_action_combos = []  # sequence of action combinations throughout the simulation

    def reset_for_new_iteration(self):
        """ clear the sequence of feature values and action combinations """

        self.seq_of_feature_values.clear()
        self.seq_of_action_combos.clear()

    def make_a_decision(self, feature_values):
        raise NotImplementedError

    def _make_a_greedy_decision(self, feature_values):
        """ makes a greedy decision given the feature values
        :param feature_values: (list) of feature values
        :returns (list of 0s and 1s) the selected combinations of actions
        """

        minimum = float('inf')
        opt_action_combo = None
        for i in range(self.nOfActionCombos):

            # if the q-function is not updated with any data yet
            if self.qFunctions[i].get_coeffs() is None:
                q_value = 0
            else:
                q_value = self.qFunctions[i].f(x=feature_values)

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
            q_function.set_coeffs(np.fromstring(row[1][1:-1], sep=' '))
            self.qFunctions.append(q_function)

    def make_a_decision(self, feature_values):
        """ make a greedy decision
        :returns (list of 0s and 1s) the greedy selection of actions
        """

        a = self._make_a_greedy_decision(feature_values=feature_values)
        self.seq_of_feature_values.append(feature_values)
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

    def make_a_decision(self, feature_values):
        """ makes an epsilon-greedy decision given the feature values
        :param feature_values: (list) of feature values
        """

        if self.rng.random_sample() < self.explorationRule.get_epsilon(itr=self.itr):
            # explore
            i = self.rng.randint(low=0, high=self.nOfActionCombos)
            a = action_combo_of_an_index(i)
        else:
            # exploit
            a = self._make_a_greedy_decision(feature_values=feature_values)

        self.seq_of_feature_values.append(feature_values)
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
            self._back_propagate(itr=itr,
                                 seq_of_features=self.appoxDecisionMaker.seq_of_feature_values,
                                 seq_of_action_combos=self.appoxDecisionMaker.seq_of_action_combos,
                                 seq_of_costs=self.simModel.get_seq_of_costs())

        # export q-functions:
        self.appoxDecisionMaker.export_q_functions()

    def _back_propagate(self, itr, seq_of_features, seq_of_action_combos, seq_of_costs):

        # make feature/action states
        self.states = []
        for i in range(len(seq_of_features)):
            self.states.append(State(feature_values=seq_of_features[i],
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
            predicted_cost = self.appoxDecisionMaker.qFunctions[q_index].f(x=self.states[0].featureValues)
            self.itr_error.append(self.states[0].costToGo - predicted_cost)

        # update q-functions
        forgetting_factor = self.learningRule.get_forgetting_factor(itr=itr)

        i = len(self.states) - 1
        while i >= 0:
            print(itr, i)
            q_index = index_of_an_action_combo(self.states[i].actionCombo)
            self.appoxDecisionMaker.qFunctions[q_index].update(
                x=self.states[i].featureValues,
                f=self.states[i].costToGo,
                forgetting_factor=forgetting_factor)
            i -= 1

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

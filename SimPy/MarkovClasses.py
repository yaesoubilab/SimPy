import numpy as np

import SimPy.RandomVariantGenerators as RVG


class MarkovJumpProcess:

    def __init__(self, transition_prob_matrix):
        """
        :param transition_prob_matrix: (list) transition probability matrix of a discrete-time Markov model
        """

        assert type(transition_prob_matrix) is list, \
            'Transition probability matrix should be a list'

        self._empiricalDists = []

        for i, probs in enumerate(transition_prob_matrix):

            # check if the sum of probabilities in this row is 1.
            s = sum(probs)
            if s < 0.99999 or s > 1.00001:
                raise ValueError('Sum of each row in a probability matrix should be 1. '
                                 'Sum of row {0} is {1}.'.format(i, s))

            # create an empirical distribution over the future states from this state
            self._empiricalDists.append(RVG.Empirical(probabilities=probs))

    def get_next_state(self, current_state_index, rng):
        """
        :param current_state_index: index of the current state
        :param rng: random number generator object
        :return: the index of the next state
        """

        if not (0 <= current_state_index < len(self._empiricalDists)):
            raise ValueError('The value of the current state index should be greater '
                             'than 0 and smaller than the number of states.'
                             'Value provided for current state index is {}.'.format(current_state_index))

        # find the next state index by drawing a sample from
        # the empirical distribution associated with this state
        return self._empiricalDists[current_state_index].sample(rng=rng)


class Gillespie:
    def __init__(self, transition_rate_matrix):
        """
        :param transition_rate_matrix: transition rate matrix of the continuous-time Markov model
        """

        self._rateMatrix = transition_rate_matrix
        self._expDists = []
        self._empiricalDists = []

        for i, row in enumerate(transition_rate_matrix):
            # find sum of rates out of this state
            rate_out = out_rate(row, i)
            # if the rate is 0, put None as the exponential and empirical distributions
            if rate_out > 0:
                # create an exponential distribution with rate equal to sum of rates out of this state
                self._expDists.append(RVG.Exponential(scale=1/rate_out))
                # find the transition rates to other states
                # assume that the rate into this state is 0
                rates = []
                for j, v in enumerate(row):
                    if i == j:
                        rates.append(0)
                    else:
                        rates.append(v)

                # calculate the probability of each event (prob_j = rate_j / (sum over j of rate_j)
                probs = np.array(rates) / rate_out
                # create an empirical distribution over the future states from this state
                self._empiricalDists.append(RVG.Empirical(probs))

            else:  # if the sum of rates out of this state is 0
                self._expDists.append(None)
                self._empiricalDists.append(None)

    def get_next_state(self, current_state_index, rng):
        """
        :param current_state_index: index of the current state
        :param rng: random number generator object
        :return: (dt, i) where dt is the time until next event, and i is the index of the next state
         it returns None for dt if the process is in an absorbing state
        """

        if not (0 <= current_state_index < len(self._rateMatrix)):
            raise ValueError('The value of the current state index should be greater '
                             'than 0 and smaller than the number of states.')

        # if this is an absorbing state (i.e. sum of rates out of this state is 0)
        if self._expDists[current_state_index] is None:
            # the process stays in the current state
            dt = None
            i = current_state_index
        else:
            # find the time until next event
            dt = self._expDists[current_state_index].sample(rng=rng)
            # find the next state
            i = self._empiricalDists[current_state_index].sample(rng=rng)

        return dt, i


def continuous_to_discrete(trans_rate_matrix, delta_t):
    """
    :param trans_rate_matrix: (list of lists) transition rate matrix (assumes None or 0 for diagonal elements)
    :param delta_t: cycle length
    :return: transition probability matrix (list of lists)
             and the upper bound for the probability of two transitions within delta_t (float)
        converting [p_ij] to [lambda_ij] where
            mu_i = sum of rates out of state i
            p_ij = exp(-mu_i*delta_t),      if i = j,
            p_ij = (1-exp(-mu_i*delta_t))*lambda_ij/mu_i,      if i != j.

    """

    # list of rates out of each row
    rates_out = []
    for row in trans_rate_matrix:
        rates_out.append(out_rate(row))

    prob_matrix = []
    for i in range(len(trans_rate_matrix)):
        prob_row = []   # list of probabilities
        # calculate probabilities
        for j in range(len(trans_rate_matrix[i])):
            prob = 0
            if i == j:
                prob = np.exp(-rates_out[i] * delta_t)
            else:
                if rates_out[i] > 0:
                    prob = (1 - np.exp(-rates_out[i] * delta_t)) * trans_rate_matrix[i][j] / rates_out[i]
            # append this probability
            prob_row.append(prob)

        # append this row of probabilities
        prob_matrix.append(prob_row)

    # probability that transition occurs within delta_t for each state
    probs_out = []
    for rate in rates_out:
        probs_out.append(1-np.exp(-delta_t*rate))

    # calculate the probability of two transitions within delta_t for each state
    prob_out_out = []
    for i in range(len(trans_rate_matrix)):

        # probability of leaving state i withing delta_t
        prob_out_i = probs_out[i]

        # probability of leaving the new state after i withing delta_t
        prob_out_again = 0

        for j in range(len(trans_rate_matrix[i])):
            if not i == j:

                # probability of transition from i to j
                prob_i_j = 0
                if rates_out[i]>0:
                    prob_i_j = trans_rate_matrix[i][j] / rates_out[i]
                # probability of transition from i to j and then out of j within delta_t
                prob_i_j_out = prob_i_j * probs_out[j]
                # update the probability of transition out of i and again leaving the new state
                prob_out_again += prob_i_j_out

        # store the probability of leaving state i to a new state and leaving the new state withing delta_t
        prob_out_out.append(prob_out_i*prob_out_again)

    # return the probability matrix and the upper bound for the probability of two transitions with delta_t
    return prob_matrix, max(prob_out_out)


def out_rate(rates, idx):
    """
    :param rates: list of rates leaving this state
    :param inx: index of this state
    :returns the rate of leaving this sate (the sum of rates)
    """

    sum_rates = 0
    for i, v in enumerate(rates):
        if i != idx and v is not None:
            sum_rates += v
    return sum_rates


def discrete_to_continuous(trans_prob_matrix, delta_t):
    """
    :param trans_prob_matrix: (list of lists) transition probability matrix
    :param delta_t: cycle length
    :return: (list of lists) transition rate matrix
        Converting [p_ij] to [lambda_ij] where
            lambda_ii = None, and
            lambda_ij = -ln(p_ii) * p_ij / ((1-p_ii)*Delta_t)
    """

    assert type(trans_prob_matrix) == list, \
        "prob_matrix is a matrix that should be represented as a list of lists: " \
        "For example: [ [0.1, 0.9], [0.8, 0.2] ]."

    rate_matrix = []
    for i, row in enumerate(trans_prob_matrix):
        rate_row = []   # list of rates
        # calculate rates
        for j in range(len(row)):
            # rate is None for diagonal elements
            if i == j:
                rate = None
            else:
                # rate is zero if this is an absorbing state
                if trans_prob_matrix[i][i] == 1:
                    rate = 0
                else:
                    rate = -np.log(trans_prob_matrix[i][i]) * trans_prob_matrix[i][j] / ((1 - trans_prob_matrix[i][i]) * delta_t)
            # append this rate
            rate_row.append(rate) 
        # append this row of rates
        rate_matrix.append(rate_row)

    return rate_matrix

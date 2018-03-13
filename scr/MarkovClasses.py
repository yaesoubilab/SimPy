import numpy as np


def continuous_to_discrete(rate_matrix, delta_t):
    """
    :param rate_matrix: (list of lists) transition rate matrix
    :param delta_t: cycle length
    :return: transition probability matrix (list of lists)
             and the upper bound for the probability of wo transitions within delta_t (float)
    """

    # list of rates out of each row
    rates_out = []
    for row in rate_matrix:
        rates_out.append(out_rate(row))

    prob_matrix = []
    for i in range(len(rate_matrix)):
        prob_row = []   # list of probabilities
        # calculate probabilities
        for j in range(len(rate_matrix[i])):
            prob = 0
            if i == j:
                prob = np.exp(-rates_out[i] * delta_t)
            else:
                if rates_out[i] > 0:
                    prob = (1 - np.exp(-rates_out[i] * delta_t)) * rate_matrix[i][j] / rates_out[i]
            # append this probability
            prob_row.append(prob)

        # append this row of probabilities
        prob_matrix.append(prob_row)

    # probability that transition occurs within delta_t for each state
    probs_out=[]
    for rate in rates_out:
        probs_out.append(1-np.exp(-delta_t*rate))

    # calculate the probability of two transitions within delta_t for each state
    prob_out_out = []
    for i in range(len(rate_matrix)):

        # probability of leaving state i withing delta_t
        prob_out_i = probs_out[i]

        # probability of leaving the new state after i withing delta_t
        prob_out_again = 0

        for j in range(len(rate_matrix[i])):
            if not i == j:

                # probability of transition from i to j
                prob_i_j = 0
                if rates_out[i]>0:
                    prob_i_j = rate_matrix[i][j]/rates_out[i]
                # probability of transition from i to j and then out of j within delta_t
                prob_i_j_out = prob_i_j * probs_out[j]
                # update the probability of transition out of i and again leaving the new state
                prob_out_again += prob_i_j_out

        # store the probability of leaving state i to a new state and leaving the new state withing delta_t
        prob_out_out.append(prob_out_i*prob_out_again)

    # return the probability matrix and the upper bound for the probability of two transitions with delta_t
    return prob_matrix, max(prob_out_out)


def out_rate(rates):
    """
    :param rates: list of rates leaving this state
    :returns the rate of leaving this sate (the sum of rates)
    """

    sum_rates = 0
    for v in rates:
        if not (v is None):
            sum_rates += v
    return sum_rates


def discrete_to_continuous(prob_matrix, delta_t):
    """
    :param prob_matrix: (list of lists) transition probability matrix
    :param delta_t: cycle length
    :return: (list of lists) transition rate matrix
    """

    rate_matrix = []
    for i in range(len(prob_matrix)):
        rate_row = []   # list of rates
        # calculate rates
        for j in range(len(prob_matrix[i])):
            rate = None
            if not i == j:
                if prob_matrix[i][i] == 1:
                    rate = 0
                else:
                    rate = -np.log(prob_matrix[i][i]) * prob_matrix[i][j] / ((1 - prob_matrix[i][i]) * delta_t)
            # append this rate
            rate_row.append(rate) 
        # append this row of rates
        rate_matrix.append(rate_row)

    return rate_matrix

import numpy as np


def continuous_to_discrete(rate_matrix, delta_t=None):
    """
    :param rate_matrix: (list of lists) transition rate matrix
    :param delta_t: cycle length
    :return: (list of lists) transition probability matrix
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

    return prob_matrix


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

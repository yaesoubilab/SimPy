import random

import numpy as np


def proper_file_name(text):
    """
    :param text: filename
    :return: filename where invalid characters are removed
    """

    return text.replace('|', ',').replace('<', 'l').replace('>', 'g').replace('\n', '')


def get_moving_average(data, window=2):
    """
    calculates the moving average of a time-series
    :param data: list of observations
    :param window: the window (number of data points) over which the average should be calculated
    :return: list of moving averages
    """

    if window < 2:
        raise ValueError('The window over which the moving averages '
                         'should be calculated should be greater than 1.')
    if window >= len(data):
        raise ValueError('The window over which the moving averages '
                         'should be calculated should be less than the number of data points.')

    window = int(window)

    averages = []

    # the 'window - 1' averages cannot be calculated
    for i in range(window-1):
        averages.append(None)

    # calculate the first moving average
    total = 0
    n = 0
    for i in range(window):
        if data[i] is not None:
            total += data[i]
            n += 1
    moving_ave = total/n
    averages.append(moving_ave)

    for i in range(window, len(data)):
        if moving_ave is None:
            moving_sum = 0
        else:
            moving_sum = moving_ave * n
        if data[i-window] is not None:
            moving_sum -= data[i-window]
            n -= 1
        if data[i] is not None:
            moving_sum += data[i]
            n += 1

        if n > 0:
            moving_ave = moving_sum/n
        else:
            moving_ave = None
        averages.append(moving_ave)

    return averages


def effective_sample_size(likelihood_weights):
    """
    :param likelihood_weights: (list) probabilities
    :returns: the effective sample size
    """

    # convert to numpy array if needed
    if not type(likelihood_weights) == np.ndarray:
        likelihood_weights = np.array(likelihood_weights)

    # normalize the weights if needed
    s = sum(likelihood_weights)
    if s < 0.99999 or s > 1.00001:
        likelihood_weights = likelihood_weights / s

    return 1 / np.sum(likelihood_weights ** 2)


def get_random_colors(n, seed=0):
    """
    :param n: (int) number of colors randomly selected 
    :param seed: (int) seed of the random number generator 
    :return: (list) of color codes 
    """
    
    random.seed(seed)

    colors = []
    for j in range(n):
        rand_colors = "#" + ''.join([random.choice('ABCDEF0123456789') for i in range(6)])
        colors.append(rand_colors)
    
    return colors


def convert_lnl_to_prob(ln_likelihoods):

    for i, lnl in enumerate(ln_likelihoods):
        if np.isnan(lnl):
            ln_likelihoods[i] = -np.inf

    # find the maximum lnl
    max_lnl = max(ln_likelihoods)

    # find probability weights
    probs = [np.exp(s - max_lnl) for s in ln_likelihoods]
    sum_prob = sum(probs)

    # normalize the probability weights
    return np.array(probs)/sum_prob


def get_percentile_of_empirical_dist(xs, probs, q):
    """
    :param xs: (list or np.array) values that the random variable can take
        (assumed to be sorted in the increasing order)
    :param probs: (list or np.array) probability of each value
    :param q: (float) percentile value (has to be between 0 and 1)
    :return: the value of x where q*100% of cumulative probability distribution is below q
    """

    raise ValueError('needs to be debugged.')

    if not 0 <= q <= 1:
        raise ValueError('q should be between 0 and 1.')

    # remove x values with 0 probability
    xs_nonzero_prob = []
    nonzero_probs = []
    for i, p in enumerate(probs):
        if p > 0:
            xs_nonzero_prob.append(xs[i])
            nonzero_probs.append(p)

    sum_p = 0

    if q <= 0.5:
        i = 0
        while True:
            sum_p += nonzero_probs[i]
            if sum_p >= q:
                if i - 1 < 0:
                    return None
                else:
                    return xs_nonzero_prob[i - 1]
            i += 1

    else:
        i = len(xs_nonzero_prob) - 1
        while True:
            if q == 1:
                return xs_nonzero_prob[-1]
            else:
                sum_p += nonzero_probs[i]
                if sum_p >= 1 - q:
                    return xs_nonzero_prob[i - 1]
                i -= 1

#
# print(get_percentile_of_empirical_dist(xs=[1, 2, 3, 4], probs=[0.1, 0.2, 0.7, 0], q=0.05))
# print(get_percentile_of_empirical_dist(xs=[1, 2, 3, 4], probs=[0.1, 0.2, 0.7, 0], q=0.4))
# print(get_percentile_of_empirical_dist(xs=[1, 2, 3, 4], probs=[0.1, 0.2, 0.7, 0], q=1))
# print(get_percentile_of_empirical_dist(xs=[1, 2, 3, 4], probs=[0.1, 0.2, 0.7, 0], q=0.6))

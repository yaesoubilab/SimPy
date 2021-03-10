import numpy as np


def proper_file_name(text):
    """
    :param text: filename
    :return: filename where invalid characters are removed
    """

    return text.replace('|', ',').replace(':', ',').replace('<', 'l').replace('>', 'g').replace('\n', '')


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
        moving_sum = moving_ave * n
        if data[i-window] is not None:
            moving_sum -= data[i-window]
            n -= 1
        if data[i] is not None:
            moving_sum += data[i]
            n += 1

        moving_ave = moving_sum/n
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

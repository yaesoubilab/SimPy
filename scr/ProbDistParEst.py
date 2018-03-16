import math as math


def get_gamma_parameters(mean, st_dev):
    """
    :param mean: sample mean of an observation set
    :param st_dev: sample standard deviation of an observation set
    :return: shape and scale of the gamma distribution with mean and standard deviation matching the
    provided sample mean and sample st_dev
    """

    shape = (mean / st_dev) ** 2
    scale = (st_dev ** 2) / mean
    
    return shape, scale


def get_beta_parameters(mean, st_dev):
    """
    :param mean: sample mean of an observation set
    :param st_dev: sample standard deviation of an observation set
    :return: alpha and beta of the gamma distribution with mean and standard deviation matching the
    provided sample mean and sample st_dev
    """

    aPlusB = mean*(1-mean)/st_dev**2 - 1
    a = mean*aPlusB

    return a, aPlusB - a


def get_log_normal_parameters(mean, st_dev):
    """
    :param mean: sample mean of an observation set
    :param st_dev: sample standard deviation of an observation set
    :return: mu and sigma the lognormal distribution with mean and standard deviation matching the
    provided sample mean and sample st_dev
    """

    mu = math.log(
        mean**2 / math.sqrt(st_dev**2 + mean**2)
    )
    sigma = math.sqrt(
        math.log(1 + st_dev**2 / mean**2)
    )

    return mu, sigma

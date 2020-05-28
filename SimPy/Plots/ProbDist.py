import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat
import SimPy.Plots.FigSupport as Fig
import SimPy.RandomVariantGenerators as RVGs


COLOR_CONTINUOUS_FIT = 'r'
COLOR_DISCRETE_FIT = 'r'
MIN_PROP = 0.001
MAX_PROB = 0.999
MAX_PROB_DISCRETE = 0.999999


def find_bins(data, bin_width):
    """
    :param data: (list) of observations
    :param bin_width: desired bin width
    :return: 'auto' if bin_width is not provided; otherwise, it calculates the binds
    """
    if bin_width is None:
        bins = 'auto'
    else:
        bins = np.arange(min(data), max(data) + bin_width, bin_width)
    return bins


def add_hist(ax, data, bin_width):
    """ add the histogram of the provided data to the axis"""
    ax.hist(data, density=True, bins=find_bins(data, bin_width),
            edgecolor='black', alpha=0.5, label='Data')


def add_continuous_dist(ax, dist, label):
    """ add the distribution of the provided continuous probability distribution to the axis
    :param ax: figure axis
    :param dist: probability distribution
    :param label: label of the fitted probability distribution to used in the legend
    """

    x_values = np.linspace(dist.ppf(MIN_PROP),
                           dist.ppf(MAX_PROB), 200)
    ax.plot(x_values, dist.pdf(x_values),
            color=COLOR_CONTINUOUS_FIT, lw=2, label=label)


def add_discrete_dist(ax, dist, label):
    """ add the distribution of the provided discrete probability distribution to the axis
    :param ax: figure axis
    :param dist: probability distribution
    :param label: label of the fitted probability distribution to used in the legend
    """

    x_values = range(int(dist.ppf(MIN_PROP)), int(dist.ppf(MAX_PROB))+1, 1)

    # probability mass function needs to be shifted to match the histogram
    pmf = dist.pmf(x_values)
    pmf = np.append([0], pmf[:-1])

    ax.step(x_values, pmf, color=COLOR_DISCRETE_FIT, lw=2, label=label)


def format_fig(ax, title, x_label, x_range, y_range):
    """ adds title and x_label """

    ax.set_title(title)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0%}'))
    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")


def finish_figure(ax, data, bin_width, title, x_label, x_range, y_range, filename):

    if data is not None:
        add_hist(ax, data, bin_width)

    # format axis
    format_fig(ax=ax, title=title, x_label=x_label, x_range=x_range, y_range=y_range)
    ax.legend()

    Fig.output_figure(plt=plt, filename=filename, dpi=300)


def plot_beta_fit(data, fit_results, title=None, x_label=None, x_range=None, y_range=None,
                  fig_size=(6, 5), bin_width=None, filename=None):
    """
    :param data: (numpy.array) observations
    :param fit_results: dictionary with keys "a", "b", "loc", "scale"
    :param title: title of the figure
    :param x_label: label to show on the x-axis of the histogram
    :param x_range: (tuple) x range
    :param y_range: (tuple) y range
    :param fig_size: int, specify the figure size
    :param bin_width: bin width
    :param filename: filename to save the figure as
    """

    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    # build the beta distribution
    dist = stat.beta(fit_results['a'], fit_results['b'], fit_results['loc'], fit_results['scale'])

    # plot the distribution
    add_continuous_dist(ax, dist, label='Beta')

    finish_figure(ax=ax, data=data, bin_width=bin_width,
                  title=title, x_label=x_label, x_range=x_range, y_range=y_range,
                  filename=filename)


def plot_beta_binomial_fit(data, fit_results, title=None, x_label=None, x_range=None, y_range=None,
                           fig_size=(6, 5), bin_width=1, filename=None):
    """
    :param data: (numpy.array) observations
    :param fit_results: dictionary with keys "a", "b", "n", and "loc"
    :param title: title of the figure
    :param x_label: label to show on the x-axis of the histogram
    :param x_range: (tuple) x range
    :param y_range: (tuple) y range
    :param fig_size: int, specify the figure size
    :param bin_width: bin width
    :param filename: filename to save the figure as
    """

    # plot histogram
    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    # build the beta binomial distribution
    dist = stat.betabinom(n=fit_results['n'], a=fit_results['a'], b=fit_results['b'], loc=fit_results['loc'])

    # plot the estimated distribution
    add_discrete_dist(ax, dist, label='Beta-Binomial')

    finish_figure(ax=ax, data=data, bin_width=bin_width,
                  title=title, x_label=x_label, x_range=x_range, y_range=y_range,
                  filename=filename)


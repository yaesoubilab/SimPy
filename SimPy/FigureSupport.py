import matplotlib.pyplot as plt
import numpy as np


def output_figure(plt, output_type='show', filename='figure'):
    """
    :param plt: reference to the plot
    :param output_type: select from 'show', 'png', or 'pdf'
    :param filename: filename to save this figure as
    """
    # output
    if output_type == 'show':
        plt.show()
    elif output_type == 'png':
        plt.savefig(filename + ".png")
    elif output_type == 'pdf':
        plt.savefig(filename + ".pdf")
    else:
        raise ValueError("Invalid value for figure output type. "
                         "Valid values are: 'show', 'png', and 'pdf' ")


def add_histogram_to_ax(ax, data, color=None, bin_width=None, x_range=None,
                        transparency=1.0, label=None):

    ax.hist(data,
            bins=find_bins(data, x_range, bin_width),
            color=color,
            edgecolor='black',
            linewidth=1,
            alpha=transparency,
            label=label)


def graph_histogram(data, title,
                    x_label=None, y_label=None, bin_width=None,
                    x_range=None, y_range=None, figure_size=None,
                    color=None, legend=None, output_type='show', file_name=None):
    """ graphs a histogram
    :param data: (list) observations
    :param title: (string) title of the figure
    :param x_label: (string) x-axis label
    :param y_label: (string) y-axis label
    :param bin_width: bin width
    :param x_range: (list with 2 elements) minimum and maximum of x-axis
    :param y_range: (list with 2 elements) minimum and maximum of y-axis
    :param figure_size: (tuple) figure size
    :param output_type: select from 'show', 'png', or 'pdf'
    :param color: (string) color
    :param legend: string for the legend
    :param file_name: (string) filename to to save the histogram as
    """

    fig, ax = plt.subplots(figsize=figure_size)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # add histogram
    add_histogram_to_ax(ax=ax,
                        data=data,
                        color=color,
                        bin_width=bin_width,
                        x_range=x_range,
                        transparency=0.75)

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    # add legend if provided
    if legend is not None:
        ax.legend([legend])

    # output figure
    if not (file_name is None):
        output_figure(fig, output_type, file_name)
    else:
        output_figure(fig, output_type, title)


def graph_histograms(data_sets, title,
                     x_label=None, y_label=None, bin_width=None,
                     x_range=None, y_range=None, figure_size=None,
                     legends=None, color_codes=None, output_type='show', transparency=1):
    """
    plots multiple histograms on a single figure
    :param data_sets: (list of lists) observations
    :param title: (string) title of the figure
    :param x_label: (string) x-axis label
    :param y_label: (string) y-axis label
    :param bin_width: bin width
    :param x_range: (list with 2 elements) minimum and maximum of x-axis
    :param y_range: (list with 2 elements) minimum and maximum of y-axis
    :param figure_size: (tuple) figure size
    :param output_type: select from 'show', 'png', or 'pdf'
    :param legends: (list) string for the legend
    :param transparency: (float) 0.0 transparent through 1.0 opaque
    """

    fig, ax = plt.subplots(figsize=figure_size)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # add histograms
    for i, data in enumerate(data_sets):
        color = None
        if color_codes is not None:
            color = color_codes[i]

        add_histogram_to_ax(ax=ax,
                            data=data,
                            bin_width=bin_width,
                            x_range=x_range,
                            color=color,
                            transparency=transparency,
                            label=legends[i])

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.legend()

    # output figure
    output_figure(plt, output_type, title)


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
    moving_ave = sum(data[0:window])/window
    averages.append(moving_ave)

    for i in range(window, len(data)):
        moving_ave = (moving_ave*window - data[i-window] + data[i])/window
        averages.append(moving_ave)

    return averages


def find_bins(data, x_range, bin_width):

    if bin_width is None:
        return 'auto'

    if x_range is not None:
        l = x_range[0]
        u = x_range[1]
    else:
        l = min(data)
        u = max(data) + bin_width
    return np.arange(l, u, bin_width)


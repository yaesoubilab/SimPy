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


def graph_histogram(data, title,
                    x_label=None, y_label=None, bin_width=None,
                    x_range=None, y_range=None,
                    legend=None, figure_size=(5, 5),
                    output_type='show', file_name=None):
    """ graphs a histogram
    :param data: (list) observations
    :param title: (string) title of the figure
    :param x_label: (string) x-axis label
    :param y_label: (string) y-axis label
    :param bin_width: bin width
    :param x_range: (list with 2 elements) minimum and maximum of x-axis
    :param y_range: (list with 2 elements) minimum and maximum of y-axis
    :param output_type: select from 'show', 'png', or 'pdf'
    :param legend: string for the legend
    :param figure_size: (tuple) figure size
    :param file_name: (string) filename to to save the histogram as
    """

    fig, ax = plt.subplots(figsize=figure_size)

    ax.set_title(title)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if bin_width is None:
        ax.hist(data,
                bins='auto',
                edgecolor='black',
                linewidth=1,
                alpha=0.75)
    else:
        ax.hist(data,
                bins=np.arange(min(data), max(data) + bin_width, bin_width),
                edgecolor='black',
                linewidth=1,
                alpha=0.75)

    if not (x_range is None):
        ax.set_xlim(x_range)
    if not (y_range is None):
        ax.set_ylim(y_range)

    # add legend if provided
    if not (legend is None):
        ax.legend([legend])

    plt.tight_layout()

    # output figure
    if not (file_name is None):
        output_figure(plt, output_type, file_name)
    else:
        output_figure(plt, output_type, title)


def graph_histogram1(data, title, x_label=None, y_label=None,
                     bin_width=None, x_range=None, y_range=None,
                     output_type='show', legend=None):
    """ graphs the histograms of multiple datasets on a single plot
    :param data: list of observations
    :param title: (string) title of the figure
    :param x_label: (string) x-axis label
    :param y_label: (string) y-axis label
    :param bin_width: bin width
    :param x_range: (list with 2 elements) minimum and maximum of x-axis
    :param y_range: (list with 2 elements) minimum and maximum of y-axis
    :param output_type: select from 'show', 'png', or 'pdf'
    :param legend: string for the legend
    """

    fig = plt.figure(title)
    plt.title(title)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)

    if bin_width is None:
        plt.hist(data,
                 bins='auto',
                 edgecolor='black',
                 linewidth=1)
    else:
        plt.hist(data,
                 bins=np.arange(min(data), max(data) + bin_width, bin_width),
                 edgecolor='black',
                 linewidth=1)

    if not (x_range is None):
        plt.xlim(x_range)
    if not (y_range is None):
        plt.ylim(y_range)

    # add legend if provided
    if not (legend is None):
        plt.legend([legend])

    # output figure
    output_figure(plt, output_type, title)


def graph_histograms(data_sets, title, x_label, y_label,
                     bin_width=None, x_range=None, y_range=None,
                     output_type='show', legend=None, transparency=1):
    """
    plots multiple histograms on a single figure
    :param data_sets: (list of lists) observations
    :param title: (string) title of the figure
    :param x_label: (string) x-axis label
    :param y_label: (string) y-axis label
    :param bin_width: bin width
    :param x_range: (list with 2 elements) minimum and maximum of x-axis
    :param y_range: (list with 2 elements) minimum and maximum of y-axis
    :param output_type: select from 'show', 'png', or 'pdf'
    :param legend: string for the legend
    :param transparency: (float) 0.0 transparent through 1.0 opaque
    """

    fig = plt.figure(title)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for data in data_sets:
        if bin_width is None:
            plt.hist(data,
                     bins='auto',
                     edgecolor='black',
                     alpha=transparency,
                     linewidth=1)
        else:
            l = min(data)
            u = max(data) + bin_width
            if not (x_range is None):
                l = x_range[0]
                u = x_range[1]
                
            plt.hist(data,
                     bins=np.arange(l, u, bin_width),
                     edgecolor='black',
                     alpha=transparency,
                     linewidth=1)

    if not (x_range is None):
        plt.xlim(x_range)
    if not (y_range is None):
        plt.ylim(y_range)

    # add legend if provided
    if not (legend is None):
        plt.legend(legend)

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




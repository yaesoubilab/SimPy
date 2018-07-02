from enum import Enum
import matplotlib.pyplot as plt
import numpy as np


class OutType(Enum):
    """output types for plotted figures"""
    SHOW = 1    # show
    JPG = 2     # save the figure as a jpg file
    PDF = 3     # save the figure as a pdf file


def output_figure(plt, output_type, title):
    """
    :param plt: reference to the plot
    :param output_type: select from OutType.SHOW, OutType.PDF, or OutType.JPG
    :param title: figure title
    :return:
    """
    # output
    if output_type == OutType.SHOW:
        plt.show()
    elif output_type == OutType.JPG:
        plt.savefig(title+".png")
    elif output_type == OutType.PDF:
        plt.savefig(title+".pdf")


def graph_histogram(data, title, x_label, y_label,
                    bin_width=None, x_range=None, y_range=None,
                    output_type=OutType.SHOW, legend=None):
    """ graphs the histograms of multiple datasets on a single plot
    :param data: list of observations
    :param title: (string) title of the figure
    :param x_label: (string) x-axis label
    :param y_label: (string) y-axis label
    :param bin_width: bin width
    :param x_range: (list with 2 elements) minimum and maximum of x-axis
    :param y_range: (list with 2 elements) minimum and maximum of y-axis
    :param output_type: select from OutType.SHOW, OutType.PDF, or OutType.JPG
    :param legend: string for the legend
    """

    fig = plt.figure(title)
    plt.title(title)
    plt.xlabel(x_label)
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
                     output_type=OutType.SHOW, legend=None, transparency=1):
    """

    :param data_sets: (list of lists) observations
    :param title: (string) title of the figure
    :param x_label: (string) x-axis label
    :param y_label: (string) y-axis label
    :param bin_width: bin width
    :param x_range: (list with 2 elements) minimum and maximum of x-axis
    :param x_range: (list with 2 elements) minimum and maximum of y-axis
    :param output_type: select from OutType.SHOW, OutType.PDF, or OutType.JPG
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
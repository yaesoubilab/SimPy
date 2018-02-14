from enum import Enum
import matplotlib.pyplot as plt
from scr import FigureSupport as Fig


def graph_histogram(observations, title, x_label, y_label, output_type=Fig.OutType.SHOW, legend=None):
    """ graphs the histogram of observations
    :param observations: list of observations
    :param title: (string) title of the figure
    :param x_label: (string) x-axis label
    :param y_label: (string) y-axis label
    :param output_type: select from OutType.SHOW, OutType.PDF, or OutType.JPG
    :param legend: string for the legend
    """

    fig = plt.figure(title)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.hist(observations,
             bins='auto',  #numpy.linspace(0, max(patient_survival_times), num_bins),
             edgecolor='black',
             linewidth=1)
    plt.xlim([0, max(observations)])

    # add legend if provided
    if not (legend is None):
        plt.legend([legend])

    # output figure
    Fig.output_figure(plt, output_type, title)

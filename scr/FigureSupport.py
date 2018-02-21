from enum import Enum
import matplotlib.pyplot as plt


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


def graph_histogram(observations, title, x_label, y_label, x_range=None, output_type=OutType.SHOW, legend=None):
    """ graphs the histogram of observations
    :param observations: list of observations
    :param title: (string) title of the figure
    :param x_label: (string) x-axis label
    :param y_label: (string) y-axis label
    :param x_range: (list with 2 elements) minimum and maximum of x-axis
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
    if not (x_range is None):
        plt.xlim(x_range)

    # add legend if provided
    if not (legend is None):
        plt.legend([legend])

    # output figure
    output_figure(plt, output_type, title)

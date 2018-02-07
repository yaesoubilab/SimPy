import matplotlib.pyplot as plt
from enum import Enum


class OutType(Enum):
    """output types for plotted figures"""
    SHOW = 1    # show
    JPG = 2     # save the figure as a jpg file
    PDF = 3     # save the figure as a pdf file


class SamplePath(object):

    def __init__(self, name, index, initial_size):
        self._name = name
        self._index = index
        self._currentSize = initial_size
        self._times = [0]
        self._observations = [initial_size]

    def record(self, time, increment):
        """
        updates the value of this sample path (e.g. number of people in the system)
        :param time: time of this chance in the system
        :param increment: (integer) change (+ or -) in value of this sample path
        """

        # store the current size
        self._times.append(time)
        self._observations.append(self._currentSize)
        # increment the size
        self._currentSize += increment
        # store the new size
        self._times.append(time)
        self._observations.append(self._currentSize)

    def get_current_size(self):
        return self._currentSize

    def get_times(self):
        return self._times

    def get_observations(self):
        return self._observations


def graph_sample_path(sample_path, title, x_label, y_label, output_type):
    """
    produces a sample path
    :param sample_path: a sample path
    :param title: (string) title of the figure
    :param x_label: (string) x-axis label
    :param y_label: (string) y-axis label
    :param output_type: select from OutType.SHOW, OutType.PDF, or OutType.JPG
    """

    fig = plt.figure(title)
    plt.title(title)        # title
    plt.xlabel(x_label)     # x-axis label
    plt.ylabel(y_label)     # y-axis label

    # x and y values
    x_values = sample_path.get_times()
    y_values = sample_path.get_observations()

    # plot
    plt.plot(x_values, y_values, '-')

    # set the minimum of y-axis to zero
    plt.ylim(ymin=0)  # the minimum has to be set after plotting the values

    # output
    if output_type.value == OutType.SHOW.value:
        plt.show()
    elif output_type.value == OutType.JPG.value:
        plt.savefig(title+".png")
    elif output_type.value == OutType.PDF:
        plt.savefig(title+".pdf")

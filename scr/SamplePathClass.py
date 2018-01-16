import matplotlib.pyplot as plt
from enum import Enum
import numpy as numpy


class OutType(Enum):
    SHOW = 1
    JPG = 2
    PDF = 3


class SamplePath(object):

    def __init__(self, name, itr, initial_size):
        self.name = name
        self.itr = itr
        self.currentSize = initial_size
        self.times = [0]
        self.observations = [initial_size]

    def record(self, time, increment):
        """
        updates the value of this sample path (e.g. number of people in the system)
        :param time: time of this chance in the system
        :param increment: (integer) change (+ or -) in value of this sample path
        """

        # store the current size
        self.times.append(time)
        self.observations.append(self.currentSize)
        # increment the size
        self.currentSize += increment
        # store the new size
        self.times.append(time)
        self.observations.append(self.currentSize)


def graph_sample_path(sample_path, title, x_label, y_label, output_type):
    """
    produces a sample path
    :param sample_path: a sample path
    :param title: (string) title of the figure
    :param x_label: (string) x-axis label
    :param y_label: (string) y-axis label
    :param output_type: select from show, pdf, or jpg
    """

    fig = plt.figure(title)
    plt.title(title)        # title
    plt.xlabel(x_label)     # x-axis label
    plt.ylabel(y_label)     # y-axis label

    # x and y values
    x_values = sample_path.times
    y_values = sample_path.observations

    # plot
    plt.plot(x_values, y_values, '-')

    # set the minimum of y-axis to zero
    plt.ylim(ymin=0)  # the minimum has to be set after plotting the values

    # output
    if output_type == OutType.SHOW.value:
        plt.show()
    elif output_type == OutType.JPG.value:
        plt.savefig(title+".png")
    elif output_type == OutType.PDF:
        plt.savefig(title+".pdf")

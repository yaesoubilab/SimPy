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


def graph_sample_path(sample_path, title, x_label, y_label, output_type, legend=None, color_code=None):
    """
    produces a sample path
    :param sample_path: a sample path
    :param title: (string) title of the figure
    :param x_label: (string) x-axis label
    :param y_label: (string) y-axis label
    :param output_type: select from OutType.SHOW, OutType.PDF, or OutType.JPG
    :param legend: string that contains the legend
    :param color_code: (string) 'b' blue 'g' green 'r' red 'c' cyan 'm' magenta 'y' yellow 'k' black
    """

    fig = plt.figure(title)
    plt.title(title)        # title
    plt.xlabel(x_label)     # x-axis label
    plt.ylabel(y_label)     # y-axis label

    # x and y values
    x_values = sample_path.get_times()
    y_values = sample_path.get_observations()

    # color
    color_marker_text = '-'
    if not (color_code is None):
        color_marker_text = color_code + color_marker_text

    # plot
    plt.plot(x_values, y_values, color_marker_text)

    # add legend if provided
    if not (legend is None):
        plt.legend([legend])
        
    # set the minimum of y-axis to zero
    plt.ylim(ymin=0)  # the minimum has to be set after plotting the values

    # output figure
    output_figure(plt, output_type, title)


def graph_sample_paths\
                (sample_paths, title, x_label, y_label, output_type,
                 legends=None, transparency=1, common_color_code=None,
                 if_same_color=False):
    """
    :param sample_paths: a list of sample paths
    :param title: (string) title of the figure
    :param x_label: (string) x-axis label
    :param y_label: (string) y-axis label
    :param output_type: select from OutType.SHOW, OutType.PDF, or OutType.JPG
    :param legends: list of strings for legend
    :param transparency: float (0.0 transparent through 1.0 opaque)
    :param common_color_code: (string) color code if all sample paths should have the same color
        'b'	blue 'g' green 'r' red 'c' cyan 'm' magenta 'y' yellow 'k' black
    :param if_same_color: logical, default False, if set True, paint the sample paths the same color
    """

    if len(sample_paths) == 1:
        raise ValueError('Only one sample path is provided. Use graph_sample_path instead.')

    fig = plt.figure(title)
    plt.title(title)        # title
    plt.xlabel(x_label)     # x-axis label
    plt.ylabel(y_label)     # y-axis label

    # color
    color_marker_text = '-'
    if not (common_color_code is None):
        color_marker_text = common_color_code+color_marker_text

        # x and y values
    if if_same_color:
        for path in sample_paths:
            x_values = path.get_times()
            y_values = path.get_observations()
            # plot
            plt.plot(x_values, y_values, common_color_code, alpha=transparency)
    else:
        for path in sample_paths:
            x_values = path.get_times()
            y_values = path.get_observations()
            # plot
            plt.plot(x_values, y_values, color_marker_text, alpha=transparency)

    # add legend if provided
    if not (legends is None):
        if common_color_code is None:
            plt.legend(legends)
        else:
            plt.legend([legends])


    # set the minimum of y-axis to zero
    plt.ylim(ymin=0)  # the minimum has to be set after plotting the values

    # output figure
    output_figure(plt, output_type, title)


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
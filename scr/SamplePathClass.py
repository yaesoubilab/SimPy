import matplotlib.pyplot as plt
from scr import FigureSupport as Fig


class SamplePath(object):

    def __init__(self, name, index, initial_size):
        self._name = name
        self._index = index
        self._currentValue = initial_size
        self._times = [0]
        self._values = [initial_size]

    def record(self, time, increment):
        """
        updates the value of this sample path (e.g. number of people in the system)
        :param time: time of this chance in the system
        :param increment: (integer) change (+ or -) in value of this sample path
        """
        raise NotImplementedError("Abstract method not implemented.")

    def get_times(self):
        """ returns the times of changes """
        raise NotImplementedError("Abstract method not implemented.")

    def get_values(self):
        """" returns the value of this sample path at times when changes occured """
        raise NotImplementedError("Abstract method not implemented.")


class SamplePathRealTimeUpdate(SamplePath):
    """ a sample path where observations are recorded in real-time (over the simulation) """
    def __init__(self, name, index, initial_size):
        SamplePath.__init__(self, name, index, initial_size)

    def record(self, time, increment):

        # store the current size
        self._times.append(time)
        self._values.append(self._currentValue)
        # increment the current value
        self._currentValue += increment
        # store the new value
        self._times.append(time)
        self._values.append(self._currentValue)

    def get_current_value(self):
        """ :return the current value of this sample path"""
        return self._currentValue

    def get_times(self):
        return self._times

    def get_values(self):
        return self._values


class SamplePathBatchUpdate(SamplePath):
    """ a sample path where observations are recorded at the end of the simulation """
    def __init__(self, name, index, initial_size):
        SamplePath.__init__(self, name, index, initial_size)

        self._samplePath = SamplePathRealTimeUpdate(name, index, initial_size)
        self._ifProcessed = False
        self._t = []
        self._obs = []

    def record(self, time, increment):
        self._t.append(time)
        self._obs.append(increment)

    def get_times(self):

        # create the sample path if not created yet
        if not self._ifProcessed:
            self.__process();

        return self._samplePath.get_times()

    def get_values(self):

        # create the sample path if not created yet
        if not self._ifProcessed:
            self.__process();

        return self._samplePath.get_values()

    def __process(self):

        # population a list with recorded times and incremetal values
        l = []
        for i in range(len(self._t)):
            l.append([self._t[i], self._obs[i]])

        # sort this list based on the recorded time
        new_list = sorted(l, key=lambda x : x[0])

        # create teh sample path
        for item in new_list:
            self._samplePath.record(item[0], item[1])

        # proceed
        self._ifProcessed = True


def graph_sample_path(sample_path, title, x_label, y_label,
                      output_type=Fig.OutType.SHOW, legend=None, color_code=None):
    """
    produces a sample path
    :param sample_path: a sample path
    :param title: (string) title of the figure
    :param x_label: (string) x-axis label
    :param y_label: (string) y-axis label
    :param output_type: select from OutType.SHOW, OutType.PDF, or OutType.JPG
    :param legend: string for the legend
    :param color_code: (string) 'b' blue 'g' green 'r' red 'c' cyan 'm' magenta 'y' yellow 'k' black
    """

    fig = plt.figure(title)
    plt.title(title)        # title
    plt.xlabel(x_label)     # x-axis label
    plt.ylabel(y_label)     # y-axis label

    # x and y values
    x_values = sample_path.get_times()
    y_values = sample_path.get_values()

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
    Fig.output_figure(plt, output_type, title)


def graph_sample_paths\
                (sample_paths, title, x_label, y_label, output_type=Fig.OutType.SHOW,
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
            y_values = path.get_values()
            # plot
            plt.plot(x_values, y_values, common_color_code, alpha=transparency)
    else:
        for path in sample_paths:
            x_values = path.get_times()
            y_values = path.get_values()
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
    Fig.output_figure(plt, output_type, title)

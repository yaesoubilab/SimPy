import matplotlib.pyplot as plt

import SimPy.StatisticalClasses as Stat
from SimPy import FigureSupport as Fig


class _SamplePath:

    def __init__(self, name, sim_rep=0, collect_stat=True):
        """
        :param name: name of this sample path
        :param sim_rep: (int) simulation replication of this sample path
        :param collect_stat: set to True to collect statistics on
                            average, max, min, stDev, etc for this sample path
        """
        self.name = name
        self.simRep = sim_rep
        self._period_num = 0

        self._times = []  # times at which observations should be recorded
        self._values = []  # value of this sample path over time
        # statistics on this prevalence sample path
        self.ifCollectStat = collect_stat

    def record_increment(self, time, increment):
        raise NotImplemented()

    def record_value(self, time, value):
        raise NotImplemented()

    def close(self, time):
        raise NotImplemented()

    def get_values(self):
        return self._values


class PrevalenceSamplePath(_SamplePath):

    def __init__(self, name, initial_size=0, sim_rep=0, collect_stat=True, t_warm_up=0):
        """
        :param name: name of this sample path
        :param initial_size: value of the sample path at simulation time 0
        :param sim_rep: (int) simulation replication of this sample path

        :param collect_stat: set to True to collect statistics
                        on average, max, min, stDev, etc for this sample path
        """
        _SamplePath.__init__(self, name=name, sim_rep=sim_rep, collect_stat=collect_stat)
        self.currentSize = initial_size  # current size of the sample path
        self._times = [0]
        self._values = [initial_size]
        # statistics on this prevalence sample path
        if collect_stat:
            self.stat = Stat.ContinuousTimeStat(name=name, initial_time=t_warm_up)

    def record_increment(self, time, increment):
        """
        updates the value of this sample path (e.g. number of people in the system)
        :param time: time of this change
        :param increment: (integer) change (+ or -) in value of this sample path
        """

        if time < self._times[-1]:
            raise ValueError(self.name + ' | Current time cannot be less than the last recorded time.')
        if increment is None:
            raise ValueError(self.name + ' | increment cannot be None.')

        self.currentSize += increment

        # update stat
        if self.ifCollectStat:
            self.stat.record(time=time, increment=increment)

        if time == self._times[-1]:
            self._values[-1] = self.currentSize
        else:
            self._times.append(time)
            self._values.append(self.currentSize)

    def record_value(self, time, value):
        """
        updates the value of this sample path (e.g. number of people in the system)
        :param time: time of this change
        :param value:
        """

        if value is None:
            raise ValueError(self.name + ' | value cannot be None.')

        self.record_increment(time=time, increment=value-self.currentSize)

    def close(self, time):
        self.record_increment(time=time, increment=0)

    def get_times(self):
        return self._times


class IncidenceSamplePath(_SamplePath):
    def __init__(self, name, delta_t, sim_rep=0, collect_stat=True, t_warm_up=0):
        """
        :param name: name of this sample path
        :param delta_t: length of equally-spaced observation periods
        :param sim_rep: (int) simulation replication of this sample path
        :param collect_stat: set to True to collect statistics on average, max, min, stDev, etc for this sample path

        """
        _SamplePath.__init__(self, name=name, sim_rep=sim_rep, collect_stat=collect_stat)
        self._deltaT = delta_t
        self._period_num = 0
        self._t_warm_up = t_warm_up
        self._period_nums = []  # times represent the observation period number
        self._values = []
        # statistics on this incidence sample path
        if self.ifCollectStat:
            self.stat = Stat.DiscreteTimeStat(name=name)

    def record_increment(self, time, increment):
        """
        updates the value of this sample path (e.g. number of people in the system)
        :param time: time of this change
        :param increment: (integer) change (+ or -) in value of this sample path
        """

        if len(self._period_nums) > 0 and time < (self._period_nums[-1]-1)*self._deltaT:
            raise ValueError(self.name + ' | Current time cannot be less than the last recorded time.')
        if increment is None:
            raise ValueError(self.name + ' | increment cannot be None.')
        if increment < 0:
            raise ValueError(self.name + ' | increment cannot be negative.')

        if time > self._period_num * self._deltaT:
            self.__fill_unrecorded_periods(time=time)

            self._period_num += 1
            self._values.append(increment)
            self._period_nums.append(self._period_num)

            if self.ifCollectStat:
                self.stat.record(obs=self._values[-1])

        else:
            self._values[-1] += increment

    def record_value(self, time, value):
        """
        updates the value of this sample path (e.g. number of people diagnosed in the past year)
        :param time:
        :param value:
        """

        if value is None:
            raise ValueError(self.name + ' | value cannot be None.')

        self.record_increment(time=time, increment=value)

    def close(self, time):
        self.__fill_unrecorded_periods(time=time)

    def get_period_numbers(self):
        return self._period_nums

    def __fill_unrecorded_periods(self, time):
        while time > (self._period_num + 1) * self._deltaT:
            self._period_num += 1
            self._values.append(0)
            self._period_nums.append(self._period_num)

            if self.ifCollectStat:
                self.stat.record(obs=self._values[-1])


class PrevalencePathBatchUpdate(PrevalenceSamplePath):
    """ a sample path for which observations are recorded at the end of the simulation """

    def __init__(self, name, initial_size, times_of_changes, increments, sim_rep=0):
        """
        :param name: name of this sample path
        :param initial_size: (int) value of the sample path at simulation time 0
        :param times_of_changes: (list) times at which changes occurred
        :param increments: (list) level of changes
        :param sim_rep: (int) simulation replication of this sample path
        """

        if len(times_of_changes) != len(increments):
            raise ValueError('The list of times at which changes occurred should '
                             'be the same size as the list of changes.')

        PrevalenceSamplePath.__init__(self, name=name, initial_size=initial_size, sim_rep=sim_rep)

        # populate the list of [time, value] of recordings
        self._time_and_values = []
        for i in range(len(times_of_changes)):
            self._time_and_values.append([times_of_changes[i], increments[i]])

        # the batch sample path will be converted to real-time sample path
        self._samplePath = PrevalenceSamplePath(name, initial_size, sim_rep)
        self._ifProcessed = False   # set to True when the sample path is built

    def record(self, time, increment):
        pass

    def get_times(self):
        """
        :return: times when changes in the sample path recorded
        """

        # build the sample path if not build yet
        if not self._ifProcessed:
            self.__process()
        return self._samplePath._times

    def get_values(self):
        """
        :return: value of the sample path at times when changes in the sampel path recorded
        """

        # build the sample path if not build yet
        if not self._ifProcessed:
            self.__process()
        return self._samplePath._values

    def __process(self):
        """ will build the sample path when requested """

        # sort this list based on the recorded time
        new_list = sorted(self._time_and_values, key=lambda x: x[0])

        # create the sample path
        for item in new_list:
            self._samplePath.record_increment(item[0], item[1])

        # proceed
        self._ifProcessed = True


def graph_sample_path(sample_path,
                      title=None, x_label=None, y_label=None,
                      figure_size=None, output_type='show',
                      legend=None, color_code=None, connect='step'):
    """
    plot a sample path
    :param sample_path: a sample path
    :param title: (string) title of the figure
    :param x_label: (string) x-axis label
    :param y_label: (string) y-axis label
    :param figure_size: (tuple) figure size
    :param output_type: select from 'show', 'pdf' or 'png'
    :param legend: string for the legend
    :param color_code: (string) 'b' blue 'g' green 'r' red 'c' cyan 'm' magenta 'y' yellow 'k' black
    :param connect: (string) set to 'step' to produce an step graph and to 'line' to produce a line graph
    """

    if not isinstance(sample_path, _SamplePath):
        raise ValueError(
            'sample_path should be an instance of PrevalenceSamplePath or PrevalencePathBatchUpdate.')

    fig, ax = plt.subplots(figsize=figure_size)
    ax.set_title(title)  # title
    ax.set_xlabel(x_label)  # x-axis label
    ax.set_ylabel(y_label)  # y-axis label

    # add a sample path to this ax
    add_sample_path_to_ax(sample_path=sample_path,
                          ax=ax,
                          color_code=color_code,
                          legend=legend,
                          connect=connect)
    ax.set_ylim(bottom=0)  # the minimum has to be set after plotting the values

    if isinstance(sample_path, PrevalenceSamplePath):
        ax.set_xlim(left=0)
    elif isinstance(sample_path, IncidenceSamplePath):
        ax.set_xlim(left=0.5)

    # output figure
    Fig.output_figure(fig, output_type, title)


def graph_sample_paths(sample_paths,
                       title=None, x_label=None, y_label=None,
                       figure_size=None, output_type='show',
                       legends=None, transparency=1, common_color_code=None, connect='step'):
    """ graphs multiple sample paths
    :param sample_paths: a list of sample paths
    :param title: (string) title of the figure
    :param x_label: (string) x-axis label
    :param y_label: (string) y-axis label
    :param figure_size: (tuple) figure size
    :param output_type: select from 'show', 'pdf' or 'png'
    :param legends: list of strings for legend
    :param transparency: float (0.0 transparent through 1.0 opaque)
    :param common_color_code: (string) color code if all sample paths should have the same color
        'b'	blue 'g' green 'r' red 'c' cyan 'm' magenta 'y' yellow 'k' black
    :param connect: (string) set to 'step' to produce an step graph and to 'line' to produce a line graph
    """

    if len(sample_paths) == 1:
        raise ValueError('Only one sample path is provided. Use graph_sample_path instead.')

    fig, ax = plt.subplots(figsize=figure_size)
    ax.set_title(title)  # title
    ax.set_xlabel(x_label)  # x-axis label
    ax.set_ylabel(y_label)  # y-axis label

    # add all sample paths
    add_sample_paths_to_ax(sample_paths=sample_paths,
                           ax=ax,
                           common_color_code=common_color_code,
                           transparency=transparency,
                           connect=connect)

    # add legend if provided
    if legends is not None:
        if common_color_code is None:
            ax.legend(legends)
        else:
            ax.legend([legends])

    # set the minimum of y-axis to zero
    ax.set_ylim(bottom=0)  # the minimum has to be set after plotting the values
    # output figure
    Fig.output_figure(fig, output_type, title)


def graph_sets_of_sample_paths(sets_of_sample_paths,
                               title=None, x_label=None, y_label=None,
                               figure_size=None, output_type='show',
                               legends=None, transparency=1, color_codes=None, connect='step'):
    """ graphs multiple sample paths
    :param sets_of_sample_paths: (list) of list of sample paths
    :param title: (string) title of the figure
    :param x_label: (string) x-axis label
    :param y_label: (string) y-axis label
    :param figure_size: (tuple) figure size
    :param output_type: select from 'show', 'pdf' or 'png'
    :param legends: list of strings for legend
    :param transparency: float (0.0 transparent through 1.0 opaque)
    :param color_codes: (list of strings) color code of sample path sets
            e.g. 'b' blue 'g' green 'r' red 'c' cyan 'm' magenta 'y' yellow 'k' black
    :param connect: (string) set to 'step' to produce an step graph and to 'line' to produce a line graph
    """

    if len(sets_of_sample_paths) == 1:
        raise ValueError('Only one set of sample paths is provided. Use graph_sample_paths instead.')

    fig, ax = plt.subplots(figsize=figure_size)
    ax.set_title(title)  # title
    ax.set_xlabel(x_label)  # x-axis label
    ax.set_ylabel(y_label)  # y-axis label

    # add all sample paths
    add_sets_of_sample_paths_to_ax(sets_of_sample_paths=sets_of_sample_paths,
                                   ax=ax,
                                   color_codes=color_codes,
                                   legends=legends,
                                   transparency=transparency,
                                   connect=connect)

    # set the minimum of y-axis to zero
    ax.set_ylim(bottom=0)  # the minimum has to be set after plotting the values
    # output figure
    Fig.output_figure(fig, output_type, title)


def add_sample_path_to_ax(sample_path, ax, color_code=None, legend=None, transparency=1, connect='step'):

    # x and y values
    if isinstance(sample_path, PrevalenceSamplePath):
        x_values = sample_path.get_times()
    elif isinstance(sample_path, IncidenceSamplePath):
        x_values = sample_path.get_period_numbers()

    y_values = sample_path.get_values()

    # plot the sample path
    if connect == 'step':
        ax.step(x=x_values, y=y_values, where='post', color=color_code, label=legend, alpha=transparency)
    else:
        ax.plot(x_values, y_values, color=color_code, label=legend, alpha=transparency)

    # add legend if provided
    if legend is not None:
        ax.legend()


def add_sample_paths_to_ax(sample_paths, ax, common_color_code, transparency, connect):

    # add every path
    for path in sample_paths:
        add_sample_path_to_ax(sample_path=path,
                              ax=ax,
                              color_code=common_color_code,
                              transparency=transparency,
                              connect=connect)


def add_sets_of_sample_paths_to_ax(sets_of_sample_paths, ax, color_codes, legends, transparency, connect):

    # add every path
    for i, sample_paths in enumerate(sets_of_sample_paths):
        for j, path in enumerate(sample_paths):
            if j == 0:
                legend = legends[i]
            else:
                legend = None
            add_sample_path_to_ax(sample_path=path,
                                  ax=ax,
                                  color_code=color_codes[i],
                                  legend=legend,
                                  transparency=transparency,
                                  connect=connect)
import SimPy.StatisticalClasses as Stat
import SimPy.DataFrames as DF


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

    def get_current_value(self):
        return self._values[-1]


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


class PrevalencePyramidSamplePath:
    """
    example:
        age,   sex,      Prevalence Sample Path
        0,     0,        SamplePath1,
        0,     1,        SamplePath2,
        5,     0,        SamplePath3,
        5,     1,        SamplePath4,
        10,    0,        SamplePath5,
        10,    1,        SamplePath6
    """

    def __init__(self, age_min, age_max, age_delta, sim_rep=0,
                 collect_stat=False, t_warm_up=0):
        """
        :param age_min: list of minimum value of x (in example above: [0, 0])
        :param age_max: list of maximum value of x (in example above: [10, 1])
        :param age_delta: list of interval between break points of x
                    if set to 'int', x is treated as categorical variable
                    (in example above: [5, 'int'])
        """

        self.ageMin = age_min
        self.ageMax = age_max
        self.ageDelta = age_delta

        self.pyramid = DF._DataFrame(list_x_min=[age_min, 0],
                                     list_x_max=[age_max, 1],
                                     list_x_delta=[age_delta, 'int'])

        age = age_min
        sex = 0

        while age <= self.ageMax:
            while sex <= 1:
                self.pyramid.set_obj(x_value=[age, sex],
                                     obj=PrevalenceSamplePath(
                                         name='Age {}, sex {}'.format(age, sex),
                                         initial_size=0,
                                         sim_rep=sim_rep,
                                         collect_stat=collect_stat,
                                         t_warm_up=t_warm_up
                                     )
                )
                sex += 1

            age += self.ageDelta

    def record_increment(self, time, increment, age, sex):
        """
        updates the value of this sample path (e.g. number of people in the system)
        :param time: time of this change
        :param increment: (integer) change (+ or -) in value of this sample path
        """

        self.pyramid.get_obj(x_value=[age, sex]).record_increment(
            time=time, increment=increment)

    def record_value(self, time, value, age, sex):
        """
        updates the value of this sample path (e.g. number of people in the system)
        :param time: time of this change
        :param value:
        """

        self.pyramid.get_obj(x_value=[age, sex]).record_value(
            time=time, value=value)

    def get_current_value(self, age, sex):
        """
        updates the value of this sample path (e.g. number of people in the system)
        :param time: time of this change
        :param value:
        """

        self.pyramid.get_obj(x_value=[age, sex]).get_current_value()

    def close(self, time):

        age = self.ageMin
        sex = 0
        while age <= self.ageMax:
            while sex <= 1:
                self.pyramid.get_obj(x_value=[age, sex]).close(time=time)
                sex += 1

            age += self.ageDelta

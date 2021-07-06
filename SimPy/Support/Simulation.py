import os

from numpy import iinfo, int32

import SimPy.InOutFunctions as io


class _Trace:
    def __init__(self, if_should_trace, deci):
        """ creates a list to store trace messages
        :param if_should_trace: enter true to trace the simulation
        :param deci: number of decimals to round time values to
        """

        self._on = if_should_trace          # if set to False, the trace messages will not be stored
        self._deci = deci
        self._messages = []                 # list of strings to store trace messages
        self._tOfLastMessage = 0            # time when the last message was recorded

    def _add_message(self, time, message):
        """ adds the entered text to the trace list
        :param time: (float) current simulation time
        :param message: (string) the message to describe what happened at the current time
        """
        if not self._on:
            return

        # if the time has changed since the last message, add an empty message
        if time > self._tOfLastMessage:
            self._messages.append('---')

        # message
        text = "At {t:.{prec}f}: ".format(t=time, prec=self._deci) + message

        # record the message
        self._messages.append(text)

        # update the time of last message
        self._tOfLastMessage = time

    def get_trace(self):
        """
        :return: the list of trace messages
        """
        if not self._on:
            return

        return self._messages

    def print_trace(self, filename, directory='Trace', delete_existing_files=True):
        """ print the trace messages into a text file with the specified filename.
        It creates a sub directory where the python script is located.
        :param filename: filename of the text file where trace message should be exported to
        :param directory: directory (relative to the current root) where the trace files should be located
        :param delete_existing_files: set to True to delete the existing trace files in the directory
        """
        if not self._on:
            return

        # create the directory if does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        # delete existing files
        if delete_existing_files:
            io.delete_files(extension='.txt', path=os.getcwd()+'/'+directory)

        # create a new file
        filename = os.path.join(directory, filename)
        # open the file with a write access
        file = open(filename, 'w')
        # write the trace messages
        for message in self._messages:
            file.write('%s\n' % message)
        # close the file
        file.close()


class Trace(_Trace):
    def __init__(self, if_should_trace, deci):
        """ creates a list to store trace messages
        :param if_should_trace: enter true to trace the simulation
        :param deci: number of decimals to round time values to
        """
        _Trace.__init__(self, if_should_trace=if_should_trace, deci=deci)

    def add_message(self, time, message):
        """ adds the entered text to the trace list
        :param time: (float) current simulation time
        :param message: (string) the message to describe what happened at the current time
        """
        self._add_message(time=time, message=message)


class DiscreteEventSimTrace(_Trace):
    # trace for discrete-event simulation models

    def __init__(self, sim_calendar, if_should_trace, deci):
        """ creates a list to store trace messages
        :param sim_calendar: simulation calendar
        :param if_should_trace: enter true to trace the simulation
        :param deci: number of decimals to round time values to
        """

        _Trace.__init__(self, if_should_trace=if_should_trace, deci=deci)
        self._simCalendar = sim_calendar    # simulation calender (to get the current simulation time)

    def add_message(self, message):
        """ adds the entered text to the trace list
        :param message: (string) the message to describe what happened at the current time
        """

        self._add_message(time=self._simCalendar.time, message=message)


class SeedGenerator:

    def __init__(self, seeds=None, weights=None):
        """
        :param seeds: (list) of seeds
        :param weights: (list) of seed weighs
        """

        if seeds is not None and weights is not None:
            if len(seeds) != len(weights):
                raise ValueError('There should be equal number of seeds and weights.')
            if sum(weights) == 0:
                raise ValueError('Weights of all provided seeds are zero. '
                                 'Make sure that at least one seed has a positive weight.')

        self.seeds = seeds
        self.weights = weights
        self.max = iinfo(int32).max

        self.i = -1
        self.posWeights = []
        self.seedsWithPosWeights = []
        if self.weights is not None:
            for s, w in zip(self.seeds, self.weights):
                if w > 0:
                    self.seedsWithPosWeights.append(s)
                    self.posWeights.append(w)

    def next_seed(self, rng=None, sample_by_weight=False):

        if self.seeds is not None:
            # if seeds are provided
            if sample_by_weight:
                # if sample seeds according to weights
                return rng.choice(a=self.seeds, size=1, p=self.weights)
            else:
                # if seeds should not be sampled, return the next seed with positive weight
                if self.i + 1 < len(self.seedsWithPosWeights):
                    self.i += 1
                else:
                    self.i = 0
                return self.seedsWithPosWeights[self.i]
        else:
            # if seeds are not provided, return a random integer
            return rng.randint(0, self.max)

    def next_seeds(self, n, rng=None, sample_by_weight=False):

        seeds = []
        for i in range(n):
            seeds.append(self.next_seed(rng=rng, sample_by_weight=sample_by_weight))

        return seeds

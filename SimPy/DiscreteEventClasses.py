import heapq
import os
import SimPy.InOutFunctions as io


class SimulationEvent:
    def __init__(self, time, priority):
        """
        :param time: (float) time of the event
        :param priority: priority of the event (the lowest value implies the highest priority)
        """
        self._time = time
        self._priority = priority

    def get_time(self):
        """
        :returns time when this event is going to occur """
        return self._time

    def get_priority(self):
        """
        :returns priority of the event"""
        return self._priority

    def process(self):
        """ implements instruction to process this event once occurs
        abstract method to be overridden in derived classes to process an event """
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")


class SimulationCalendar:
    def __init__(self):
        """  create a simulation calendar """

        self._q = []  # a list (priority queue) to store simulation events
        self._currentTime = 0

    def get_current_time(self):
        """
        :returns the current time """

        return self._currentTime

    def n_events(self):
        """
        :returns number of scheduled events"""

        return len(self._q)

    def add_event(self, event):
        """ add a new event to the calendar (sorted by event time and then priority)
        :param event: a simulation event to be added to the simulation calendar """

        entry = [event.get_time(), event.get_priority(), event]
        heapq.heappush(self._q, entry)

    def get_next_event(self):
        """
        :return: the next simulation event
        (if time of events are equal, the event with lowest value for priority will be returned.) """

        self._currentTime, priority, next_event = heapq.heappop(self._q)
        return next_event

    def clear_calendar(self):
        """ clear the simulation calendar (deletes all scheduled events) """

        self._q.clear()


class Trace:
    def __init__(self, sim_calendar, if_should_trace, deci):
        """ creates a list to store trace messages
        :param sim_calendar: simulation calendar
        :param if_should_trace: enter true to trace the simulation
        :param deci: number of decimals to round time values to
        """

        self._on = if_should_trace          # if set to False, the trace messages will not be stored
        self._deci = deci
        self._messages = []                 # list of strings to store trace messages
        self._simCalendar = sim_calendar    # simulation calender (to get the current simulation time)

    def add_message(self, message):
        """ adds the entered text to the trace list
        :param message: (string) the message to describe what happened at the current time
        """
        if not self._on:
            return

        text = "At {t:.{prec}f}: ".format(t=self._simCalendar.get_current_time(), prec=self._deci) + message
        self._messages.append(text)

    def get_trace(self):
        """
        :return: the list of trace messages
        """
        if not self._on:
            return

        return self._messages

    def print_trace(self, filename, path='..'):
        """ print the trace messages into a text file with the specified filename
        :param filename: filename of the text file where trace message should be exported to
        :param path: path (relative to the current root '..') where the trace files should be located
        (the folder should already exist)
        """
        if not self._on:
            return

        # create a new file
        filename = os.path.join(path, filename)
        # open the file with a write access
        file = open(filename, 'w')
        # write the trace messages
        for message in self._messages:
            file.write('%s\n' % message)
        # close the file
        file.close()


def clear_txt_files(path='..'):
    """ removes every .txt files inside the directory
    :param path: path (relative to the current root) where the .txt files are located
    (the folder should already exist)
    """

    io.delete_files('.txt', path)
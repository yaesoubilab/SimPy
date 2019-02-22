import heapq
import os
import SimPy.InOutFunctions as io


class SimulationEvent:
    def __init__(self, time, priority):
        """
        :param time: (float) time of the event
        :param priority: priority of the event (the lowest value implies the highest priority)
        """
        self.time = time            # event time
        self.priority = priority    # event priority

    def process(self):
        """ implements instruction to process this event once occurs
        abstract method to be overridden in derived classes to process an event """
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")


class SimulationCalendar:
    def __init__(self):
        """  create a simulation calendar """

        self._q = []  # a list (priority queue) to store simulation events
        self.time = 0   # current time

    def n_events(self):
        """
        :returns number of scheduled events"""

        return len(self._q)

    def add_event(self, event):
        """ add a new event to the calendar (sorted by event time and then priority)
        :param event: a simulation event to be added to the simulation calendar """

        if event.time < self.time:
            raise ValueError('An event with event time past the current time cannot be added to the calendar.')

        entry = [event.time, event.priority, event]
        heapq.heappush(self._q, entry)

    def get_next_event(self):
        """
        :return: the next simulation event
        (if time of events are equal, the event with lowest value for priority will be returned.) """

        self.time, priority, next_event = heapq.heappop(self._q)
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
        self._tOfLastMessage = 0            # time when the last message was recorded

    def add_message(self, message):
        """ adds the entered text to the trace list
        :param message: (string) the message to describe what happened at the current time
        """
        if not self._on:
            return

        # if the time has changed since the last message, add an empty message
        if self._simCalendar.time > self._tOfLastMessage:
            self._messages.append('---')

        # message
        text = "At {t:.{prec}f}: ".format(t=self._simCalendar.time, prec=self._deci) + message

        # record the message
        self._messages.append(text)

        # update the time of last message
        self._tOfLastMessage = self._simCalendar.time

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


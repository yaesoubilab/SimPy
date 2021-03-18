import heapq


class SimulationEvent:
    def __init__(self, time, priority):
        """
        :param time: (float) time of the event
        :param priority: priority of the event (the lowest value implies the highest priority)
        """
        self.time = time            # event time
        self.priority = priority    # event priority

    def process(self, rng=None):
        """ implements instruction to process this event once occurs
        abstract method to be overridden in derived classes to process an event
        :param rng: random number generator
        """

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
            raise ValueError('An event with event time less than the current time cannot be added to the calendar.')

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

    def reset(self):
        """ clear the simulation calendar (deletes all scheduled events) """

        self.time = 0
        self._q.clear()

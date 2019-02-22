import SimPy.DiscreteEventSim as Cls

# create a simulation event calendar
calendar = Cls.SimulationCalendar()

# create simulation events and add them to the calendar
calendar.add_event(Cls.SimulationEvent(time=2.0, priority=1))
calendar.add_event(Cls.SimulationEvent(time=10.0, priority=2))
calendar.add_event(Cls.SimulationEvent(time=1.1, priority=2))
calendar.add_event(Cls.SimulationEvent(time=1.1, priority=1))

# print the first two events
nextEvent = calendar.get_next_event()
print(nextEvent.get_time(), nextEvent.get_priority())

nextEvent = calendar.get_next_event()
print (nextEvent.get_time(), nextEvent.get_priority())


# testing trace
def make_trace():
    calendar = Cls.SimulationCalendar()
    myTrace = Cls.Trace(calendar, True, 2)
    myTrace._add_message("Message 1")
    myTrace._add_message("Message 2")

    myTrace.print_trace(filename='TraceTest2.txt', directory='trace')


make_trace()

#Cls.clear_trace_files('../tests/trace')

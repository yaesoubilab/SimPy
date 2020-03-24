from SimPy.DiscreteEventSim import *


class UpdateCompartments(SimulationEvent):
    def __init__(self, time, epi_model):
        SimulationEvent.__init__(self, time=time, priority=0)
        self.epiModel = epi_model

    def process(self, rng=None):
        self.epiModel.update_compartments(rng)

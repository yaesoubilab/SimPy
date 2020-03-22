import math
import SimPy.RandomVariantGenerators as RVGs
from SimPy.DiscreteEventSim import *


class _Compartment:

    def __init__(self, id, name, size):
        self.id = id
        self.name = name
        self.size = size
        self.n_incoming = 0

    def update_size(self):
        self.size += self.n_incoming


class Compartment(_Compartment):

    def __init__(self, id, name, size=0):
        _Compartment.__init__(self, id, name, size)
        self.events = []

    def add_event(self, epi_event):
        self.events.append(epi_event)

    def sample_outgoing(self, rng, delta_t):

        rates_out = []
        for e in self.events:
            rates_out.append(e.rate)

        probs_out = []
        sum_rates = sum(rates_out)
        probs_out.append(math.exp(-sum_rates*delta_t))

        for e in self.events:
            probs_out.append((1-probs_out[0]) * e.rate/sum_rates)

        outs = RVGs.Multinomial(N=self.size, pvals=probs_out).sample(rng=rng)

        for i, e in enumerate(self.events):
            e.destComp.n_incoming += outs[i+1]
            self.size -= outs[i+1]


class ChanceNode(_Compartment):
    def __init__(self, id, name, dest_comps):
        _Compartment.__init__(self, id, name, size=0)
        self.destComps = dest_comps


class _EpiEvent:

    def __init__(self, id, name, dest_comp):
        self.id = id
        self.name = name
        self.rate = 0
        self.destComp = dest_comp

    def update_rate(self):
        pass


class EpiDepEvent(_EpiEvent):
    def __init__(self, id, name, trans_par, inf_comp, denom, dest_comp):
        _EpiEvent.__init__(self, id, name, dest_comp)
        self.transPar = trans_par
        self.infComp = inf_comp
        self.denom = denom

    def update_rate(self):
        self.rate = self.transPar.value * self.infComp.size/self.denom


class EpiIndepEvent(_EpiEvent):
    def __init__(self, id, name, rate_par, dest_comp):
        _EpiEvent.__init__(self, id, name, dest_comp)
        self.ratePar = rate_par

    def update_rate(self):
        self.rate = self.ratePar.value


class EpiModel:
    def __init__(self, id, parameters, delta_t):
        self.id = id
        self.params = parameters
        self.deltaT = delta_t
        self.comparts = []
        self.events = []
        self.simCal = SimulationCalendar()

    def __initialize(self):

        for c in self.comparts:
            for e in c.events:
                self.events.append(e)

        self.simCal.add_event(event=UpdateCompartments(time=0, epi_model=self))

    def build_model(self):
        pass

    def process_end_of_sim(self):
        pass

    def simulate(self, sim_duration):

        rng = RVGs.RNG(seed=self.id)
        self.build_model()

        self.__initialize()
        while self.simCal.n_events() > 0 and self.simCal.time <= sim_duration:
            self.simCal.get_next_event().process(rng=rng)

        self.process_end_of_sim()

    def update_compartments(self, rng):

        print(self.get_epi_state())

        self.params.update(rng=rng, time=self.simCal.time)
        for e in self.events:
            e.update_rate()
        for c in self.comparts:
            c.sample_outgoing(rng=rng, delta_t=self.deltaT)
        for c in self.comparts:
            c.update_size()

        self.simCal.add_event(event=UpdateCompartments(time=self.simCal.time + self.deltaT,
                                                       epi_model=self))

    def get_epi_state(self):

        return [c.size for c in self.comparts]


class UpdateCompartments(SimulationEvent):
    def __init__(self, time, epi_model):
        SimulationEvent.__init__(self, time=time, priority=0)
        self.epiModel = epi_model

    def process(self, rng=None):
        self.epiModel.update_compartments(rng)

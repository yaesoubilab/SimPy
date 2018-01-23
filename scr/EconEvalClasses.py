import numpy as numpy
import scr.StatisticalClasses as Stat


class ICER:
    def __init__(self, name, cost_intervention, health_intervention, cost_base, health_base):
        self._name = name
        self._costIntervention = numpy.array(cost_intervention)
        self._healthIntervention = numpy.array(health_intervention)
        self._costBase = numpy.array(cost_base)
        self._healthBase = numpy.array(health_base)
        self._ICER = 0
        
    def get_ICER(self):
        return self._ICER

    def get_CI(self, alpha, num_bootstrap_samples):
        pass

    def get_PI(self, alpha):
        pass


class ICERPaired(ICER):

    def __init__(self, name, cost_intervention, health_intervention, cost_base, health_base):
        ICER.__init__(self, name, cost_intervention, health_intervention, cost_base, health_base)

        self._deltaCost = self._costIntervention - self._costBase
        self._deltaHealth = self._healthIntervention - self._healthBase
        self._ICER = numpy.average(self._deltaCost) / numpy.average(self._deltaHealth)

    def get_CI(self, alpha, num_bootstrap_samples):
        ratio_stat = Stat.RatioStatPaired(self._name, self._deltaCost, self._deltaHealth)
        return ratio_stat.get_bootstrap_CI(alpha, num_bootstrap_samples)

    def get_PI(self, alpha):
        ratio_stat = Stat.RatioStatPaired(self._name, self._deltaCost, self._deltaHealth)
        return ratio_stat.get_PI(alpha)


class ICERIndepedent(ICER):

    def __init__(self, name, cost_intervention, health_intervention, cost_base, health_base):
        ICER.__init__(self, name, cost_intervention, health_intervention, cost_base, health_base)

        delta_ave_cost = numpy.average(self._costIntervention) - numpy.average(self._costBase)
        delta_ave_health = numpy.average(self._healthIntervention) - numpy.average(self._healthBase)
        self._ICER = delta_ave_cost / delta_ave_health

    def get_CI(self, alpha, num_bootstrap_samples):

        pass

        #################### needs to be updated ##########################
        # calculate the CI of E(x)/E(y) with confidence 100-alpha
        delta = numpy.zeros(num_bootstrap_samples)

        # assert all the means should not be 0
        if self.y.mean() == 0:
            raise ValueError('invalid value of mean of y, the ratio is not computable')

        r_bar = self.x.mean() / self.y.mean()

        for i in range(num_bootstrap_samples):
            x_i = numpy.random.choice(self.x, size=len(self.x), replace=True)
            y_i = numpy.random.choice(self.y, size=len(self.y), replace=True)

            # assert all the means should not be 0
            if y_i.mean() == 0:
                raise ValueError('invalid value of mean of y, the ratio is not computable')
            ri_bar = x_i.mean() / y_i.mean()

            delta[i] = ri_bar - r_bar

        result = -numpy.percentile(delta, [100 - alpha / 2.0, alpha / 2.0]) + r_bar

    def get_PI(self, alpha):
        #################### needs to be updated ##########################
        pass


class NMB:
    def __init__(self, name, cost_intervention, health_intervention, cost_base, health_base):
        self._name = name
        self._costIntervention = numpy.array(cost_intervention)
        self._healthIntervention = numpy.array(health_intervention)
        self._costBase = numpy.array(cost_base)
        self._healthBase = numpy.array(health_base)

        self._delta_ave_cost = numpy.average(self._costIntervention) - numpy.average(self._costBase)
        self._delta_ave_health = numpy.average(self._healthIntervention) - numpy.average(self._healthBase)

    def get_mean_NMB(self, wtp):
        return wtp * self._delta_ave_health - self._delta_ave_cost

    def get_CI(self, wtp, alpha):
        pass

    def get_PI(self, wtp, alpha):
        pass


class NMBPaired(NMB):
    def __init__(self, name, cost_intervention, health_intervention, cost_base, health_base):
        self.__init__(name, cost_intervention, health_intervention, cost_base, health_base)

        self._deltaCost = self._costIntervention - self._costBase
        self._deltaHealth = self._healthIntervention - self._healthBase

    def get_CI(self, wtp, alpha):
        NMBsamples = wtp * self._deltaHealth - self._deltaCost
        stat = Stat.SummaryStat(self._name, NMBsamples)
        return stat.get_t_CI(alpha)

class NMBIndepedent(NMB):
    def __init__(self, name, cost_intervention, health_intervention, cost_base, health_base):
        self.__init__(name, cost_intervention, health_intervention, cost_base, health_base)
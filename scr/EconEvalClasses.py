import numpy as numpy
import scr.StatisticalClasses as Stat


class ICER():
    def __init__(self, name, cost_intervention, health_intervention, cost_base, health_base):

        self._name = name

        # convert input data to numpy.array if needed
        if type(cost_intervention) == list:
            self._cost_intervention = numpy.array(cost_intervention)
        if type(health_intervention) == list:
            self._health_intervention = numpy.array(health_intervention)
        if type(cost_base) == list:
            self._cost_base = numpy.array(cost_base)
        if type(health_base) == list:
            self._health_base = numpy.array(health_base)

        # calculate ICER
        delta_ave_cost = numpy.average(self._costIntervention) - numpy.average(self._costBase)
        delta_ave_health = numpy.average(self._healthIntervention) - numpy.average(self._healthBase)
        self._ICER = delta_ave_cost / delta_ave_health

    def get_ICER(self):
        return self._ICER

    def get_ICER_CI(self, alpha, num_bootstrap_samples):
        """
        :param alpha: significance level, a value from [0, 1]
        :param num_bootstrap_samples: number of bootstrap samples
        :return: confidence interval in the format of list [l, u]
        """
        # abstract method to be overridden in derived classes to process an event
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")

    def get_ICER_PI(self, alpha):
        """
        :param alpha: significance level, a value from [0, 1]
        :return: percentile interval in the format of list [l, u]
        """
        # abstract method to be overridden in derived classes to process an event
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")


class ICERpaired(ICER):

    def __init__(self, name, cost_intervention, health_intervention, cost_base, health_base):

        # initialize the base class
        ICER.__init__(name, cost_intervention, health_intervention, cost_base, health_base)

        # incremental observations
        self._deltaCost = self._cost_intervention - self._cost_base
        self._deltaHealth = self._health_intervention - self._health_base

        # create a ratio stat
        self._ratio_stat = Stat.RatioStatPaired.__init__(self, name, self._deltaCost, self._deltaHealth)

        # calculate ICER
        self._ICER = numpy.average(self._deltaCost) / numpy.average(self._deltaHealth)

    def get_ICER_CI(self, alpha, num_bootstrap_samples):

        # bootstrap algorithm
        ICERs = numpy.zeros(num_bootstrap_samples)
        for i in range(num_bootstrap_samples):
            d_cost = numpy.random.choice(self._deltaCost, size=len(self._deltaCost), replace=True)
            d_health = numpy.random.choice(self._deltaHealth, size=len(self._deltaHealth), replace=True)

            ave_dCost = numpy.average(d_cost)
            ave_dHealth = numpy.average(d_health)

            # assert all the means should not be 0
            if numpy.average(ave_dHealth) == 0:
                raise ValueError('invalid value of mean of y, the ratio is not computable')

            ICERs[i] = ave_dCost/ave_dHealth - self._ICER

        return self._ICER - numpy.percentile(ICERs, [100*(1 - alpha / 2.0), 100*alpha / 2.0])

    def get_ICER_PI(self, alpha):
        return self._ratio_stat.get_PI(alpha)


class ICERindp(ICER):

    def __init__(self, name, cost_intervention, health_intervention, cost_base, health_base):
        # initialize the base class
        ICER.__init__(name, cost_intervention, health_intervention, cost_base, health_base)


class NMB:
    def __init__(self, name, cost_intervention, health_intervention, cost_base, health_base):

        self._name = name

        # convert input data to numpy.array if needed
        if type(cost_intervention) == list:
            self._cost_intervention = numpy.array(cost_intervention)
        if type(health_intervention) == list:
            self._health_intervention = numpy.array(health_intervention)
        if type(cost_base) == list:
            self._cost_base = numpy.array(cost_base)
        if type(health_base) == list:
            self._health_base = numpy.array(health_base)

        self._delta_ave_cost = numpy.average(self._costIntervention) - numpy.average(self._costBase)
        self._delta_ave_health = numpy.average(self._healthIntervention) - numpy.average(self._healthBase)

    def get_mean(self, wtp):

        return wtp * self._delta_ave_health - self._delta_ave_cost

    def get_CI(self, wtp, alpha):
        """
        :param wtp: willingness-to-pay value
        :param alpha: significance level, a value from [0, 1]
        :return: confidence interval in the format of list [l, u]
        """
        # abstract method to be overridden in derived classes to process an event
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")

    def get_PI(self, wtp, alpha):
        """
        :param wtp: willingness-to-pay value
        :param alpha: significance level, a value from [0, 1]
        :return: percentile interval in the format of list [l, u]
        """
        # abstract method to be overridden in derived classes to process an event
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")


class NMBpaired(NMB):

    def __init__(self, name, cost_intervention, health_intervention, cost_base, health_base):
        NMB.__init__(name, cost_intervention, health_intervention, cost_base, health_base)

        self._deltaCost = self._costIntervention - self._costBase
        self._deltaHealth = self._healthIntervention - self._healthBase

    def get_CI(self, wtp, alpha):
        NMBsamples = wtp * self._deltaHealth - self._deltaCost
        stat = Stat.SummaryStat(self._name, NMBsamples)
        return stat.get_t_CI(alpha)

    def get_PI(self, wtp, alpha):
        NMBsamples = wtp * self._deltaHealth - self._deltaCost
        stat = Stat.SummaryStat(self._name, NMBsamples)
        return stat.get_PI(alpha)


class NMBindp(NMB):

    def __init__(self, name, cost_intervention, health_intervention, cost_base, health_base):
        NMB.__init__(name, cost_intervention, health_intervention, cost_base, health_base)

    def get_CI(self, wtp, alpha):
        pass

    def get_PI(self, wtp, alpha):
        pass
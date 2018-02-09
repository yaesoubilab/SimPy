import numpy as numpy
import scr.StatisticalClasses as Stat


class ComparativeEconMeasure():
    def __init__(self, name, cost_new, health_new, cost_base, health_base):
        """
        :param name: descrition
        :param cost_new: (list or numpy.array) cost data for the new strategy
        :param health_new: (list or numpy.array) health data for the new strategy
        :param cost_base: (list or numpy.array) cost data for teh base line
        :param health_base: (list or numpy.array) health data for the base line
        """

        self._name = name
        self._costNew = None        # cost data for the new strategy
        self._healthNew = None      # health data for the new strategy
        self._costBase = None      # cost data for teh base line
        self._healthBase = None    # health data for the base line

        # convert input data to numpy.array if needed
        if type(cost_new) == list:
            self._costNew = numpy.array(cost_new)
        if type(health_new) == list:
            self._healthNew = numpy.array(health_new)
        if type(cost_base) == list:
            self._costBase = numpy.array(cost_base)
        if type(health_base) == list:
            self._healthBase = numpy.array(health_base)

        # calculate the difference in average cost and health
        self._delta_ave_cost = numpy.average(self._costNew) - numpy.average(self._costBase)
        self._delta_ave_health = numpy.average(self._healthNew) - numpy.average(self._healthBase)


class ICER(ComparativeEconMeasure):
    def __init__(self, name, cost_new, health_new, cost_base, health_base):
        # initialize the base class
        ComparativeEconMeasure.__init__(self, name, cost_new, health_new, cost_base, health_base)
        # calcualte ICER
        self._ICER = self._delta_ave_cost/self._delta_ave_health

    def get_ICER(self):
        """ return ICER """
        return self._ICER

    def get_CI(self, alpha, num_bootstrap_samples):
        """
        :param alpha: significance level, a value from [0, 1]
        :param num_bootstrap_samples: number of bootstrap samples
        :return: confidence interval in the format of list [l, u]
        """
        # abstract method to be overridden in derived classes to process an event
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")

    def get_PI(self, alpha):
        """
        :param alpha: significance level, a value from [0, 1]
        :return: percentile interval in the format of list [l, u]
        """
        # abstract method to be overridden in derived classes to process an event
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")


class ICER_paired(ICER):

    def __init__(self, name, cost_new, health_new, cost_base, health_base):
        # initialize the base class
        ICER.__init__(name, cost_new, health_new, cost_base, health_base)

        # incremental observations
        self._deltaCost = self._costNew - self._costBase
        self._deltaHealth = self._healthNew - self._healthBase

        # create a ratio stat
        self._ratio_stat = Stat.RatioStatPaired(name, self._deltaCost, self._deltaHealth)

    def get_CI(self, alpha, num_bootstrap_samples):

        # bootstrap algorithm
        ICERs = numpy.zeros(num_bootstrap_samples)
        for i in range(num_bootstrap_samples):
            d_cost = numpy.random.choice(self._deltaCost, size=len(self._deltaCost), replace=True)
            d_health = numpy.random.choice(self._deltaHealth, size=len(self._deltaHealth), replace=True)

            ave_d_cost = numpy.average(d_cost)
            ave_d_health = numpy.average(d_health)

            # assert all the means should not be 0
            if numpy.average(ave_d_health) == 0:
                raise ValueError('invalid value of mean of y, the ratio is not computable')

            ICERs[i] = ave_d_cost/ave_d_health - self._ICER

        return self._ICER - numpy.percentile(ICERs, [100*(1 - alpha / 2.0), 100*alpha / 2.0])

    def get_PI(self, alpha):
        return self._ratio_stat.get_PI(alpha)


class ICER_indp(ICER):

    def __init__(self, name, cost_new, health_new, cost_base, health_base):
        # initialize the base class
        ICER.__init__(name, cost_new, health_new, cost_base, health_base)

    def get_CI(self, alpha, num_bootstrap_samples):
        pass

    def get_PI(self, alpha):
        pass


class NMB(ComparativeEconMeasure):
    def __init__(self, name, cost_new, health_new, cost_base, health_base):
        # initialize the base class
        ComparativeEconMeasure.__init__(self, name, cost_new, health_new, cost_base, health_base)

    def get_NMB(self, wtp):
        """
        :param wtp: willingness-to-pay
        :returns: the net monetary benefit at the provided willingness-to-pay value
        """
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


class NMB_paired(NMB):

    def __init__(self, name, cost_new, health_new, cost_base, health_base):
        NMB.__init__(name, cost_new, health_new, cost_base, health_base)

        # incremental observations
        self._deltaCost = self._costNew - self._costBase
        self._deltaHealth = self._healthNew - self._healthBase

    def get_CI(self, wtp, alpha):

        # create a sumary statistics
        stat = Stat.SummaryStat(self._name, wtp * self._deltaHealth - self._deltaCost)
        return stat.get_t_CI(alpha)

    def get_PI(self, wtp, alpha):

        # create a summary statistics
        stat = Stat.SummaryStat(self._name, wtp * self._deltaHealth - self._deltaCost)
        return stat.get_PI(alpha)


class NMB_indp(NMB):

    def __init__(self, name, cost_new, health_new, cost_base, health_base):
        NMB.__init__(name, cost_new, health_new, cost_base, health_base)

    def get_CI(self, wtp, alpha):
        pass

    def get_PI(self, wtp, alpha):
        pass
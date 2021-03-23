from math import pow


class _ExplorationRule:

    def get_epsilon(self, itr):
        pass


class _LearningRule:

    def get_step_size(self, itr):
        # goes to zero as itr goes to infinity
        return 0

    def get_forgetting_factor(self, itr):
        # goes to 1 as itr goes to infinity
        return 1/(1+self.get_step_size(itr))


class EpsilonGreedy(_ExplorationRule):
    # For selecting the greedy action with probability 1-epsilon.
    # epsilon_n = 1/n^beta, beta over (0.5, 1], n > 0

    def __init__(self, beta):
        self._beta = beta

    def __str__(self):
        return 'Beta{}'.format(self._beta)

    def get_epsilon(self, itr):

        return pow(itr, -self._beta)


class Harmonic(_LearningRule):
    # step_n = b / (b + n), for n >= 0 and b >= 1
    # (i is the iteration of the optimization algorithm)

    def __init__(self, b):
        self._b = b

    def __str__(self):
        return 'b{}'.format(self._b)

    def get_step_size(self, itr):
        return self._b / (self._b + itr - 1)

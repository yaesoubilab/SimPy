import matplotlib.pyplot as plt

class SimModel:
    def __init__(self):
        pass

    def get_obj_value(self, x):
        """ abstract method to return one realization of the objective function to optimize """
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")


class StepSize:
    def __init__(self, a):
        self._a = a

    def get_value(self, i):
        return self._a/(i+1)


class StochasticApproximation:
    def __init__(self, sim_model, step_size, derivative_step):
        self._simModel = sim_model
        self._stepSize = step_size
        self._derivativeStep = derivative_step
        self._xStar = None
        self._fStar = None

        self._is = []
        self._xs = []
        self._fs = []

    def minimize(self, max_itr, x0):

        x = x0
        f = self._simModel.get_obj_value(x)

        for i in range(max_itr):
            derivative = (self._simModel.get_obj_value(x+self._derivativeStep) - f)/self._derivativeStep
            x = x - self._stepSize.get_value(i)*derivative
            f = self._simModel.get_obj_value(x)

            self._is.append(i)
            self._xs.append(x)
            self._fs.append(f)

        self._xStar = x
        self._fStar = f

    def plot_fs(self):

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(self._is, self._fs)

        plt.xlabel('Iteration')
        plt.ylabel('Objective Function')
        plt.show()

    def plot_xs(self):

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(self._is, self._xs)

        plt.xlabel('Iteration')
        plt.ylabel('x')
        plt.show()
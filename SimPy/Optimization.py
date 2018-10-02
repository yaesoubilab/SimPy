import matplotlib.pyplot as plt
import numpy as np


class SimModel:
    # abstract class to be overridden by the simulation model to optimize
    def __init__(self):
        pass

    def get_obj_value(self, x):
        """ abstract method to return one realization of the objective function to optimize """
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")


class StepSize:
    # step size: a/i where a>0 and i>=0 is the iteration of the optimization algorithm
    def __init__(self, a):
        self._a = a

    def get_value(self, i):
        return self._a/(i+1)


class StochasticApproximation:
    # stochastic approximation algorithm

    def __init__(self, sim_model, derivative_step, step_size=StepSize(a=1)):
        """
        :param sim_model: the simulation model to optimize
        :param derivative_step: derivative step if calculating slopes
        :param step_size: the step size rule
        """
        self._simModel = sim_model
        self._stepSize = step_size
        self._derivativeStep = derivative_step
        self.xStar = None   # optimal x value
        self.fStar = None   # f(xStar)

        self.itr_i = []     # iteration indices
        self.itr_x = []     # x values over iterations
        self.itr_f = []     # f values over iterations
        self._deltax=[]

    def minimize(self, max_itr, x0):
        """
        :param max_itr: maximum iteration to terminate the algorithm
        :param x0: (numpy.array) starting point
        """

        x = x0
        f = self._simModel.get_obj_value(x)

        self.itr_i.append(0)
        self.itr_x.append(x)
        self.itr_f.append(f)

        #generate arrays to calculate derivative
        for k in range(0, len(x)):
            zero_s = np.zeros(len(x))
            zero_s[k] = self._derivativeStep
            self._deltax.append(zero_s)


        for i in range(1, max_itr):

            # estimate the derivative at x and y
            derivativevector=np.array([])
            for j in range(0,len(x)):
               derivative=(self._simModel.get_obj_value(x+self._deltax[j])-f)/self._derivativeStep
               derivativevector=np.append(derivativevector,[derivative])

            # find a new x
            x = x - self._stepSize.get_value(i)*derivativevector

            # evaluate the model at x
            f = self._simModel.get_obj_value(x)

            self.itr_i.append(i)
            self.itr_x.append(x)
            self.itr_f.append(f)

        # store the optimal x and optimal objective value
        self.xStar = x
        self.fStar = f

    def plot_fs(self, fStar=None):

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(self.itr_i, self.itr_f)

        plt.xlabel('Iteration')
        plt.ylabel('Objective Function')
        if fStar is not None:
            plt.axhline(y=fStar, linestyle='--', color='black', linewidth=1)
        plt.show()

    def plot_xs(self, xStar=None):

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(self.itr_i, self.itr_x)

        plt.xlabel('Iteration')
        plt.ylabel('x')
        if xStar is not None:
            plt.axhline(y=xStar, linestyle='--', color='black', linewidth=1)
        plt.show()


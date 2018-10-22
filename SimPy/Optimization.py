import matplotlib.pyplot as plt
import numpy as np


class SimModel:
    # abstract class to be overridden by the simulation model to optimize
    def __init__(self):
        pass

    def get_obj_value(self, x):
        """ abstract method to return one realization of the objective function to optimize """
        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")


class StepSize_a:
    # step size: a/i where a>0 and i>=0 is the iteration of the optimization algorithm
    def __init__(self, a):
        self._a = a

    def get_value(self, i):
        return self._a/(i+1)


class StepSize_e:
    # step size: c/i^(-0.25) where e>0 and i>=0 is the iteration of the optimization algorithm
    def __init__(self, e):
        self._e = e

    def get_value(self, i):
        return self._e * pow(i, -0.25)

class Penalty:
    #penalty for any of x<0. penalty size: p/(i+1) where p>0 and i>=0 is the iteration of the optimization algorithm
    def __init__(self, p):
        self._p = p

    def get_value(self,i):
        return self._p/(i+1)



class StochasticApproximation:
    # stochastic approximation algorithm

    def __init__(self, sim_model, derivative_step=StepSize_e(e=1), step_size=StepSize_a(a=1), penalty=Penalty(p=100)):
        """
        :param sim_model: the simulation model to optimize
        :param derivative_step: derivative step if calculating slopes
        :param step_size: the step size rule
        """
        self._simModel = sim_model
        self._stepSize = step_size
        self._derivativeStep = derivative_step
        self._penalty=penalty
        self.xStar = None   # optimal x value
        self.fStar = None   # optimal objective function 

        self.itr_i = []     # iteration indices
        self.itr_x = []     # x values over iterations
        self.itr_nDf = []    # normalized derivatives of f over iterations
        self.itr_f = []     # f values over iterations

    def minimize(self, max_itr, nLastItrsToAve, x0, penalty_or_not):
        """
        :param max_itr: maximum iteration to terminate the algorithm
        :param nLastItrsToAve: the number of last iterations to average as estimates of optimal x and f(x)
        :param x0: (list or numpy.array) starting point
        """

        # x0 has to be of type numpy.array
        if type(x0) is not np.ndarray:
            try:
                x0 = np.array(x0)
            except:
                raise ValueError('x0 should be a list or a numpy.array.')

        x = x0
        f = self._simModel.get_obj_value(x)

        # store information at iteration 0
        self.itr_i.append(0)
        self.itr_x.append(x)
        self.itr_f.append(f)
        # last n xs and fs to calculate optimal x and f
        self._nLastItrsToAve=nLastItrsToAve


        # iterations of stochastic approximation algorithm
        for itr in range(1, max_itr):

            # generate an array of numpy.array to calculate derivative
            # epsilon_matrix =
            #   [
            #       [ε, 0, 0, 0],
            #       [0, ε, 0, 0],
            #       [0, 0, ε, 0],
            #       [0, 0, 0, ε],
            #   ]

            epsilon_matrix = []
            for i in range(0, len(x)):
                # create an all zero array
                epsilon_array = np.zeros(len(x))
                # set ith element to epsilon (derivative step)
                epsilon_array[i] = self._derivativeStep.get_value(itr)
                # append the array to the epsilon matrix
                epsilon_matrix.append(epsilon_array)


            # estimate the derivative of f at current x
            Df = np.array([])
            for i in range(0, len(x)):
                # partial derivative of variable i
                partial_derivative_i = (self._simModel.get_obj_value(x + epsilon_matrix[i])-f)/self._derivativeStep.get_value(itr)
                # append this partial derivative
                Df = np.append(Df, [partial_derivative_i])

            # normalize derivative
            nDf = Df / np.linalg.norm(Df, 2)

            # find a new x: x_new = x - step_size*f'(x)/||f'(x)||
            x = x - self._stepSize.get_value(itr)*nDf
            # penalty if any of x < 0
            if penalty_or_not==1 and (x<0).any():
                x=abs(x)*self._penalty.get_value(itr)

            # evaluate the model at x
            f = self._simModel.get_obj_value(x)

            # store information at this iteration
            self.itr_i.append(itr)
            self.itr_x.append(x)
            self.itr_nDf.append(nDf)
            self.itr_f.append(f)

        #use the last nLastItrsToAve to calculate optimal x and optimal objective value
        length=len(self.itr_x)-self._nLastItrsToAve
        x_cal=self.itr_x[length:]
        f_cal=self.itr_f[length:]

        # store the optimal x and optimal objective value
        self.xStar = sum(x_cal)/len(x_cal)
        self.fStar = sum(f_cal)/len(x_cal) # should be the average of last f's

        # last derivative is not calculated
        self.itr_nDf.append(None)

    def plot_f_itr(self, f_star=None):
        """
        :param f_star: optimal f value if known
        :return: a plot of f values as the algorithm iterates
        """

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(self.itr_i, self.itr_f)

        plt.xlabel('Iteration')
        plt.ylabel('Objective Function')
        if f_star is not None:
            plt.axhline(y=f_star, linestyle='--', color='black', linewidth=1)
        plt.show()

    def plot_x_irs(self, x_star=None):
        """
        :param x_star: (list or numpy.array) optimal x value
        :return: plots of x_i in x as the the algorithm iterates
        """

        for i in range(len(x_star)):

            fig, ax = plt.subplots(figsize=(6, 5))

            # find x_i (ith dimension of x) over iterations
            x_i_itr = []
            for itr in range(len(self.itr_i)):
                x_i_itr.append(self.itr_x[itr][i])

            # plot
            ax.plot(self.itr_i, x_i_itr)
            plt.xlabel('Iteration')
            plt.ylabel('x'+str(i))
            if x_star is not None:
                plt.axhline(y=x_star[i], linestyle='--', color='black', linewidth=1)
            plt.show()

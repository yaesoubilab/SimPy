import matplotlib.pyplot as plt
import numpy as np


class SimModel:
    # abstract class to be overridden by the simulation model to optimize
    def __init__(self):
        pass

    def get_obj_value(self, x, seed_index=0):
        """
        abstract method to return one realization of the objective function to optimize
        :param x: the values of the variables
        :param seed_index: specify if need to use a different seed for this simulation replication
        """

        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")


class StepSize_GeneralizedHarmonic:
    # generalized harmonic(GH) stepsize
    # step_n = a0 * b / (b + i), for i >= 0, a0 > 0, and b >= 1
    # (i is the iteration of the optimization algorithm)

    def __init__(self, a0, b=1):
        self._a0 = a0
        self._b = b

    def get_value(self, itr):
        return self._a0 * self._b / (itr + self._b)


class StepSize_Df:
    # step size: c/i^(-0.25) where c>0 and i>=0 is the iteration of the optimization_figs algorithm
    def __init__(self, c0):
        self._c0 = c0

    def get_value(self, itr):
        return self._c0 * pow(itr + 1, -0.25)


class StochasticApproximation:
    # stochastic approximation algorithm

    def __init__(self, sim_model,
                 derivative_step=StepSize_Df(c0=1),
                 step_size=StepSize_GeneralizedHarmonic(a0=1)
                 ):
        """
        :param sim_model: the simulation model to optimize
        :param derivative_step: derivative step if calculating slopes
        :param step_size: the step size rule
        """
        self._simModel = sim_model
        self._stepSize = step_size
        self._derivativeStep = derivative_step
        self.xStar = None   # optimal x value
        self.fStar = None   # optimal objective function 

        self.itr_i = []     # iteration indices
        self.itr_x = []     # x values over iterations
        self.itr_nDf = []    # normalized derivatives of f over iterations
        self.itr_f = []     # f values over iterations
        self.itr_stepmove=[] # moving steps over iterations
        self.itr_stepdf=[] # harmonic step size over iterations

    def minimize(self, max_itr, n_last_itrs_to_ave, x0):
        """
        :param max_itr: maximum iteration to terminate the algorithm
        :param n_last_itrs_to_ave: the number of last iterations to average as estimates of optimal x and f(x)
        :param x0: (list or numpy.array) starting point
        """

        # x0 has to be of type numpy.array
        if type(x0) is not np.ndarray:
            try:
                x0 = np.array(x0)
            except:
                raise ValueError('x0 should be a list or a numpy.array.')

        x = x0

        # iterations of stochastic approximation algorithm
        for itr in range(0, max_itr):

            # generate an array of numpy.array to calculate derivative
            # epsilon_matrix =
            #   [
            #       [ε, 0, 0, 0],
            #       [0, ε, 0, 0],
            #       [0, 0, ε, 0],
            #       [0, 0, 0, ε],
            #   ]

            step_df = self._derivativeStep.get_value(itr)
            step_move = self._stepSize.get_value(itr)

            epsilon_matrix = []
            for i in range(0, len(x)):
                # create an all zero array
                epsilon_array = np.zeros(len(x))
                # set ith element to epsilon (derivative step)
                epsilon_array[i] = step_df
                # append the array to the epsilon matrix
                epsilon_matrix.append(epsilon_array)

            # estimate the derivative of f at current x
            Df = np.array([])
            for i in range(0, len(x)):
                # partial derivative of variable i
                partial_derivative_i \
                    = (self._simModel.get_obj_value(x + epsilon_matrix[i], seed_index=itr) -
                       self._simModel.get_obj_value(x - epsilon_matrix[i], seed_index=itr))/(2*step_df)

                # append this partial derivative
                Df = np.append(Df, [partial_derivative_i])

            # normalize derivative
            norm = np.linalg.norm(Df, 2)
            if norm == 0:
                nDf = Df
            else:
                nDf = Df / np.linalg.norm(Df, 2)

            # evaluate the model at x
            f = self._simModel.get_obj_value(x, seed_index=itr)

            # store information at this iteration
            self.itr_i.append(itr)
            self.itr_x.append(x)
            self.itr_f.append(f)
            self.itr_nDf.append(nDf)
            self.itr_stepdf.append(step_df)
            self.itr_stepmove.append(step_move)

            # find a new x: x_new = x - step_size*f'(x)/||f'(x)||
            x = x - step_move * nDf

        # use the last n iterations to calculate optimal x and optimal objective value
        length = len(self.itr_x) - n_last_itrs_to_ave
        x_cal = self.itr_x[length:]
        f_cal = self.itr_f[length:]

        # store the optimal x and optimal objective value
        self.xStar = sum(x_cal)/len(x_cal)
        self.fStar = sum(f_cal)/len(x_cal) # should be the average of last f's

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

    def plot_x_irs(self, x_stars=None):
        """
        :param x_star: (list or numpy.array) optimal x value
        :return: plots of x_i in x as the the algorithm iterates
        """

        n_vars = len(self.itr_x[0])
        f, axarr = plt.subplots(n_vars, 1, sharex=True)

        for i in range(n_vars):
            # find x_i (ith dimension of x) over iterations
            x_i_itr = []
            for itr in range(len(self.itr_i)):
                x_i_itr.append(self.itr_x[itr][i])

            axarr[i].plot(self.itr_i, x_i_itr)  # , color=ser.color, alpha=0.5)
            axarr[i].set(ylabel='x' + str(i))

            if x_stars is not None:
                axarr[i].axhline(y=x_stars[i], linestyle='--', color='black', linewidth=1)

        # label the x-axis of the last figure
        axarr[n_vars-1].set(xlabel='Iteration')

        plt.show()

    def plot_Df_irs(self):
        """
        :return: plot of how the derivative is changing over iterations.
        """
        n_vars=len(self.itr_nDf[0])
        fig, ax = plt.subplots(n_vars, 1, sharex=True)

        for i in range(n_vars):
            # find x_i (ith dimension of x) over iterations
            df_i_itr = []
            for itr in range(len(self.itr_i)):
                df_i_itr.append(self.itr_nDf[itr][i])

            ax[i].plot(self.itr_i, df_i_itr)  # , color=ser.color, alpha=0.5)
            ax[i].set(ylabel='nDf' + str(i))
        # label the x-axis of nDf
        ax[n_vars - 1].set(xlabel='Iteration')

        plt.show()

    def plot_step_move(self):
        """
        :return: plot of how the harmonic step size are changing over iterations.
        """
        plt.plot(self.itr_i,self.itr_stepmove)
        plt.ylabel('harmonic step size')
        plt.xlabel('iterations')
        plt.show()

    def plot_step_Df(self):
        """
        :return: plot of how the derivative step size are changing over iterations.
        """
        plt.plot(self.itr_i,self.itr_stepdf)
        plt.ylabel('derivative step size')
        plt.xlabel('iterations')
        plt.show()


def plot_step_size(a0s, bs, c0s, nItrs):
      
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlabel('Iteration')
    
    for a0 in a0s:
        for b in bs:
            GH = StepSize_GeneralizedHarmonic(a0=a0, b=b)
            
            y = []
            for itr in range(nItrs+1):
                y.append(GH.get_value(itr=itr))

            ax.plot(range(nItrs+1), y, label='GH: a0={}, b={}'.format(a0, b))

    for c0 in c0s:
        Df = StepSize_Df(c0=c0)
        y = []
        for itr in range(nItrs + 1):
            y.append(Df.get_value(itr=itr))

        ax.plot(range(nItrs + 1), y, linestyle='dashed', linewidth=3, label='Df: c0={}'.format(c0))
    
    fig.legend()
    fig.show()

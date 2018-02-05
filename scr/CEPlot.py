import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Strategy:
    def __init__(self, cost, effect):
        self.cost = cost
        self.effect = effect

class CEA:
    def __init__(self, strategies):
        """
        :param strategies: the list of strategies
        """
        self.cost = strategies.cost
        self.effect = strategies.effect

        self.n = len(self.cost)
        # default dominated status for strategies
        self.dominated = [False] * self.n

    def FindFrontier(self):
        # convert data to panda data frame
        # sort by cost, ascending
        data = pd.DataFrame({'Expected Cost': self.cost, 'Expected Utility': self.effect,\
                             'dominated': self.dominated, 'color': ['red']*self.n})
        data = data.sort_values('Expected Cost')
        # assign names to strategies, alphabeta may not enough
        data['Strategy'] = range(1, self.n + 1)

        # apply criteria 1
        for i in range(self.n):
            # get the points that dominated by point i and change their status
            data.loc[(data['Expected Cost'] > data['Expected Cost'][i])\
                     & (data['Expected Utility'] < data['Expected Utility'][i]), 'dominated'] = True

        data.loc[data['dominated'] == True, 'color'] = 'blue'

        # apply criteria 2
        data2 = data.loc[data['dominated'] == False]
        n2 = len(data2['Expected Cost'])

        for i in range(0, n2): # can't decide for first and last point
            for j in range(i+1, n2):
                # according to row numbers
                x1 = data2['Expected Utility'].iloc[i]
                y1 = data2['Expected Cost'].iloc[i]

                x2 = data2['Expected Utility'].iloc[j]
                y2 = data2['Expected Cost'].iloc[j]

                v1 = np.array([x2-x1, y2-y1])

                if ((data2['Expected Utility'] > x1) & (data2['Expected Utility'] < x2)).any() == False: continue
                else:
                    inner_points = data2.loc[(data2['Expected Utility'] > x1) & (data2['Expected Utility'] < x2)]
                    v2_x = inner_points['Expected Utility'] - x1
                    v2_y = inner_points['Expected Cost']-y1
                    cross_product = v1[0] * np.array(v2_y) - v1[1] * np.array(v2_x)
                    # if cross_product > 0 the point is above the line
                    # (because the point are sorted vertically)
                    # ref: https://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line
                    ind_remove = inner_points[cross_product > 0].index
                    data.loc[list(ind_remove), 'dominated'] = True
                    data.loc[list(ind_remove), 'color'] = 'blue'

        # plots
        linedat = data.loc[data["dominated"] == False].sort_values('Expected Utility')
        plt.scatter(data['Expected Utility'], data['Expected Cost'], c=list(data['color']), alpha=0.6)
        plt.plot(linedat['Expected Utility'], linedat['Expected Cost'], c='red', alpha=0.6)
        plt.axhline(y=0, color='k')
        plt.axvline(x=0, color='k')
        plt.xlabel('Expected Utility')
        plt.ylabel('Expected Cost')
        plt.show()

        return data



    def BuildCETable(self):
        data = self.FindFrontier()
        data['Expected Incremental Cost'] = "-"
        data['Expected Incremental Utility'] = "-"
        data['ICER'] = "Dominated"
        not_dominated_points = data.loc[data["dominated"] == False].sort_values('Expected Cost')

        n_not_dominated = not_dominated_points.shape[0]


        incre_cost = []
        incre_utility = []
        ICER = []
        for i in range(1, n_not_dominated):
            temp_num = not_dominated_points["Expected Cost"].iloc[i]-not_dominated_points["Expected Cost"].iloc[i-1]
            incre_cost = np.append(incre_cost, temp_num)

            temp_den = not_dominated_points["Expected Utility"].iloc[i]-not_dominated_points["Expected Utility"].iloc[i-1]
            if temp_den == 0:
                raise ValueError('invalid value of Expected Incremental Utility, the ratio is not computable')
            incre_utility = np.append(incre_utility, temp_den)

            ICER = np.append(ICER, temp_num/temp_den)

        ind_change = not_dominated_points.index[1:]
        data.loc[ind_change, 'Expected Incremental Cost'] = incre_cost
        data.loc[ind_change, 'Expected Incremental Utility'] = incre_utility
        data.loc[ind_change, 'ICER'] = ICER
        data.loc[not_dominated_points.index[0], 'ICER'] = '-'

        return data[['Strategy', 'Expected Cost', 'Expected Utility', 'Expected Incremental Cost', 'Expected Incremental Utility',\
            'ICER']]




np.random.seed(573)
cost = np.random.normal(0, 5, 20)
effect = np.random.normal(0, 5, 20)
s = Strategy(cost,effect) # unsorted

myCEA = CEA(s)
data = myCEA.FindFrontier()
table = myCEA.BuildCETable()
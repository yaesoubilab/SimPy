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
        data = pd.DataFrame({'cost': self.cost, 'effect': self.effect,\
                             'dominated': self.dominated, 'color': ['red']*self.n})
        data = data.sort_values('cost')

        # apply criteria 1
        for i in range(self.n):
            # get the points that dominated by point i and change their status
            data.loc[(data['cost'] > data['cost'][i]) & (data['effect'] < data['effect'][i]), 'dominated'] = True

        data.loc[data['dominated'] == True, 'color'] = 'blue'

        # apply criteria 2
        data2 = data.loc[data['dominated'] == False]
        n2 = len(data2['cost'])

        for i in range(0, n2): # can't decide for first and last point
            for j in range(i+1, n2):
                # according to row numbers
                x1 = data2['effect'].iloc[i]
                y1 = data2['cost'].iloc[i]

                x2 = data2['effect'].iloc[j]
                y2 = data2['cost'].iloc[j]

                v1 = np.array([x2-x1, y2-y1])

                if ((data2['effect'] > x1) & (data2['effect'] < x2)).any() == False: continue
                else:
                    inner_points = data2.loc[(data2['effect'] > x1) & (data2['effect'] < x2)]
                    v2_x = inner_points['effect'] - x1
                    v2_y = inner_points['cost']-y1
                    cross_product = v1[0] * np.array(v2_y) - v1[1] * np.array(v2_x)
                    # if cross_product > 0 the point is above the line
                    # (because the point are sorted vertically)
                    # ref: https://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line
                    ind_remove = inner_points[cross_product > 0].index
                    data.loc[list(ind_remove), 'dominated'] = True
                    data.loc[list(ind_remove), 'color'] = 'blue'

        plt.scatter(data['effect'], data['cost'], c=list(data['color']), alpha=0.6)
        plt.xlabel('effect')
        plt.ylabel('cost')
        plt.show()

        return data



    def BuildCETable(self):
        pass

np.random.seed(573)
cost = np.random.normal(0, 5, 50)
effect = np.random.normal(0, 5, 50)

s = Strategy(cost,effect)

myCEA = CEA(s)
data = myCEA.FindFrontier()

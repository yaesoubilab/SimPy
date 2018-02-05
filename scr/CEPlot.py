import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Strategy:
    def __init__(self, name, cost, effect):
        self.name = name
        self.cost = cost
        self.effect = effect
        self.ifDominated = False

class CEA:
    def __init__(self, strategies):
        """
        :param strategies: the list of strategies
        """
        self.strategies = pd.DataFrame(index=range(len(strategies)), \
                            columns=['Name', 'Expected Cost', 'Expected Utility', \
                                                                 'dominated', 'color'])

        for i in range(len(strategies)):
            self.strategies.loc[i, 'Name'] = strategies[i].name
            self.strategies.loc[i, 'Expected Cost'] = strategies[i].cost
            self.strategies.loc[i, 'Expected Utility'] = strategies[i].effect
            self.strategies.loc[i, 'dominated'] = strategies[i].ifDominated
            self.strategies.loc[i, 'color'] = "k"  # not dominated black, dominated blue

        self.n = len(strategies)

        # seems no need to define following attributes?

        # self.cost = strategies.cost
        # self.effect = strategies.effect
        # self.dominated = [False] * self.n

    def FindFrontier(self):
        # sort strategies by cost, ascending
        # operate on local variable data rather than self attribute
        data = self.strategies.sort_values('Expected Cost')

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

        # update strategies
        self.strategies = data

        return data.loc[data['dominated']==False]

    def ShowCEPlane(self):
        # plots
        # operate on local variable data rather than self attribute
        data = self.strategies

        # re-sorted according to Effect to draw line
        linedat = data.loc[data["dominated"] == False].sort_values('Expected Utility')

        plt.scatter(data['Expected Utility'], data['Expected Cost'], c=list(data['color']))
        plt.plot(linedat['Expected Utility'], linedat['Expected Cost'], c='k')
        plt.axhline(y=0, color='k',linewidth=0.5)
        plt.axvline(x=0, color='k',linewidth=0.5)
        plt.xlabel('Expected Utility')
        plt.ylabel('Expected Cost')
        plt.show()


    def BuildCETable(self):
        data = self.strategies
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

        return data[['Name', 'Expected Cost', 'Expected Utility', 'Expected Incremental Cost', 'Expected Incremental Utility',\
            'ICER']]


np.random.seed(573)

s0 = Strategy('s0',0, 0)
s1 = Strategy("s1",np.random.normal(0, 5),np.random.normal(0, 5))
s2 = Strategy("s2",np.random.normal(0, 5),np.random.normal(0, 5))
s3 = Strategy("s3",np.random.normal(0, 5),np.random.normal(0, 5))
s4 = Strategy("s4",np.random.normal(0, 5),np.random.normal(0, 5))
s5 = Strategy("s5",np.random.normal(0, 5),np.random.normal(0, 5))
s6 = Strategy("s6",np.random.normal(0, 5),np.random.normal(0, 5))
s7 = Strategy("s7",np.random.normal(0, 5),np.random.normal(0, 5))
s8 = Strategy("s8",np.random.normal(0, 5),np.random.normal(0, 5))
s9 = Strategy("s9",np.random.normal(0, 5),np.random.normal(0, 5))
s10 = Strategy("s10",np.random.normal(0, 5),np.random.normal(0, 5))



strategies = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]
myCEA = CEA(strategies)

# frontier results
frontiers = myCEA.FindFrontier()

# updated strategies
myCEA.strategies

# plot
myCEA.ShowCEPlane()


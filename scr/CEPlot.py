import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd


class Strategy:
    def __init__(self, name, cost, effect):
        """        
        :param name: name of the strategy
        :param cost: list o numpy.array  
        :param effect: list or numpy.array
        """
        self.name = name
        self.cost = cost
        self.effect = effect
        self.aveCost = np.average(self.cost)
        self.aveEffect = np.average(self.effect)
        self.ifDominated = False


class CEA:
    def __init__(self, strategies):
        """
        :param strategies: the list of strategies
        """
        self._n = len(strategies)

        # store all observations for all strategies
        self._dataCloud = strategies

        # data frame to contain the strategies on the frontier
        self._strategiesOnFrontier = pd.DataFrame()

        # data frame for all strategies' expected outcomes
        self._dfStrategies = pd.DataFrame(
            index=range(self._n),
            columns=['Name', 'E[Cost]', 'E[Effect]', 'Dominated', 'Color'])

        for i in range(len(strategies)):
            self._dfStrategies.loc[i, 'Name'] = strategies[i].name
            self._dfStrategies.loc[i, 'E[Cost]'] = strategies[i].aveCost
            self._dfStrategies.loc[i, 'E[Effect]'] = strategies[i].aveEffect
            self._dfStrategies.loc[i, 'Dominated'] = strategies[i].ifDominated
            self._dfStrategies.loc[i, 'Color'] = "k"  # not Dominated black, Dominated blue

        # find the CE frontier
        self.__find_frontier()

    def get_frontier(self):
        return self._strategiesOnFrontier

    def __find_frontier(self):
        # sort strategies by cost, ascending
        # operate on local variable data rather than self attribute
        df1 = self._dfStrategies.sort_values('E[Cost]')

        # apply criteria 1
        for i in range(self._n):
            # strategies with higher cost and lower Effect are dominated
            df1.loc[
                (df1['E[Cost]'] > df1['E[Cost]'][i]) &
                (df1['E[Effect]'] <= df1['E[Effect]'][i]),
                'Dominated'] = True
        # change the color of dominated strategies to blue
        df1.loc[df1['Dominated'] == True, 'Color'] = 'blue'

        # apply criteria 2
        # select all non-dominated strategies
        df2 = df1.loc[df1['Dominated']==False]
        n2 = len(df2['E[Cost]'])

        for i in range(0, n2): # can't decide for first and last point
            for j in range(i+1, n2):
                # cost and effect of strategy i
                effect_i = df2['E[Effect]'].iloc[i]
                cost_i = df2['E[Cost]'].iloc[i]
                # cost and effect of strategy j
                effect_j = df2['E[Effect]'].iloc[j]
                cost_j = df2['E[Cost]'].iloc[j]
                # vector connecting strategy i to j
                v_i_to_j = np.array([effect_j-effect_i, cost_j-cost_i])

                # if the effect of no strategy is between the effects of strategies i and j
                if not ((df2['E[Effect]'] > effect_i) & (df2['E[Effect]'] < effect_j)).any():
                    continue    # to the next j
                else:
                    # get all the strategies with effect between strategies i and j
                    inner_points = df2.loc[(df2['E[Effect]'] > effect_i) & (df2['E[Effect]'] < effect_j)]
                    # difference in effect of inner points and strategy i
                    v2_x = inner_points['E[Effect]'] - effect_i
                    # difference in cost of inner points and strategy i
                    v2_y = inner_points['E[Cost]']-cost_i

                    # cross products of vector i to j and the vectors i to all inner points
                    cross_product = v_i_to_j[0] * np.array(v2_y) - v_i_to_j[1] * np.array(v2_x)

                    # if cross_product > 0 the point is above the line
                    # (because the point are sorted vertically)
                    # ref: https://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line
                    dominated_index = inner_points[cross_product > 0].index
                    df1.loc[list(dominated_index), 'Dominated'] = True
                    df1.loc[list(dominated_index), 'Color'] = 'blue'

        # update strategies
        self._dfStrategies = df1
        self._strategiesOnFrontier = df1.loc[df1['Dominated'] == False, ['Name', 'E[Cost]', 'E[Effect]']]

    def show_CE_plane(self, x_label, y_label, show_names=False, show_clouds=False):
        """
        :param x_label: (string) x-axis label
        :param y_label: (string) y-axis label
        :param show_names: logical, show strategy names
        :param show_clouds: logical, show true sample observation of strategies
        """
        # plots
        # operate on local variable data rather than self attribute
        data = self._dfStrategies

        # re-sorted according to Effect to draw line
        linedat = data.loc[data["Dominated"] == False].sort_values('E[Effect]')

        # show observation clouds for strategies
        if show_clouds:
            for strategy_i, color in zip(self._dataCloud, cm.rainbow(np.linspace(0, 1, self._n))):
                x_values = strategy_i.effect
                y_values = strategy_i.cost
                # plot
                plt.scatter(x_values, y_values, c=color, alpha=0.5, s=50)

        plt.scatter(data['E[Effect]'], data['E[Cost]'], c=list(data['Color']), s=50)
        plt.plot(linedat['E[Effect]'], linedat['E[Cost]'], c='k')
        plt.axhline(y=0, c='k',linewidth=0.5)
        plt.axvline(x=0, c='k',linewidth=0.5)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # show names of strategies
        if show_names:
            for label, x, y in zip(data['Name'], data['E[Effect]'], data['E[Cost]']):
                plt.annotate(
                    label,
                    xy=(x, y), xytext=(-20, 20),
                    textcoords='offset points', ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.show()


    def BuildCETable(self, digits=5):
        """
        :param digits: specify how many digits return in the table, 5 by default
        :return: output csv file called "CETable.csv" in local environment
        """
        data = self._dfStrategies
        data['Expected Incremental Cost'] = "-"
        data['Expected Incremental Effect'] = "-"
        data['ICER'] = "Dominated"
        not_Dominated_points = data.loc[data["Dominated"] == False].sort_values('E[Cost]')

        n_not_Dominated = not_Dominated_points.shape[0]


        incre_cost = []
        incre_Effect = []
        ICER = []
        for i in range(1, n_not_Dominated):
            temp_num = not_Dominated_points["E[Cost]"].iloc[i]-not_Dominated_points["E[Cost]"].iloc[i-1]
            incre_cost = np.append(incre_cost, temp_num)

            temp_den = not_Dominated_points["E[Effect]"].iloc[i]-not_Dominated_points["E[Effect]"].iloc[i-1]
            if temp_den == 0:
                raise ValueError('invalid value of Expected Incremental Effect, the ratio is not computable')
            incre_Effect = np.append(incre_Effect, temp_den)

            ICER = np.append(ICER, temp_num/temp_den)

        ind_change = not_Dominated_points.index[1:]
        data.loc[ind_change, 'Expected Incremental Cost'] = incre_cost.astype(float).round(digits)
        data.loc[ind_change, 'Expected Incremental Effect'] = incre_Effect.astype(float).round(digits)
        data.loc[ind_change, 'ICER'] = ICER.astype(float).round(digits)
        data.loc[not_Dominated_points.index[0], 'ICER'] = '-'

        output = data[['Name', 'E[Cost]', 'E[Effect]', 'Expected Incremental Cost', 'Expected Incremental Effect',\
            'ICER']]

        # set digits number to display
        output.loc[:, 'E[Cost]'] = output['E[Cost]'].astype(float).round(digits)
        output.loc[:, 'E[Effect]'] = output['E[Effect]'].astype(float).round(digits)

        # write csv
        output.to_csv("CETable.csv", encoding='utf-8', index=False)

        return output


np.random.seed(573)

s_center = np.random.normal(0, 5, (10, 2))


s0 = Strategy('s0',0, 0)
s1 = Strategy("s1",s_center[0,0]+np.random.normal(0, 0.5, 10), s_center[0,1]+np.random.normal(0, 0.5, 10))
s2 = Strategy("s2",s_center[1,0]+np.random.normal(0, 0.5, 10), s_center[1,1]+np.random.normal(0, 0.5, 10))
s3 = Strategy("s3",s_center[2,0]+np.random.normal(0, 0.5, 10), s_center[2,1]+np.random.normal(0, 0.5, 10))
s4 = Strategy("s4",s_center[3,0]+np.random.normal(0, 0.5, 10), s_center[3,1]+np.random.normal(0, 0.5, 10))
s5 = Strategy("s5",s_center[4,0]+np.random.normal(0, 0.5, 10), s_center[4,1]+np.random.normal(0, 0.5, 10))
s6 = Strategy("s6",s_center[5,0]+np.random.normal(0, 0.5, 10), s_center[5,1]+np.random.normal(0, 0.5, 10))
s7 = Strategy("s7",s_center[6,0]+np.random.normal(0, 0.5, 10), s_center[6,1]+np.random.normal(0, 0.5, 10))
s8 = Strategy("s8",s_center[7,0]+np.random.normal(0, 0.5, 10), s_center[7,1]+np.random.normal(0, 0.5, 10))
s9 = Strategy("s9",s_center[8,0]+np.random.normal(0, 0.5, 10), s_center[8,1]+np.random.normal(0, 0.5, 10))
s10 = Strategy("s10",s_center[9,0]+np.random.normal(0, 0.5, 10), s_center[9,1]+np.random.normal(0, 0.5, 10))


myCEA = CEA([s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10])

# frontier results
myCEA.get_frontier()

# updated strategies
myCEA._dfStrategies

# plot with label and sample cloud
myCEA.show_CE_plane('E[Effect]','E[Cost]', True, True)

# table
print(myCEA.BuildCETable(digits=2))
from Model import *

N = 1000
ACTION_COST = 3
COST_SIGMA = 0

multiModel = MultiModel(decision_rule=AlwaysOff(),
                        cost_sigma=COST_SIGMA, action_cost=ACTION_COST)
multiModel.simulate(n=N)
print('Always off: ', multiModel.statCost.get_formatted_mean_and_interval(
    interval_type='c', sig_digits=4))

multiModel = MultiModel(decision_rule=AlwaysOn(),
                        cost_sigma=COST_SIGMA, action_cost=ACTION_COST)
multiModel.simulate(n=N)
print('Always on: ', multiModel.statCost.get_formatted_mean_and_interval(
    interval_type='c', sig_digits=4))

multiModel = MultiModel(decision_rule=Myopic(),
                        cost_sigma=COST_SIGMA, action_cost=ACTION_COST)
multiModel.simulate(n=N)
print('Myopic: ', multiModel.statCost.get_formatted_mean_and_interval(
    interval_type='c', sig_digits=4))

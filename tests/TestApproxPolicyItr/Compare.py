from Model import *
from SimPy.Optimization.ApproxPolicyIteration import GreedyApproxDecisionMaker


N = 1000
ACTION_COST = 50
COST_SIGMA = 0

# always-off strategy
multiModel = MultiModel(decision_rule=AlwaysOff(),
                        cost_sigma=COST_SIGMA, action_cost=ACTION_COST)
multiModel.simulate(n=N)
print('Always off: ', multiModel.statCost.get_formatted_mean_and_interval(
    interval_type='c', sig_digits=4))

# always-on strategy
multiModel = MultiModel(decision_rule=AlwaysOn(),
                        cost_sigma=COST_SIGMA, action_cost=ACTION_COST)
multiModel.simulate(n=N)
print('Always on: ', multiModel.statCost.get_formatted_mean_and_interval(
    interval_type='c', sig_digits=4))

# myopic strategy
multiModel = MultiModel(decision_rule=Myopic(),
                        cost_sigma=COST_SIGMA, action_cost=ACTION_COST)
multiModel.simulate(n=N)
print('Myopic: ', multiModel.statCost.get_formatted_mean_and_interval(
    interval_type='c', sig_digits=4))

# dynamic strategy
approx_decision_maker = GreedyApproxDecisionMaker(num_of_actions=2, q_functions_csv_file='q-functions.csv')
multiModel = MultiModel(decision_rule=Dynamic(approx_decision_maker=approx_decision_maker),
                        cost_sigma=COST_SIGMA, action_cost=ACTION_COST)
multiModel.simulate(n=N)
print('Dynamic: ', multiModel.statCost.get_formatted_mean_and_interval(
    interval_type='c', sig_digits=4))

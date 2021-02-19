from TestApproxPolicyItr import *

multiModel = MultiModel(decision_rule=AlwaysOff(), reward_sigma=0.5, action_cost=1)
multiModel.simulate(n=100)
print('\nAlways off: ', multiModel.statRewards.get_formatted_mean_and_interval(interval_type='c', sig_digits=4))

multiModel = MultiModel(decision_rule=AlwaysOn(), reward_sigma=0.5, action_cost=1)
multiModel.simulate(n=100)
print('\nAlways on: ', multiModel.statRewards.get_formatted_mean_and_interval(interval_type='c', sig_digits=4))
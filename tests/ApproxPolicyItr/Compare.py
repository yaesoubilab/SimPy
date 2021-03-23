from Model import *
from SimPy.Optimization.ApproxPolicyIteration import GreedyApproxDecisionMaker

N = 1000
ACTION_COST = 50
COST_SIGMA = 0
Q_FUNC_DEGREE = 2
Q_FUNC_CSVFILE = 'q-functions.csv' # or optimal_q_functions.csv


def compare(q_function_degree=None, q_functions_csvfile=None):

    if q_function_degree is None:
        q_function_degree = Q_FUNC_DEGREE
    if q_functions_csvfile is None:
        q_functions_csvfile = Q_FUNC_CSVFILE

    # always-off strategy
    multi_model = MultiModel(decision_rule=AlwaysOff(),
                             cost_sigma=COST_SIGMA, action_cost=ACTION_COST)
    multi_model.simulate(n=N)
    print('Always off: ', multi_model.statCost.get_formatted_mean_and_interval(
        interval_type='c', sig_digits=4))

    # always-on strategy
    multi_model = MultiModel(decision_rule=AlwaysOn(),
                             cost_sigma=COST_SIGMA, action_cost=ACTION_COST)
    multi_model.simulate(n=N)
    print('Always on: ', multi_model.statCost.get_formatted_mean_and_interval(
        interval_type='c', sig_digits=4))

    # myopic strategy
    multi_model = MultiModel(decision_rule=Myopic(),
                             cost_sigma=COST_SIGMA, action_cost=ACTION_COST)
    multi_model.simulate(n=N)
    print('Myopic: ', multi_model.statCost.get_formatted_mean_and_interval(
        interval_type='c', sig_digits=4))

    # dynamic strategy
    approx_decision_maker = GreedyApproxDecisionMaker(num_of_actions=1,
                                                      q_function_degree=q_function_degree,
                                                      q_functions_csv_file=q_functions_csvfile)
    multi_model = MultiModel(decision_rule=Dynamic(approx_decision_maker=approx_decision_maker),
                             cost_sigma=COST_SIGMA, action_cost=ACTION_COST)
    multi_model.simulate(n=N)
    print('Dynamic: ', multi_model.statCost.get_formatted_mean_and_interval(
        interval_type='c', sig_digits=4))


if __name__ == '__main__':
    compare()

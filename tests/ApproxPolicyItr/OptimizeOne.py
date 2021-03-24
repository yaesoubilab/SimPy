import  Compare as C
from Model import Model
from SimPy.Optimization.ApproxPolicyIteration import ApproximatePolicyIteration
from SimPy.Optimization.LearningAndExplorationRules import *


N_ITRS = 5000
B = 50
BETA = 0.5
Q_FUNC_DEGREE = 2
L2_PENALTY = 0.01

sim_model = Model(cost_sigma=C.COST_SIGMA,
                  action_cost=C.ACTION_COST)

api = ApproximatePolicyIteration(sim_model=sim_model,
                                 num_of_actions=1,
                                 learning_rule=Harmonic(b=B),
                                 exploration_rule=EpsilonGreedy(beta=BETA),
                                 q_function_degree=Q_FUNC_DEGREE,
                                 l2_penalty=L2_PENALTY)

api.minimize(n_iterations=N_ITRS)

api.export_results(csv_file='iterations.csv')
api.plot_iterations(moving_ave_window=int(N_ITRS / 20), fig_size=(5, 6),
                    n_last_iterations_to_ave=N_ITRS/5)

C.compare(q_function_degree=Q_FUNC_DEGREE)


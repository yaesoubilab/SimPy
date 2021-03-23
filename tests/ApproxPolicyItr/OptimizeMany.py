from Model import Model
from SimPy.Optimization.ApproxPolicyIteration import MultiApproximatePolicyIteration
from SimPy.Optimization.LearningAndExplorationRules import *


ACTION_COST = 3
COST_SIGMA = 0

N_ITRS = 1000
IF_PARALLEL = False
B = [25, 50]
BETA = [0.4, 0.5]
Q_FUNC_DEGREES = [2]
L2_PENALTIES = [0.01]

# build models
n = len(B) * len(BETA) * len(Q_FUNC_DEGREES) * len(L2_PENALTIES)
models = []
for i in range(n):
    models.append(Model(cost_sigma=COST_SIGMA, action_cost=ACTION_COST))

optimizer = MultiApproximatePolicyIteration(
    sim_models=models,
    num_of_actions=1,
    discount_factor=1/(1+0.03),
    learning_rules=[Harmonic(b=b) for b in B],
    exploration_rules=[EpsilonGreedy(beta=beta) for beta in BETA],
    q_function_degrees=Q_FUNC_DEGREES,
    l2_penalties=L2_PENALTIES,
    q_functions_folder='q-functions')

optimizer.minimize_all(n_iterations=N_ITRS, if_parallel=IF_PARALLEL,
                       folder_to_save_iterations='optimization_results')

optimizer.plot_iterations(moving_ave_window=int(N_ITRS / 20), fig_size=(5, 6),
                          folder_to_save_figures='optimization_figures')

# compare(q_function_degrees=Q_FUNC_DEGREES)


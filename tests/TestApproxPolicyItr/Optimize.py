from SimPy.Optimization.ApproxPolicyIteration import ApproximatePolicyIteration
from SimPy.Optimization.LearningRules import *
from Model import Model

N = 1000
ACTION_COST = 3
COST_SIGMA = 0
B = 10
BETA = 0.7

sim_model = Model(cost_sigma=COST_SIGMA,
                  action_cost=ACTION_COST)

api = ApproximatePolicyIteration(sim_model=sim_model,
                                 num_of_actions=1,
                                 learning_rule=Harmonic(b=B),
                                 exploration_rule=EpsilonGreedy(beta=0.7),
                                 discount_factor=1/(1+0.03),
                                 q_function_degree=2,
                                 l2_penalty=0.01)

api.optimize(n_iterations=100)



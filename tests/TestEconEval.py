import scr.EconEvalClasses as EconEval
import random

cost_intervention = [250, 325, 385]
health_intervention = [1, 0.9, .8]
cost_base = [20, 30, 40]
health_base = [0.7, 0.6, 0.5]

random.seed(1)
# ICER calculation assuming paired observations
ICER = EconEval.ICER_paired('test', cost_intervention, health_intervention, cost_base, health_base)
print(ICER.get_ICER(), ICER.get_CI(0.05, 1000), ICER.get_PI(0.05,));
# the data to get percentile is like [  766.66666667,   983.33333333,  1150.]

# ICER calculation assuming independent observations
ICER = EconEval.ICER_indp('test', cost_intervention, health_intervention, cost_base, health_base)
print(ICER.get_ICER(), ICER.get_CI(0.05, 1000), ICER.get_PI(0.05,));
# the data to get percentile is like [ 3650. ,   712.5,  3650. ]
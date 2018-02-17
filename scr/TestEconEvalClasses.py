from scr import EconEvalClasses as ce
import numpy as np


np.random.seed(573)

s0 = ce.Strategy("s1", 1+np.random.normal(0, 0.5, 10), 2+np.random.normal(0, 0.5, 10))
s1 = ce.Strategy("s2", 2+np.random.normal(0, 0.5, 10), 4+np.random.normal(0, 0.5, 10))

# test paired case
test_pair = ce.ICER_paired("test_pair", s1.costObs, s1.effectObs, s0.costObs,s0.effectObs)
print(test_pair.get_ICER())
print(test_pair.get_PI(0.05))
print(test_pair.get_CI(0.05,1000))

# test independent case
test_indp = ce.ICER_indp("test_pair", s1.costObs, s1.effectObs, s0.costObs,s0.effectObs)
print(test_indp.get_ICER())
print(test_indp.get_PI(0.05))
print(test_indp.get_CI(0.05,1000))

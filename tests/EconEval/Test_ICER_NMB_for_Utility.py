import SimPy.EconEval as EconEval
import numpy as np

np.random.seed(573)

cost_base = np.random.normal(loc=10000, scale=100, size=1000)
effect_base = np.random.normal(loc=1, scale=.1, size=1000)
cost_intervention = np.random.normal(loc=20000, scale=200, size=1000)
effect_intervention = np.random.normal(loc=2, scale=.2, size=1000)

print('')

# ICER calculation assuming paired observations
ICER_paired = EconEval.ICER_Paired('Testing paired ICER',
                                   cost_intervention, effect_intervention, cost_base, effect_base)
print('Paired ICER (confidence and prediction interval): ',
      ICER_paired.get_ICER(),
      ICER_paired.get_CI(0.05, 1000),
      ICER_paired.get_PI(0.05, ))

# ICER calculation assuming independent observations
ICER_indp = EconEval.ICER_Indp('Testing independent ICER',
                               cost_intervention, effect_intervention, cost_base, effect_base)
print('Independent ICER (confidence and prediction interval): ',
      ICER_indp.get_ICER(),
      ICER_indp.get_CI(0.05, 1000),
      ICER_indp.get_PI(0.05, ))

# try NMB
NMB_paired = EconEval.INMB_Paired("Testing paired NMB",
                                  cost_intervention, effect_intervention, cost_base, effect_base)
print('Paired NMB (confidence and prediction interval): ',
      NMB_paired.get_INMB(wtp=10000),
      NMB_paired.get_CI(wtp=10000, alpha=.05),
      NMB_paired.get_PI(wtp=10000, alpha=.05))

NMB_indp = EconEval.INMB_Indp("Testing independent NMB",
                              cost_intervention, effect_intervention, cost_base, effect_base)
print('Independent NMB (confidence and prediction interval): ',
      NMB_indp.get_INMB(wtp=10000),
      NMB_indp.get_CI(wtp=10000, alpha=.05),
      NMB_indp.get_PI(wtp=10000, alpha=.05))

print('')

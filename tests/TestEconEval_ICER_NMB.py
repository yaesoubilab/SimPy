import scr.EconEvalClasses as EconEval
import random

cost_intervention = [250, 325, 385]
health_intervention = [1, 0.9, .8]
cost_base = [20, 30, 40]
health_base = [0.7, 0.6, 0.5]

random.seed(1)
# ICER calculation assuming paired observations
ICER_paired = EconEval.ICER_paired('Testing paired ICER', cost_intervention, health_intervention, cost_base, health_base)
print('Paired ICER (confidence and prediction interval): ',
      ICER_paired.get_ICER(),
      ICER_paired.get_CI(0.05, 1000),
      ICER_paired.get_PI(0.05, ));

# ICER calculation assuming independent observations
ICER_indp = EconEval.ICER_indp('Testing independent ICER', cost_intervention, health_intervention, cost_base, health_base)
print('Independent ICER (confidence and prediction interval): ',
      ICER_indp.get_ICER(),
      ICER_indp.get_CI(0.05, 1000),
      ICER_indp.get_PI(0.05, ));

# try NMB
NMB_paired = EconEval.NMB_indp("Testing paired NMB", cost_intervention, health_intervention, cost_base, health_base)
print('Paired NMB (confidence and prediction interval): ',
      NMB_paired.get_NMB(wtp=100),
      NMB_paired.get_PI(wtp=100, alpha=.05),
      NMB_paired.get_CI(wtp=100, alpha=.05))

NMB_indp = EconEval.NMB_paired("Testing independent NMB", cost_intervention, health_intervention, cost_base, health_base)
print('Independent NMB (confidence and prediction interval): ',
      NMB_indp.get_NMB(wtp=100),
      NMB_indp.get_CI(wtp=100, alpha=.05),
      NMB_indp.get_PI(wtp=100, alpha=.05))
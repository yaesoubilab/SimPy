import scr.EconEvalClasses as EconEval

cost_intervention = [200, 300, 400]
health_intervention = [1, 0.5, .8]
cost_base = [20, 30, 40]
health_base = [0.5, 0.2, 0.7]

ICER = EconEval.ICERPaired('test', cost_intervention, health_intervention, cost_base, health_base)

print(ICER.get_ICER());

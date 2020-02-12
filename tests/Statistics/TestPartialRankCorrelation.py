import SimPy.SensitivityAnalysis as PRC
from numpy.random import rand

# prepare data
param1 = rand(100) * 20
param2 = rand(100) * -1
output = param1 + 2*param2 + rand(100) * 10

parameterValues = {'Par1': param1, 'Par2': param2}

prc = PRC.PartialRankCorrelation(
    dic_parameter_values=parameterValues,
    output_values=output)

print(prc.results)
prc.export_to_csv(decimal=4)

# # interpret the significance
# alpha = 0.05
# if p > alpha:
#     print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
# else:
#     print('Samples are correlated (reject H0) p=%.3f' % p)
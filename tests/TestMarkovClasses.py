import scr.MarkovClasses as Cls

probMatrix = [
    [0.721,	0.202,	0.067,	0.010],
    [0.000,	0.581,	0.407,	0.012],
    [0.000,	0.000,	0.750,	0.250],
    [0.000,	0.000,	0.000,	1.000]
]

deltaT = 1
rateMatrix = Cls.discrete_to_continuous(probMatrix, deltaT)

print(rateMatrix, '\n')

newProbMatrix = Cls.continuous_to_discrete(rateMatrix, 1)
print(newProbMatrix)



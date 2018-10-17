import SimPy.DecisionTree as dt

# Migraine model displayed here: tests/DecisionTree/Migraine Model.tif

# dictionary for decision nodes
#               // key: cost, utility, [future nodes]
dictDecisions = {'d1': [0,     0,       ['c1', 'c5']]};

# dictionary for chance nodes
#           // key: cost,   utility,  [future nodes],  [probabilities]
dictChances = {'c1': [16.1,   0,       ['c2', 'c3'],    [.558, .442]],
               'c2': [0,      0,       ['t1', 't2'],    [.594, .406]],
               'c3': [0,      0,       ['t3', 'c4'],    [.92, .08]],
               'c4': [63.16,  0,       ['t4', 't5'],    [.998, .002]],
               'c5': [1.32,   0,       ['c6', 'c7'],    [.379, .621]],
               'c6': [0,      0,       ['t6', 't7'],    [.703, .297]],
               'c7': [0,      0,       ['t8', 'c8'],    [.92, .08]],
               'c8': [63.13,  0,       ['t9', 't10'],   [.998, .002]]};

# dictionary for terminal nodes
#               //key: cost, utility
dictTerminals = {'t1': [0,      1],
                 't2': [16.10,  .9],
                 't3': [0,     -.3],
                 't4': [0,      .1],
                 't5': [1093,  -.3],
                 't6': [0,       1],
                 't7': [1.32,   .9],
                 't8': [0,     -.3],
                 't9': [0,      .1],
                 't10': [1093, -.3]};

# build the decision tree
myDT = dt.DecisionNode('d1', dictDecisions, dictChances, dictTerminals)

# print the expected cost and utility of each alternative
print('\nExpected cost and utility:')
print(myDT.get_cost_utility())

# print the probability of terminal nodes under each alternative
print('\nProbabilities of terminal nodes:')
print(myDT.get_terminal_prob())

# plot the expected cost and utility of each alternative
dt.graph_outcomes(myDT)

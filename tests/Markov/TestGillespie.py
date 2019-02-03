import SimPy.MarkovClasses as Cls
import numpy as np

myGillespie = Cls.Gillespie(transition_rate_matrix=
                            [[None, 0.1],
                             [0, None]])

t = 0
i = 0
rng = np.random.RandomState()
while t is not None:
    print('Current state:', i)
    t, i = myGillespie.get_next_state(current_state_index=i, rng=rng)
    print('Time to next transition:', t)


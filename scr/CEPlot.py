

class Strategy:

    def __init__(self, cost, effect):
        self.cost = cost
        self.effect = effect
        self.dominated = False


class CEA:
    def __init__(self, strategies):
        """
        :param strategies: the list of strategies
        """
        self.strategies = strategies

    def FindFrontier(self):
        pass

    def BuildCETable(self):
        pass


S0 = Strategy(0, 0)
S1 = Strategy(10, 2)
S2 = Strategy(30, 1)

strategies = [S0, S1, S2]

myCEA = CEA(strategies)
myCEA.FindFrontier()

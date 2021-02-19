
class SimModel:
    # abstract class to be overridden by the simulation model to optimize
    def __init__(self):
        pass

    def get_obj_value(self, x, seed_index=0):
        """
        abstract method to return one realization of the objective function to optimize
        :param x: the values of the variables
        :param seed_index: specify if need to use a different seed for this simulation replication
        """

        raise NotImplementedError("This is an abstract method and needs to be implemented in derived classes.")
import SimPy.RandomVariantGenerators as RVGs
import numpy as np


class OneDimDataFrame:
    """
    example:
        age,    mortality rate
        0,      0.1,
        5,      0.2,
        10,     0.3
    """
    def __init__(self, y_objects, x_min=0, x_max=1, x_delta=1):
        """
        :param y_objects: (list) of y objects (in example above: [0.1, 0.2, 0.3])
                but it could also be list of other objects of the same type.
        :param x_min: minimum value of x (in example above: 0)
        :param x_max: maximum value of x (in example above: 10)
        :param x_delta: interval between break points of x (in example above: 5)
                    if set to 'int', x is treated as categorical variable
        """

        self.yValues = y_objects
        self.xDelta = x_delta
        self.xMin = x_min
        self.xMax = x_max

    def get_index(self, x_value):
        """ :returns the index of the smallest x break point with x_value greater than or equal.
            In the example above, it returns
                0 for 0 <= x_value < 5,
                1 for 5 <= x_value < 10, and
                2 for x_value >= 10
        """

        if self.xDelta == 'int':
            if type(x_value) is not int:
                raise ValueError('x_value should be an integer for categorical variables.')
            return x_value
        else:
            if x_value >= self.xMax:
                return len(self.yValues) - 1
            else:
                return round((x_value-self.xMin)/self.xDelta)

    def get_value(self, x_value):
        """ :returns the the y-value of the smallest x break point with x_value greater than or equal.
            In the example above, it returns
                0.1 for 0 <= x_value < 5,
                0.2 for 5 <= x_value < 10, and
                0.3 for x_value >= 10
        """

        return self.yValues[self.get_index(x_value)]

    def get_value_by_index(self, x_index):
        """ :returns the the y-value of the x break point located at index x_index.
            In the example above, it returns
                0.1 for x_index = 0,
                0.2 for x_index = 1, and
                0.3 for x_index = 2
        """

        return self.yValues[x_index]


class OneDimDataFrameWithExpDist (OneDimDataFrame):
    """
    example:
        age,    mortality rate
        0,      0.1,
        5,      0.2,
        10,     0.3
    """
    def __init__(self, y_values, x_min=0, x_max=1, x_delta=1):
        """
        :param y_values: (list) of rates (in example above: [0.1, 0.2, 0.3]
        :param x_min: minimum value of x (in example above: 0)
        :param x_max: maximum value of x (in example above: 10)
        :param x_delta: interval between break points of x (in example above: 5)
                    if set to 'int', x is treated as categorical variable
        """

        y_objects = []
        for v in y_values:
            y_objects.append(RVGs.Exponential(scale=1/v))

        OneDimDataFrame.__init__(self, y_objects=y_objects,
                                 x_min=x_min,
                                 x_max=x_max,
                                 x_delta=x_delta)


class _DataFrame:
    def __init__(self, list_x_min, list_x_max, list_x_delta):

        self.xMin = list_x_min[0]
        self.xMax = list_x_max[0]
        self.xDelta = list_x_delta[0]

        self.ifOneDim = False
        if len(list_x_min) == 1:
            self.ifOneDim = True

        self.values = []
        self.dataFrames = []

        x = self.xMin
        while x <= self.xMax:
            if self.ifOneDim:
                self.values.append(0)
            else:
                self.dataFrames.append(
                    _DataFrame(list_x_min=list_x_min[1:],
                               list_x_max=list_x_max[1:],
                               list_x_delta=list_x_delta[1:]
                               )
                )
            if self.xDelta == 'int':
                x += 1
            else:
                x += self.xDelta

    def __get_index(self, x_value):

        if self.xDelta == 'int':
            if type(x_value[0]) is not int:
                raise ValueError('x_value should be an integer for categorical variables.')
            return x_value[0]
        else:
            if x_value[0] > self.xMax:
                return len(self.values)-1
            else:
                return round((x_value[0] - self.xMin) / self.xDelta)

    def get_sum_values(self):
        if self.ifOneDim:
            return sum(self.values)
        else:
            sum_df = 0
            for df in self.dataFrames:
                sum_df += df.get_sum_values()
            return  sum_df

    def update_value(self, x_value, v):

        if self.ifOneDim:
            self.values[self.__get_index(x_value)] = v
        else:
            self.dataFrames[self.__get_index(x_value)].update_value(x_value=x_value[1:], v=v)

    def get_index(self, x_value):

        if self.ifOneDim:
            return [self.__get_index(x_value=x_value)]
        else:
            return self.dataFrames[self.__get_index(x_value=x_value)].get_index(x_value=x_value[1:])

    def get_value(self, x_value):

        if self.ifOneDim:
            return self.values[self.__get_index(x_value=x_value)]
        else:
            return self.dataFrames[self.__get_index(x_value=x_value)].get_value(x_value[1:])

    def get_value_by_index(self, x_index):

        if self.ifOneDim:
            return self.values[x_index[0]]
        else:
            return self.dataFrames[x_index[0]].get_value_by_index(x_index[1:])

    def get_sample_indices(self, rng):
        """
        :param rng: random number generator
        :return: (list) indices of categories
                (in the example above, [1, 0] corresponds to age group 1 and sex group 0.
        """

        probs = []
        if self.ifOneDim:
            probs = self.values
        else:
            for df in self.dataFrames:
                probs.append(df.get_sum_values())

        emprical_dist = RVGs.Empirical(probabilities=np.array(probs)/sum(probs))

        idx = emprical_dist.sample(rng=rng)

        if self.ifOneDim:
            return [idx]
        else:
            a = [idx]
            a.extend(self.dataFrames[idx].get_sample_indices(rng))
            return a


class DataFrame(_DataFrame):
    """
    example:
        age,   sex,      mortality rate
        0,     0,        0.1,
        0,     1,        0.11,
        5,     0,        0.2,
        5,     1,        0.21,
        10,    0,        0.3
        10,    1,        0.31
    """
    def __init__(self, rows, list_x_min, list_x_max, list_x_delta):
        """
        :param rows: (list of list) the table above
        :param list_x_min: list of minimum value of x (in example above: [0, 0])
        :param list_x_max: list of maximum value of x (in example above: [10, 1])
        :param list_x_delta: list of interval between break points of x
                    if set to 'int', x is treated as categorical variable
                    (in example above: [5, 'int'])
        """

        _DataFrame.__init__(self,
                            list_x_min=list_x_min,
                            list_x_max=list_x_max,
                            list_x_delta=list_x_delta)

        for row in rows:
            self.update_value(x_value=row[0:-1], v=row[-1])


class DataFrameWithExpDist(_DataFrame):
    """
    example:
        age,   sex,      mortality rate
        0,     0,        0.1,
        0,     1,        0.11,
        5,     0,        0.2,
        5,     1,        0.21,
        10,    0,        0.3
        10,    1,        0.31
    """
    def __init__(self, rows, list_x_min, list_x_max, list_x_delta):
        """
        :param rows: (list of list) the table above
        :param list_x_min: list of minimum value of x (in example above: [0, 0])
        :param list_x_max: list of maximum value of x (in example above: [10, 1])
        :param list_x_delta: list of interval between break points of x
                    if set to 'int', x is treated as categorical variable
                    (in example above: [5, 'int'])
        """

        _DataFrame.__init__(self,
                            list_x_min=list_x_min,
                            list_x_max=list_x_max,
                            list_x_delta=list_x_delta)

        for row in rows:
            self.update_value(x_value=row[0:-1],
                              v=RVGs.Exponential(scale=1/row[-1]))


class DataFrameWithEmpiricalDist(DataFrame):
    """
    example:
        age,   sex,      probability
        0,     0,        0.1,
        0,     1,        0.2,
        5,     0,        0.3,
        5,     1,        0.4,
    """
    def __init__(self, rows, list_x_min, list_x_max, list_x_delta):
        """
        :param rows: (list of list) the table above
        :param list_x_min: list of minimum value of x (in example above: [0, 0])
        :param list_x_max: list of maximum value of x (in example above: [10, 1])
        :param list_x_delta: list of interval between break points of x
                    if set to 'int', x is treated as categorical variable
                    (in example above: [5, 'int'])
        """

        DataFrame.__init__(self, rows=rows,
                           list_x_min=list_x_min,
                           list_x_max=list_x_max,
                           list_x_delta=list_x_delta)
        self.listXDelta = list_x_delta

        # make sure probabilities add to 1
        sum_probs = 0
        for row in rows:
            sum_probs += row[-1]
        if sum_probs < 0.99999 or sum_probs > 1.000001:
            raise ValueError('Sum of probabilities should add to 1.')

    def get_sample_values(self, rng):
        """
        :param rng: random number generator
        :return: (list) values of categories
                (in the example above, [1.4, 0] corresponds to age 1.4 and sex group 0.
        """

        values = []
        idx = self.get_sample_indices(rng=rng)
        for i, deltaX in enumerate(self.listXDelta):
            if deltaX == 'int':
                values.append(idx[i])
            else:
                unif_dist = RVGs.Uniform(loc=0, scale=deltaX)
                values.append(idx[i]*deltaX + unif_dist.sample(rng=rng))

        return values
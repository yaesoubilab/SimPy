
class OneDimDataFrame:
    """
    example:
        age,    mortality rate
        0,      0.1,
        5,      0.2,
        10,     0.3
    """
    def __init__(self, y_values, x_min=0, x_max=1, x_delta=1):
        """
        :param y_values: (list) of y values (in example above: [0.1, 0.2, 0.3]
        :param x_min: minimum value of x (in example above: 0)
        :param x_max: maximum value of x (in example above: 10)
        :param x_delta: interval between break points of x (in example above: 5)
                    if set to 'int', x is treated as categorical variable
        """

        self.yValues = y_values
        self.xDelta = x_delta
        self.xMin = x_min
        self.xMax = x_max

    def get_index(self, x_value):

        if self.xDelta == 'int':
            if type(x_value) is not int:
                raise ValueError('x_value should be an integer for categorical variables.')
            return x_value
        else:

            if x_value > self.xMax:
                return len(self.yValues) - 1
            else:
                return round((x_value-self.xMin)/self.xDelta)

    def get_value(self, x_value):

        return self.yValues[self.get_index(x_value)]

    def get_value_by_index(self, x_index):

        return self.yValues[x_index]


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


class MultiDimDataFrame(_DataFrame):
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

        _DataFrame.__init__(self, list_x_min=list_x_min,
                            list_x_max=list_x_max,
                            list_x_delta=list_x_delta)

        for row in rows:
            self.update_value(x_value=row[0:-1], v=row[-1])


import math


class Line:

    def __init__(self, x1, x2, y1, y2):

        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        self.slope = (y2 - y1)/(x2 - x1)
        self.intercept = self.get_y(x=0)

    def get_y(self, x):

        return self.slope * (x - self.x1) + self.y1

    def get_intercept_with_x_axis(self):
        
        try:
            value = - self.y1/self.slope + self.x1
        except ValueError:
            value = math.nan
        
        return value


if __name__ == '__main__':
    test = Line(x1=1, x2=2, y1=0, y2=2)
    print(test.get_intercept_with_x_axis())

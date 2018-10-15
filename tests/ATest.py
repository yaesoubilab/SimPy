import math


class Solver:

    def demo(self, a, b, c):

        d = b ** 2 - 4 * a * c

        if d >= 0:

            disc = math.sqrt(d)

            root1 = (-b + disc) / (2 * a)

            root2 = (-b - disc) / (2 * a)

            print(root1, root2)

        else:

            raise Exception


Solver().demo(-2, -1, -4)
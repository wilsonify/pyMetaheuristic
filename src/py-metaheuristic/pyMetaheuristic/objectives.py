import math


def easom(variables_values=(0, 0)):
    """
    Target Function: Easom Function
    :param variables_values:
    :return:
    """
    return -math.cos(variables_values[0]) * math.cos(variables_values[1]) * math.exp(
        -(variables_values[0] - math.pi) ** 2 - (variables_values[1] - math.pi) ** 2)

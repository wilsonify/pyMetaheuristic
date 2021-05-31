import math
from pprint import pprint

import numpy as np
import pyMetaheuristic
import pytest
from pyMetaheuristic import memetic_algorithm
from pyMetaheuristic.memetic_algorithm import memetic_a
from pyMetaheuristic.objectives import easom


def test_smoke():
    """is anything on fire"""
    print("is anything on fire?")
    pprint(dir(pyMetaheuristic))
    pprint(dir(memetic_algorithm))
    pprint(dir(memetic_a))


def test_memetic_algorithm(front):
    """
    # For Instance, suppose that our Target Function is the Easom Function (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

    :param front:
    :return:
    """

    ma_inst = memetic_a.Memetic(
        target_function=easom,
        population_size=50,
        mutation_rate=0.1,
        elite=1,
        min_values=[-5, -5],
        max_values=[5, 5],
        eta=1,
        mu=1,
        std=0.1,
        generations=100,
    )
    ma = ma_inst.minimize()

    # MA - Solution
    variables = ma[:-1]
    minimum = ma[-1]
    print(
        "Variables: ",
        np.around(variables, 4),
        " Minimum Value Found: ",
        round(minimum, 4),
    )
    assert minimum == pytest.approx(-1.0, abs=0.05)
    assert list(variables) == [
        pytest.approx(math.pi, abs=0.5),
        pytest.approx(math.pi, abs=0.5),
    ]

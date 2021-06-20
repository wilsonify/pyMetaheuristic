import math
from pprint import pprint

import pyMetaheuristic
import pytest
from pyMetaheuristic import genetic_algorithm
from pyMetaheuristic.genetic_algorithm import ga
from pyMetaheuristic.objectives import easom


def test_smoke():
    """is anything on fire"""
    print("is anything on fire?")
    pprint(dir(pyMetaheuristic))
    pprint(dir(genetic_algorithm))
    pprint(dir(ga))


def test_genetic_algorithm(front):
    """
    # For Instance, suppose that our Target Function is the Easom Function (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

    :param front:
    :return:
    """

    ga_search = ga.genetic_algorithm(
        target_function=easom,
        population_size=250,
        mutation_rate=0.1,
        elite=1,
        min_values=[-5, -5],
        max_values=[5, 5],
        eta=1,
        mu=1,
        generations=200,
    )

    variables = ga_search[:-1]
    minimum = ga_search[-1]

    assert minimum == pytest.approx(-1.0, abs=0.05)
    assert list(variables) == [
        pytest.approx(math.pi, abs=0.5),
        pytest.approx(math.pi, abs=0.5),
    ]

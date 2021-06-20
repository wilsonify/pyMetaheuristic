import math
from pprint import pprint

import pyMetaheuristic
import pytest
from pyMetaheuristic import cross_entropy_method
from pyMetaheuristic.cross_entropy_method import cem
from pyMetaheuristic.objectives import easom


def test_smoke():
    """is anything on fire"""
    print("is anything on fire?")
    pprint(dir(pyMetaheuristic))
    pprint(dir(cross_entropy_method))
    pprint(dir(cem))


def test_cross_entropy_method(front):
    """
    test_cross_entropy_method
    :param front:
    :return:
    """

    cem_search = cem.cross_entropy_method(
        target_function=easom,
        n=50,
        min_values=[-5, -5],
        max_values=[5, 5],
        iterations=100,
        learning_rate=0.7,
        k_samples=15,
    )
    variables = cem_search[:-1]
    minimum = cem_search[-1]
    assert minimum == pytest.approx(-1.0, abs=0.05)
    assert list(variables) == [
        pytest.approx(math.pi, abs=0.05),
        pytest.approx(math.pi, abs=0.05),
    ]

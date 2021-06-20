"""
tests for WOA
run with pytest

"""
import math
from pprint import pprint

import pytest
from pyMetaheuristic import whale_optimization_algorithm
from pyMetaheuristic.objectives import easom
from pyMetaheuristic.whale_optimization_algorithm import whale_optimization_a


def test_smoke():
    """
    is anything on fire?
    :return:
    """
    print("is anything on fire?")
    pprint(dir(whale_optimization_algorithm))
    pprint(dir(whale_optimization_a))


def test_whale_optimization_algorithm(front):
    """
    For Instance, suppose that our Target Function is the Easom Function
    (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

    :return:
    """

    woa_instance = whale_optimization_a.WOA(
        target_function=easom,
        hunting_party=150,
        spiral_param=2,
        min_values=[-5, -5],
        max_values=[5, 5],
        iterations=500,
    )
    woa = woa_instance.minimize()

    # WOA - Solution
    variables = woa[0][:-1]
    minimum = woa[0][-1]

    assert minimum == pytest.approx(-1.0, abs=0.05)
    assert list(variables) == [
        pytest.approx(math.pi, abs=0.5),
        pytest.approx(math.pi, abs=0.5),
    ]

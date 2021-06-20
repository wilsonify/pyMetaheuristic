import math
from pprint import pprint

import pyMetaheuristic
import pytest
from pyMetaheuristic import firefly_algorithm
from pyMetaheuristic.firefly_algorithm import firefly_a
from pyMetaheuristic.objectives import easom


def test_smoke():
    """is anything on fire"""
    print("is anything on fire?")
    pprint(dir(pyMetaheuristic))
    pprint(dir(firefly_algorithm))
    pprint(dir(firefly_a))


def test_firefly_algorithm(front):
    """
    # For Instance, suppose that our Target Function is the Easom Function (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

    :param front:
    :return:
    """

    fa = firefly_a.firefly_algorithm(
        target_function=easom,
        swarm_size=50,
        min_values=[-5, -5],
        max_values=[5, 5],
        generations=100,
        alpha_0=0.2,
        beta_0=1,
        gama=1,
    )

    variables = fa[:-1]
    minimum = fa[-1]
    assert minimum == pytest.approx(-1.0, abs=0.05)
    assert list(variables) == [
        pytest.approx(math.pi, abs=0.5),
        pytest.approx(math.pi, abs=0.5),
    ]

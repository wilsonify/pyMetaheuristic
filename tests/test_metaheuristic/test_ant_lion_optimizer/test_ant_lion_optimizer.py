import math
from pprint import pprint

import pyMetaheuristic
import pytest
from pyMetaheuristic import ant_lion_optimizer
from pyMetaheuristic.ant_lion_optimizer import alo
from pyMetaheuristic.objectives import easom


def test_smoke():
    """is anything on fire"""
    print("is anything on fire?")
    pprint(dir(pyMetaheuristic))
    pprint(dir(ant_lion_optimizer))
    pprint(dir(alo))


def test_ant_lion_optimizer(front):
    """
    # For Instance, suppose that our Target Function is the Easom Function (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

    :param front:
    :return:
    """

    alo_search = alo.ant_lion_optimizer(
        target_function=easom,
        colony_size=250,
        min_values=[-5, -5],
        max_values=[5, 5],
        iterations=750,
    )
    variables = alo_search[:-1]
    minimum = alo_search[-1]
    assert minimum == pytest.approx(-1.0, abs=0.01)
    assert list(variables) == [
        pytest.approx(math.pi, abs=0.1),
        pytest.approx(math.pi, abs=0.1),
    ]

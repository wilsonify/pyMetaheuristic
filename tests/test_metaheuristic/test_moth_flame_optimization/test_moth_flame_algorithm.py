import math
from pprint import pprint

import pyMetaheuristic
import pytest
from pyMetaheuristic import moth_flame_optimization
from pyMetaheuristic.moth_flame_optimization import mfa
from pyMetaheuristic.objectives import easom


def test_smoke():
    """is anything on fire"""
    print("is anything on fire?")
    pprint(dir(pyMetaheuristic))
    pprint(dir(moth_flame_optimization))
    pprint(dir(mfa))


def test_moth_flame_algorithm(front):
    """
    # For Instance, suppose that our Target Function is the Easom Function (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

    :param front:
    :return:
    """

    mfo_search = mfa.moth_flame_algorithm(
        target_function=easom,
        swarm_size=50,
        min_values=[-5, -5],
        max_values=[5, 5],
        generations=100,
        b_constant=1,
    )

    variables = mfo_search[:-1]
    minimum = mfo_search[-1]
    assert minimum == pytest.approx(-1.0, abs=0.05)
    assert list(variables) == [
        pytest.approx(math.pi, abs=0.5),
        pytest.approx(math.pi, abs=0.5),
    ]

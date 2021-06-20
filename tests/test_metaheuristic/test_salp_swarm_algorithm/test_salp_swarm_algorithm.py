import math
from pprint import pprint

import pyMetaheuristic
import pytest
from pyMetaheuristic import salp_swarm_algorithm
from pyMetaheuristic.objectives import easom
from pyMetaheuristic.salp_swarm_algorithm import ssa


def test_smoke():
    """is anything on fire"""
    print("is anything on fire?")
    pprint(dir(pyMetaheuristic))
    pprint(dir(salp_swarm_algorithm))
    pprint(dir(ssa))


def test_salp_swarm_algorithm(front):
    """
    # For Instance, suppose that our Target Function is the Easom Function (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

    :param front:
    :return:
    """

    ssa_search = ssa.salp_swarm_algorithm(
        target_function=easom,
        swarm_size=150,
        min_values=[-5, -5],
        max_values=[5, 5],
        iterations=100,
    )

    variables = ssa_search[0][:-1]
    minimum = ssa_search[0][-1]

    assert minimum == pytest.approx(-1.0, abs=0.05)
    assert list(variables) == [
        pytest.approx(math.pi, abs=0.5),
        pytest.approx(math.pi, abs=0.5),
    ]

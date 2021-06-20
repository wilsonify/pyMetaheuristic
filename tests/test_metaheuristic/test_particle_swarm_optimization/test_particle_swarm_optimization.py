import math
from pprint import pprint

import pyMetaheuristic
import pytest
from pyMetaheuristic import particle_swarm_optimization
from pyMetaheuristic.objectives import easom
from pyMetaheuristic.particle_swarm_optimization import pso


def test_smoke():
    """is anything on fire"""
    print("is anything on fire?")
    pprint(dir(pyMetaheuristic))
    pprint(dir(particle_swarm_optimization))
    pprint(dir(pso))


def test_particle_swarm_optimization(front):
    """
    # For Instance, suppose that our Target Function is the Easom Function (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

    :param front:
    :return:
    """

    pso_search = pso.particle_swarm_optimization(
        target_function=easom,
        swarm_size=250,
        min_values=[-5, -5],
        max_values=[5, 5],
        iterations=500,
        decay=0,
        w=0.9,
        c1=2,
        c2=2,
    )

    variables = pso_search[:-1]
    minimum = pso_search[-1]
    assert minimum == pytest.approx(-1.0, abs=0.05)
    assert list(variables) == [
        pytest.approx(math.pi, abs=0.5),
        pytest.approx(math.pi, abs=0.5),
    ]

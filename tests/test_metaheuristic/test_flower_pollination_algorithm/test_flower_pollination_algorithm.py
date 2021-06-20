import math
from pprint import pprint

import pyMetaheuristic
import pytest
from pyMetaheuristic import flower_pollination_algorithm
from pyMetaheuristic.flower_pollination_algorithm import fpa
from pyMetaheuristic.objectives import easom


def test_smoke():
    """is anything on fire"""
    print("is anything on fire?")
    pprint(dir(pyMetaheuristic))
    pprint(dir(flower_pollination_algorithm))
    pprint(dir(fpa))


def test_fpa(front):
    """
    # For Instance, suppose that our Target Function is the Easom Function (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

    :param front:
    :return:
    """

    fpa_search = fpa.flower_pollination_algorithm(
        target_function=easom,
        flowers=150,
        min_values=[-5, -5],
        max_values=[5, 5],
        iterations=250,
        gama=0.5,
        lamb=1.4,
        p=0.8,
    )
    variables = fpa_search[:-1]
    minimum = fpa_search[-1]
    assert minimum == pytest.approx(-1.0, abs=0.05)
    assert list(variables) == [
        pytest.approx(math.pi, abs=0.5),
        pytest.approx(math.pi, abs=0.5),
    ]

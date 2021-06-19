import math
from pprint import pprint

import pyMetaheuristic
import pytest
from pyMetaheuristic import random_search
from pyMetaheuristic.objectives import easom
from pyMetaheuristic.random_search import random_s


def test_smoke():
    """is anything on fire"""
    print("is anything on fire?")
    pprint(dir(pyMetaheuristic))
    pprint(dir(random_search))
    pprint(dir(random_s))


def test_random_search(front):
    """
    # For Instance, suppose that our Target Function is the Easom Function (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

    :param front:
    :return:
    """

    rs = random_s.random_search(
        target_function=easom,
        solutions=150,
        min_values=[-5, -5],
        max_values=[5, 5],
        iterations=1000,
    )

    variables = rs[:-1]
    minimum = rs[-1]
    assert minimum == pytest.approx(-1.0, abs=0.05)
    assert list(variables) == [
        pytest.approx(math.pi, abs=0.5),
        pytest.approx(math.pi, abs=0.5),
    ]

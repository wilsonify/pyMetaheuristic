import math
from pprint import pprint

import pyMetaheuristic
import pytest
from pyMetaheuristic import grey_wolf_optimizer
from pyMetaheuristic.grey_wolf_optimizer import gwo
from pyMetaheuristic.objectives import easom


def test_smoke():
    """is anything on fire"""
    print("is anything on fire?")
    pprint(dir(pyMetaheuristic))
    pprint(dir(grey_wolf_optimizer))
    pprint(dir(gwo))


def test_grey_wolf_optimizer(front):
    """
    # For Instance, suppose that our Target Function is the Easom Function (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

    :param front:
    :return:
    """

    gwo_search = gwo.grey_wolf_optimizer(
        target_function=easom,
        pack_size=50,
        min_values=[-5, -5],
        max_values=[5, 5],
        iterations=100,
    )

    variables = gwo_search[0][:-1]
    minimum = gwo_search[0][-1]
    assert minimum == pytest.approx(-1.0, abs=0.05)
    assert list(variables) == [
        pytest.approx(math.pi, abs=0.5),
        pytest.approx(math.pi, abs=0.5),
    ]

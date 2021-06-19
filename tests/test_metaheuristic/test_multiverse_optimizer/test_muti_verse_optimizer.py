import math
from pprint import pprint

import pyMetaheuristic
import pytest
from pyMetaheuristic import multiverse_optimizer
from pyMetaheuristic.multiverse_optimizer import mvo
from pyMetaheuristic.objectives import easom


def test_smoke():
    """is anything on fire"""
    print("is anything on fire?")
    pprint(dir(pyMetaheuristic))
    pprint(dir(multiverse_optimizer))
    pprint(dir(mvo))


def test_muti_verse_optimizer(front):
    """
    # For Instance, suppose that our Target Function is the Easom Function (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

    :param front:
    :return:
    """

    mvo_search = mvo.muti_verse_optimizer(
        target_function=easom,
        universes=50,
        min_values=[-5, -5],
        max_values=[5, 5],
        iterations=100,
    )

    variables = mvo_search[:-1]
    minimum = mvo_search[-1]
    assert minimum == pytest.approx(-1.0, abs=0.05)
    assert list(variables) == [
        pytest.approx(math.pi, abs=0.5),
        pytest.approx(math.pi, abs=0.5),
    ]

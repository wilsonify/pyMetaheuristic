import math
import os
from pprint import pprint

import numpy as np
import pyMetaheuristic
import pytest
from matplotlib import pyplot as plt
from pyMetaheuristic import adaptive_random_search
from pyMetaheuristic.adaptive_random_search import ars
from pyMetaheuristic.objectives import easom


def test_smoke():
    """is anything on fire"""
    print("is anything on fire?")
    pprint(dir(pyMetaheuristic))
    pprint(dir(adaptive_random_search))
    pprint(dir(ars))


def test_adaptive_random_search(front):
    """
    For Instance, suppose that our Target Function is the Easom Function
    (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)
    """

    ars_search = ars.adaptive_random_search(
        target_function=easom,
        solutions=100,
        min_values=[-5, -5],
        max_values=[5, 5],
        step_size_factor=0.05,
        factor_1=3,
        factor_2=1.5,
        iterations=1000,
        large_step_threshold=15,
        improvement_threshold=25,
    )

    # ARS - Solution
    variables = ars_search[0][:-1]
    minimum = ars_search[0][-1]
    print(f"Variables: {np.around(variables, 4)}")
    print(f"Minimum Value Found: {round(minimum, 4)}")
    assert minimum == pytest.approx(-1.0, abs=0.01)
    assert list(variables) == [
        pytest.approx(math.pi, abs=0.1),
        pytest.approx(math.pi, abs=0.1),
    ]



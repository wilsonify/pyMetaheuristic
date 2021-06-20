import math
from pprint import pprint

import pyMetaheuristic
import pytest
from pyMetaheuristic import sine_cosine_algorithm
from pyMetaheuristic.objectives import easom
from pyMetaheuristic.sine_cosine_algorithm import sine_cosine_a


def test_smoke():
    """
    is anything on fire?
    :return:
    """
    print("is anything on fire?")
    pprint(dir(pyMetaheuristic))
    pprint(dir(sine_cosine_algorithm))
    pprint(dir(sine_cosine_a))


def test_sine_cosine_algorithm(front):
    """
    # For Instance, suppose that our Target Function is the Easom Function (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

    :param front:
    :return:
    """

    sca = sine_cosine_a.sine_cosine_algorithm(
        target_function=easom,
        solutions=150,
        a_linear_component=2,
        min_values=[-5, -5],
        max_values=[5, 5],
        iterations=500,
    )

    variables = sca[:-1]
    minimum = sca[-1]
    assert minimum == pytest.approx(-1.0, abs=0.05)
    assert list(variables) == [
        pytest.approx(math.pi, abs=0.5),
        pytest.approx(math.pi, abs=0.5),
    ]

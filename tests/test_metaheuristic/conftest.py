import numpy as np
import pytest
from pyMetaheuristic.objectives import easom


@pytest.fixture(name="front")
def front_fixture():
    """
    Target Function - Values
    :return:
    """

    x_nda = np.arange(-1, 7, 0.1)
    front = np.zeros((len(x_nda) ** 2, 3))
    count = 0
    for x_j in x_nda:
        for x_k in x_nda:
            front[count, 0] = x_j
            front[count, 1] = x_k
            count = count + 1
    for i in range(0, front.shape[0]):
        front[i, 2] = easom(variables_values=[front[i, 0], front[i, 1]])
    return front
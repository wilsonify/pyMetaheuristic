"""
tests for WOA
run with pytest

"""
import math
import os
from pprint import pprint

import numpy as np
import pytest
from matplotlib import pyplot as plt
from pyMetaheuristic import whale_optimization_algorithm
from pyMetaheuristic.objectives import easom
from pyMetaheuristic.whale_optimization_algorithm import whale_optimization_a


def test_smoke():
    """
    is anything on fire?
    :return:
    """
    print("is anything on fire?")
    pprint(dir(whale_optimization_algorithm))
    pprint(dir(whale_optimization_a))


def test_whale_optimization_algorithm(front):
    """
    For Instance, suppose that our Target Function is the Easom Function
    (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

    :return:
    """

    front_1 = front[:, 0]
    front_2 = front[:, 1]
    func_1_values = front[:, -1]

    # Target Function - Plot
    plt.style.use("bmh")
    fig = plt.figure(figsize=(15, 15))
    ax_3d = fig.add_subplot(111, projection="3d")
    ax_3d.set_xlabel("$x_1$", fontsize=25, labelpad=20)
    ax_3d.set_ylabel("$x_2$", fontsize=25, labelpad=20)
    ax_3d.set_zlabel("$f(x_1, x_2)$", fontsize=25, labelpad=20)
    ax_3d.scatter(front_1, front_2, func_1_values, c=func_1_values, s=50, alpha=0.3)
    ax_3d.scatter(
        math.pi, math.pi, -1, c="red", s=100, alpha=1, edgecolors="k", marker="o"
    )
    ax_3d.text(
        math.pi - 1.0,
        math.pi - 1.5,
        -1,
        "$x_1 = $" + str(round(math.pi, 2)) + " ; $x_2 = $" + str(round(math.pi, 2)),
        size=15,
        zorder=1,
        color="k",
    )
    ax_3d.text(
        math.pi + 0.5,
        math.pi - 2.5,
        -1,
        "$f(x_1;x_2) = $" + str(-1),
        size=15,
        zorder=1,
        color="k",
    )
    plt.savefig(f"{os.path.basename(__file__)}.png")

    # WOA - Parameters
    hunt_p = 150
    par_m = 2
    minv = [-5, -5]
    maxv = [5, 5]
    iterations = 500
    tgt = easom

    # WOA - Algorithm
    woa_instance = whale_optimization_a.WOA(
        target_function=tgt,
        hunting_party=hunt_p,
        spiral_param=par_m,
        min_values=minv,
        max_values=maxv,
        iterations=iterations,
    )
    woa = woa_instance.minimize()

    # WOA - Solution
    variables = woa[0][:-1]
    minimum = woa[0][-1]
    print(
        "Variables: ",
        np.around(variables, 4),
        " Minimum Value Found: ",
        round(minimum, 4),
    )
    assert minimum == pytest.approx(-1.0, abs=0.05)
    assert list(variables) == [
        pytest.approx(math.pi, abs=0.5),
        pytest.approx(math.pi, abs=0.5),
    ]

    # WOA - Plot Solution
    plt.style.use("bmh")
    fig = plt.figure(figsize=(15, 15))
    ax_3d = fig.add_subplot(111, projection="3d")
    ax_3d.set_xlabel("$x_1$", fontsize=25, labelpad=20)
    ax_3d.set_ylabel("$x_2$", fontsize=25, labelpad=20)
    ax_3d.set_zlabel("$f(x_1, x_2)$", fontsize=25, labelpad=20)
    ax_3d.scatter(front_1, front_2, func_1_values, c=func_1_values, s=50, alpha=0.3)
    ax_3d.scatter(
        variables[0],
        variables[1],
        minimum,
        c="b",
        s=150,
        alpha=1,
        edgecolors="k",
        marker="s",
    )
    ax_3d.text(
        math.pi - 1.0,
        math.pi - 1.5,
        -1,
        "$x_1 = $"
        + str(round(variables[0], 2))
        + " ; $x_2 = $"
        + str(round(variables[1], 2)),
        size=15,
        zorder=1,
        color="k",
    )
    ax_3d.text(
        math.pi + 0.5,
        math.pi - 2.5,
        -1,
        "$f(x_1;x_2) = $" + str(round(minimum, 4)),
        size=15,
        zorder=1,
        color="k",
    )
    plt.savefig(f"{os.path.basename(__file__)}.png")

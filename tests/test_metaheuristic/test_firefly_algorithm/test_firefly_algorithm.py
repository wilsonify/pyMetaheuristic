import math
from pprint import pprint

import numpy as np
import pyMetaheuristic
import pytest
from matplotlib import pyplot as plt
from pyMetaheuristic import firefly_algorithm
from pyMetaheuristic.firefly_algorithm import firefly_a
from pyMetaheuristic.objectives import easom
import os

def test_smoke():
    """is anything on fire"""
    print("is anything on fire?")
    pprint(dir(pyMetaheuristic))
    pprint(dir(firefly_algorithm))
    pprint(dir(firefly_a))


def test_firefly_algorithm(front):
    """
    # For Instance, suppose that our Target Function is the Easom Function (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

    :param front:
    :return:
    """

    # Target Function - Values
    front_1 = front[:, 0]
    front_2 = front[:, 1]
    func_1_values = front[:, -1]

    # Target Function - Plot
    plt.style.use("bmh")
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("$x_1$", fontsize=25, labelpad=20)
    ax.set_ylabel("$x_2$", fontsize=25, labelpad=20)
    ax.set_zlabel("$f(x_1, x_2)$", fontsize=25, labelpad=20)
    ax.scatter(front_1, front_2, func_1_values, c=func_1_values, s=50, alpha=0.3)
    ax.scatter(
        math.pi, math.pi, -1, c="red", s=100, alpha=1, edgecolors="k", marker="o"
    )
    ax.text(
        math.pi - 1.0,
        math.pi - 1.5,
        -1,
        "$x_1 = $" + str(round(math.pi, 2)) + " ; $x_2 = $" + str(round(math.pi, 2)),
        size=15,
        zorder=1,
        color="k",
    )
    ax.text(
        math.pi + 0.5,
        math.pi - 2.5,
        -1,
        "$f(x_1;x_2) = $" + str(-1),
        size=15,
        zorder=1,
        color="k",
    )
    plt.savefig(f"{os.path.basename(__file__)}.png")

    # + id="ABmpmFqPt6gd"
    # FA - Parameters
    ss = 50
    minv = [-5, -5]
    maxv = [5, 5]
    iterations = 100
    alp = 0.2
    bet = 1
    gam = 1
    tgt = easom

    # FA - Algorithm
    fa = firefly_a.firefly_algorithm(
        target_function=tgt,
        swarm_size=ss,
        min_values=minv,
        max_values=maxv,
        generations=iterations,
        alpha_0=alp,
        beta_0=bet,
        gama=gam,
    )

    # FA - Solution
    variables = fa[:-1]
    minimum = fa[-1]
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

    # FA - Plot Solution
    plt.style.use("bmh")
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("$x_1$", fontsize=25, labelpad=20)
    ax.set_ylabel("$x_2$", fontsize=25, labelpad=20)
    ax.set_zlabel("$f(x_1, x_2)$", fontsize=25, labelpad=20)
    ax.scatter(front_1, front_2, func_1_values, c=func_1_values, s=50, alpha=0.3)
    ax.scatter(
        variables[0],
        variables[1],
        minimum,
        c="b",
        s=150,
        alpha=1,
        edgecolors="k",
        marker="s",
    )
    ax.text(
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
    ax.text(
        math.pi + 0.5,
        math.pi - 2.5,
        -1,
        "$f(x_1;x_2) = $" + str(round(minimum, 4)),
        size=15,
        zorder=1,
        color="k",
    )
    plt.savefig(f"{os.path.basename(__file__)}.png")

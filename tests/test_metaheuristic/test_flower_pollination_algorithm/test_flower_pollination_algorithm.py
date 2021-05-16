import math
from pprint import pprint

import numpy as np
import pyMetaheuristic
import pytest
from matplotlib import pyplot as plt
from pyMetaheuristic import flower_pollination_algorithm
from pyMetaheuristic.flower_pollination_algorithm import fpa
from pyMetaheuristic.objectives import easom


def test_smoke():
    print("is anything on fire?")
    pprint(dir(pyMetaheuristic))
    pprint(dir(flower_pollination_algorithm))
    pprint(dir(fpa))


def test_fpa():
    # Target Function - It can be any function that needs to be minimize, However it has to have only one argument: 'variables_values'. This Argument must be a list of variables.
    # For Instance, suppose that our Target Function is the Easom Function (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

    # Target Function - Values
    x = np.arange(-1, 7, 0.1)
    front = np.zeros((len(x) ** 2, 3))
    count = 0
    for j in range(0, len(x)):
        for k in range(0, len(x)):
            front[count, 0] = x[j]
            front[count, 1] = x[k]
            count = count + 1
    for i in range(0, front.shape[0]):
        front[i, 2] = easom(variables_values=[front[i, 0], front[i, 1]])
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
    plt.show()

    # + id="sC_ozpQAvxDn"
    # FPA - Parameters
    n_sols = 150
    minv = [-5, -5]
    maxv = [5, 5]
    iterations = 250
    gam = 0.5
    lam = 1.4
    par_p = 0.8
    tgt = easom

    # FPA - Algorithm
    fpa_search = fpa.flower_pollination_algorithm(
        target_function=tgt,
        flowers=n_sols,
        min_values=minv,
        max_values=maxv,
        iterations=iterations,
        gama=gam,
        lamb=1.4,
        p=par_p,
    )

    # FPA - Solution
    variables = fpa_search[:-1]
    minimum = fpa_search[-1]
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

    # FPA - Plot Solution
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
    plt.show()

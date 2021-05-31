############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Random Search

# Citation:
# PEREIRA, V. (2018). Project: Metaheuristic-Random_Search,
# File: Python-MH-Random Search.py,
# GitHub repository: <https://github.com/Valdecy/Metaheuristic-Random_Search>

############################################################################

import copy
import os
import random

# Required Libraries
import numpy as np


def initial_position(
        target_function, solutions=5, min_values=(-5, -5), max_values=(5, 5)
):
    """Initialize Variables"""
    position = np.zeros((solutions, len(min_values) + 1))
    for i in range(solutions):
        for j in range(len(min_values)):
            position[i, j] = random.uniform(min_values[j], max_values[j])
        position[i, -1] = target_function(position[i, 0: position.shape[1] - 1])
    return position


def step(
        target_function, position, min_values=(-5, -5), max_values=(5, 5), step_size=(0, 0)
):
    """non-large Steps"""
    position_temp = np.copy(position)
    for i in range(position.shape[0]):
        for j in range(position.shape[1] - 1):
            minimun = min(min_values[j], position[i, j] + step_size[i][j])
            maximum = max(max_values[j], position[i, j] - step_size[i][j])
            rand = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            position_temp[i, j] = np.clip(
                minimun + (maximum - minimun) * rand, min_values[j], max_values[j]
            )
        position_temp[i, -1] = target_function(
            position_temp[i, 0: position_temp.shape[1] - 1]
        )
    return position_temp


def large_step(
        target_function,
        position,
        min_values=(-5, -5),
        max_values=(5, 5),
        step_size=(0, 0),
        count=0,
        large_step_threshold=10,
        factor_1=3,
        factor_2=1.5,
):
    """Large Steps"""
    factor = 0
    position_temp = np.copy(position)
    step_size_temp = copy.deepcopy(step_size)
    for i in range(position.shape[0]):
        if count > 0 and count % large_step_threshold == 0:
            factor = factor_1
        else:
            factor = factor_2
        for j in range(len(min_values)):
            step_size_temp[i][j] = step_size[i][j] * factor
    for i in range(position.shape[0]):
        for j in range(position.shape[1] - 1):
            minimun = min(min_values[j], position[i, j] + step_size[i][j])
            maximum = max(max_values[j], position[i, j] - step_size[i][j])
            rand = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            position_temp[i, j] = np.clip(
                minimun + (maximum - minimun) * rand, min_values[j], max_values[j]
            )
        position_temp[i, -1] = target_function(
            position_temp[i, 0: position_temp.shape[1] - 1]
        )
    return step_size_temp, position_temp


def adaptive_random_search(
        target_function,
        solutions=5,
        min_values=(-5, -5),
        max_values=(5, 5),
        step_size_factor=0.05,
        factor_1=3,
        factor_2=1.5,
        iterations=50,
        large_step_threshold=10,
        improvement_threshold=25,
):
    """
    ARS Function

    :param target_function:
    It can be any function that needs to be minimize,
    However it has to have only one argument: 'variables_values'.
    This Argument must be a list of variables.

    :param solutions:
    :param min_values:
    :param max_values:
    :param step_size_factor:
    :param factor_1:
    :param factor_2:
    :param iterations:
    :param large_step_threshold:
    :param improvement_threshold:
    :return:
    """
    count = 0
    threshold = [0] * solutions
    position = initial_position(
        target_function=target_function,
        solutions=solutions,
        min_values=min_values,
        max_values=max_values,
    )
    best_solution = np.copy(position[position[:, -1].argsort()][0, :])
    step_size = []
    for i in range(position.shape[0]):
        step_size.append([0] * len(min_values))
        for j in range(len(min_values)):
            step_size[i][j] = (max_values[j] - min_values[j]) * step_size_factor
    while count <= iterations:
        print("Iteration = ", count, " f(x) = ", best_solution[-1])
        position_step = step(
            target_function=target_function,
            position=position,
            min_values=min_values,
            max_values=max_values,
            step_size=step_size,
        )
        step_large, position_large_step = large_step(
            target_function=target_function,
            position=position,
            min_values=min_values,
            max_values=max_values,
            step_size=step_size,
            count=count,
            large_step_threshold=large_step_threshold,
            factor_1=factor_1,
            factor_2=factor_2,
        )
        for i in range(position.shape[0]):
            if (
                    position_step[i, -1] < position[i, -1]
                    or position_large_step[i, -1] < position[i, -1]
            ):
                if position_large_step[i, -1] < position_step[i, -1]:
                    position[i, :] = np.copy(position_large_step[i, :])
                    for j in range(position.shape[1] - 1):
                        step_size[i][j] = step_large[i][j]
                else:
                    position[i, :] = np.copy(position_step[i, :])
                threshold[i] = 0
            else:
                threshold[i] = threshold[i] + 1
            if threshold[i] >= improvement_threshold:
                threshold[i] = 0
                for j in range(len(min_values)):
                    step_size[i][j] = step_size[i][j] / factor_2
        if best_solution[-1] > position[position[:, -1].argsort()][0, -1]:
            best_solution = np.copy(position[position[:, -1].argsort()][0, :])
        count = count + 1
    print(best_solution)
    return best_solution, position


def Target_Function_Plot(front_1, front_2, func_1_values):
    """
    plot the target function
    :param front_1:
    :param front_2:
    :param func_1_values:
    :return:
    """
    from matplotlib import pyplot as plt
    import math
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


def ARS_Solution_Plot(minimum, variables, front_1, front_2, func_1_values):
    """
    plot a solution found by optimizer
    :param minimum:
    :param variables:
    :param front_1:
    :param front_2:
    :param func_1_values:
    :return:
    """
    from matplotlib import pyplot as plt
    import math
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

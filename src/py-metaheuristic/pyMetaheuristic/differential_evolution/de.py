############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Differential Evolution

# Citation:
# PEREIRA, V. (2018). Project: Metaheuristic-Differential_Evolution,
# File: Python-MH-Differential Evolution.py,
# GitHub repository: <https://github.com/Valdecy/Metaheuristic-Differential_Evolution>

############################################################################

import random

# Required Libraries
import numpy as np
# Function: Initialize Variables
from pyMetaheuristic import rando


def initial_position(target_function, n=3, min_values=(-5, -5), max_values=(5, 5)):
    position = np.zeros((n, len(min_values) + 1))
    for i in range(n):
        for j in range(len(min_values)):
            position[i, j] = random.uniform(min_values[j], max_values[j])
        position[i, -1] = target_function(position[i, 0: position.shape[1] - 1])
    return position


# Function: Velocity
def velocity(
        target_function,
        position,
        best_global,
        k0=0,
        k1=1,
        k2=2,
        f=0.9,
        min_values=(-5, -5),
        max_values=(5, 5),
        cr=0.2,
):
    v = np.copy(best_global)
    for i in range(len(best_global)):
        ri = rando()
        if ri <= cr:
            v[i] = best_global[i] + f * (position[k1, i] - position[k2, i])
        else:
            v[i] = position[k0, i]
        if i < len(min_values) and v[i] > max_values[i]:
            v[i] = max_values[i]
        elif i < len(min_values) and v[i] < min_values[i]:
            v[i] = min_values[i]
    v[-1] = target_function(v[0: len(min_values)])
    return v


def differential_evolution(
        target_function,
        n=3,
        min_values=(-5, -5),
        max_values=(5, 5),
        iterations=50,
        f=0.9,
        cr=0.2,
):
    """
    DE Function. DE/Best/1/Bin Scheme.

    :param target_function:
        # Target Function - It can be any function that needs to be minimize, However it has to have only one argument: 'variables_values'. This Argument must be a list of variables.

    :param n:
    :param min_values:
    :param max_values:
    :param iterations:
    :param f:
    :param cr:
    :return:
    """
    count = 0
    position = initial_position(
        target_function=target_function,
        n=n,
        min_values=min_values,
        max_values=max_values,
    )
    best_global = np.copy(position[position[:, -1].argsort()][0, :])
    while count <= iterations:
        print("Iteration = ", count)
        for i in range(position.shape[0]):
            k1 = int(np.random.randint(position.shape[0], size=1))
            k2 = int(np.random.randint(position.shape[0], size=1))
            while k1 == k2:
                k1 = int(np.random.randint(position.shape[0], size=1))
            vi = velocity(
                target_function=target_function,
                position=position,
                best_global=best_global,
                k0=i,
                k1=k1,
                k2=k2,
                f=f,
                min_values=min_values,
                max_values=max_values,
                cr=cr,
            )
            if vi[-1] <= position[i, -1]:
                for j in range(position.shape[1]):
                    position[i, j] = vi[j]
            if best_global[-1] > position[position[:, -1].argsort()][0, :][-1]:
                best_global = np.copy(position[position[:, -1].argsort()][0, :])
        count = count + 1
    print(best_global)
    return best_global

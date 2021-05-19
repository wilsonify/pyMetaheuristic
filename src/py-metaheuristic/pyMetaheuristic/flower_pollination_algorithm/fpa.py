############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Flower Pollination Algorithm

# Citation:
# PEREIRA, V. (2018). Project: Metaheuristic-Flower_Pollination_Algorithm,
# File: Python-MH-Flower Pollination Algorithm.py,
# GitHub repository: <https://github.com/Valdecy/Metaheuristic-Flower_Pollination_Algorithm>

############################################################################

import os
import random
from math import gamma

# Required Libraries
import numpy as np


# Function: Initialize Variables
def initial_position(
        target_function, flowers=3, min_values=(-5, -5), max_values=(5, 5)
):
    position = np.zeros((flowers, len(min_values) + 1))
    for i in range(flowers):
        for j in range(len(min_values)):
            position[i, j] = random.uniform(min_values[j], max_values[j])
        position[i, -1] = target_function(position[i, 0: position.shape[1] - 1])
    return position


# Function Levy Distribution
def levy_flight(beta=1.5):
    r1 = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
    r2 = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
    sig_num = gamma(1 + beta) * np.sin((np.pi * beta) / 2.0)
    sig_den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (sig_num / sig_den) ** (1 / beta)
    levy = (0.01 * r1 * sigma) / (abs(r2) ** (1 / beta))
    return levy


# Function: Global Pollination
def pollination_global(
        target_function,
        position,
        best_global,
        flower=0,
        gama=0.5,
        lamb=1.4,
        min_values=(-5, -5),
        max_values=(5, 5),
):
    x = np.copy(best_global)
    for j in range(len(min_values)):
        x[j] = np.clip(
            (
                    position[flower, j]
                    + gama * levy_flight(lamb) * (position[flower, j] - best_global[j])
            ),
            min_values[j],
            max_values[j],
        )
    x[-1] = target_function(x[0: len(min_values)])
    return x


# Function: Local Pollination
def pollination_local(
        target_function,
        position,
        best_global,
        flower=0,
        nb_flower_1=0,
        nb_flower_2=1,
        min_values=(-5, -5),
        max_values=(5, 5),
):
    x = np.copy(best_global)
    for j in range(len(min_values)):
        r = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
        x[j] = np.clip(
            (
                    position[flower, j]
                    + r * (position[nb_flower_1, j] - position[nb_flower_2, j])
            ),
            min_values[j],
            max_values[j],
        )
    x[-1] = target_function(x[0: len(min_values)])
    return x


def flower_pollination_algorithm(
        target_function,
        flowers=3,
        min_values=(-5, -5),
        max_values=(5, 5),
        iterations=50,
        gama=0.5,
        lamb=1.4,
        p=0.8,
):
    """
    # FPA Function.

    :param target_function:
        # Target Function - It can be any function that needs to be minimize, However it has to have only one argument: 'variables_values'. This Argument must be a list of variables.

    :param flowers:
    :param min_values:
    :param max_values:
    :param iterations:
    :param gama:
    :param lamb:
    :param p:
    :return:
    """
    count = 0
    position = initial_position(
        target_function=target_function,
        flowers=flowers,
        min_values=min_values,
        max_values=max_values,
    )
    best_global = np.copy(position[position[:, -1].argsort()][0, :])
    x = np.copy(best_global)
    while count <= iterations:
        print("Iteration = ", count, " f(x) = ", best_global[-1])
        for i in range(position.shape[0]):
            nb_flower_1 = int(np.random.randint(position.shape[0], size=1))
            nb_flower_2 = int(np.random.randint(position.shape[0], size=1))
            while nb_flower_1 == nb_flower_2:
                nb_flower_1 = int(np.random.randint(position.shape[0], size=1))
            r = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            if r < p:
                x = pollination_global(
                    target_function=target_function,
                    position=position,
                    best_global=best_global,
                    flower=i,
                    gama=gama,
                    lamb=lamb,
                    min_values=min_values,
                    max_values=max_values,
                )
            else:
                x = pollination_local(
                    target_function=target_function,
                    position=position,
                    best_global=best_global,
                    flower=i,
                    nb_flower_1=nb_flower_1,
                    nb_flower_2=nb_flower_2,
                    min_values=min_values,
                    max_values=max_values,
                )
            if x[-1] <= position[i, -1]:
                for j in range(position.shape[1]):
                    position[i, j] = x[j]
            value = np.copy(position[position[:, -1].argsort()][0, :])
            if best_global[-1] > value[-1]:
                best_global = np.copy(value)
        count = count + 1
    print(best_global)
    return best_global

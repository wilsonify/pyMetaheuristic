############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Sine Cosine Algorithm

# Citation:
# PEREIRA, V. (2018). Project: Metaheuristic-Sine_Cosine_Algorithm,
# File: Python-MH-Sine Cosine Algorithm.py,
# GitHub repository: <https://github.com/Valdecy/Metaheuristic-Sine Cosine_Algorithm>

############################################################################

import math
import os
import random

# Required Libraries
import numpy as np


# Function: Initialize Variables
def initial_position(
    target_function, solutions=5, min_values=(-5, -5), max_values=(5, 5)
):
    position = np.zeros((solutions, len(min_values) + 1))
    for i in range(solutions):
        for j in range(len(min_values)):
            position[i, j] = random.uniform(min_values[j], max_values[j])
        position[i, -1] = target_function(position[i, 0 : position.shape[1] - 1])
    return position


# Function: Update Position
def update_position(
    target_function, position, destination, r1=2, min_values=(-5, -5), max_values=(5, 5)
):
    for i in range(position.shape[0]):
        for j in range(len(min_values)):
            r2 = (
                2
                * math.pi
                * (int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1))
            )
            r3 = 2 * (int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1))
            r4 = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            if r4 < 0.5:
                position[i, j] = np.clip(
                    (
                        position[i, j]
                        + (
                            r1
                            * math.sin(r2)
                            * abs(r3 * destination[j] - position[i, j])
                        )
                    ),
                    min_values[j],
                    max_values[j],
                )
            else:
                position[i, j] = np.clip(
                    (
                        position[i, j]
                        + (
                            r1
                            * math.cos(r2)
                            * abs(r3 * destination[j] - position[i, j])
                        )
                    ),
                    min_values[j],
                    max_values[j],
                )
        position[i, -1] = target_function(position[i, 0 : position.shape[1] - 1])
    return position


# SCA Function
def sine_cosine_algorithm(
    target_function,
    solutions=5,
    a_linear_component=2,
    min_values=(-5, -5),
    max_values=(5, 5),
    iterations=50,
):
    """

    :param target_function:
    # Target Function - It can be any function that needs to be minimize, However it has to have only one argument: 'variables_values'. This Argument must be a list of variables.

    :param solutions:
    :param a_linear_component:
    :param min_values:
    :param max_values:
    :param iterations:
    :return:
    """
    count = 0
    position = initial_position(
        target_function=target_function,
        solutions=solutions,
        min_values=min_values,
        max_values=max_values,
    )
    destination = np.copy(position[position[:, -1].argsort()][0, :])
    while count <= iterations:
        print("Iteration = ", count, " f(x) = ", destination[-1])
        r1 = a_linear_component - count * (a_linear_component / iterations)
        position = update_position(
            target_function=target_function,
            position=position,
            destination=destination,
            r1=r1,
            min_values=min_values,
            max_values=max_values,
        )
        value = np.copy(position[position[:, -1].argsort()][0, :])
        if destination[-1] > value[-1]:
            destination = np.copy(value)
        count = count + 1
    print(destination)
    return destination

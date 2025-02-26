############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Firefly Algorithm

# Citation:
# PEREIRA, V. (2018). Project: Metaheuristic-Firefly_Algorithm,
# File: Python-MH-Firefly Algorithm.py,
# GitHub repository: <https://github.com/Valdecy/Metaheuristic-Firefly_Algorithm>

############################################################################

import math
import random

# Required Libraries
import numpy as np
# Function: Initialize Variables
from pyMetaheuristic import rando


def initial_fireflies(
        target_function, swarm_size=3, min_values=(-5, -5), max_values=(5, 5)
):
    position = np.zeros((swarm_size, len(min_values) + 1))
    for i in range(swarm_size):
        for j in range(len(min_values)):
            position[i, j] = random.uniform(min_values[j], max_values[j])
        position[i, -1] = target_function(position[i, 0: position.shape[1] - 1])
    return position


# Function: Distance Calculations
def euclidean_distance(x, y):
    distance = 0
    for j in range(len(x)):
        distance = (x[j] - y[j]) ** 2 + distance
    return distance ** (1 / 2)


# Function: Beta Value
def beta_value(x, y, gama=1, beta_0=1):
    rij = euclidean_distance(x, y)
    beta = beta_0 * math.exp(-gama * rij ** 2)
    return beta


# Function: Ligth Intensity
def ligth_value(light_0, x, y, gama=1):
    rij = euclidean_distance(x, y)
    light = light_0 * math.exp(-gama * rij ** 2)
    return light


# Function: Update Position
def update_position(
        target_function,
        position,
        x,
        y,
        alpha_0=0.2,
        beta_0=1,
        gama=1,
        firefly=0,
        min_values=(-5, -5),
        max_values=(5, 5),
):
    for j in range(len(x)):
        epson = rando() - (
                1 / 2
        )
        position[firefly, j] = np.clip(
            (
                    x[j]
                    + beta_value(x, y, gama=gama, beta_0=beta_0) * (y[j] - x[j])
                    + alpha_0 * epson
            ),
            min_values[j],
            max_values[j],
        )
    position[firefly, -1] = target_function(
        position[firefly, 0: position.shape[1] - 1]
    )
    return position


def firefly_algorithm(
        target_function,
        swarm_size=3,
        min_values=(-5, -5),
        max_values=(5, 5),
        generations=50,
        alpha_0=0.2,
        beta_0=1,
        gama=1,
):
    """
    # FA Function

    :param target_function:
        # Target Function - It can be any function that needs to be minimize, However it has to have only one argument: 'variables_values'. This Argument must be a list of variables.

    :param swarm_size:
    :param min_values:
    :param max_values:
    :param generations:
    :param alpha_0:
    :param beta_0:
    :param gama:
    :return:
    """
    count = 0
    position = initial_fireflies(
        target_function=target_function,
        swarm_size=swarm_size,
        min_values=min_values,
        max_values=max_values,
    )
    while count <= generations:
        print(
            "Generation: ",
            count,
            " f(x) = ",
            position[position[:, -1].argsort()][0, :][-1],
        )
        for i in range(swarm_size):
            for j in range(swarm_size):
                if i != j:
                    firefly_i = np.copy(position[i, 0: position.shape[1] - 1])
                    firefly_j = np.copy(position[j, 0: position.shape[1] - 1])
                    ligth_i = ligth_value(
                        position[i, -1], firefly_i, firefly_j, gama=gama
                    )
                    ligth_j = ligth_value(
                        position[j, -1], firefly_i, firefly_j, gama=gama
                    )
                    if ligth_i > ligth_j:
                        position = update_position(
                            target_function=target_function,
                            position=position,
                            x=firefly_i,
                            y=firefly_j,
                            alpha_0=alpha_0,
                            beta_0=beta_0,
                            gama=gama,
                            firefly=i,
                            min_values=min_values,
                            max_values=max_values,
                        )
        count = count + 1
    best_firefly = np.copy(position[position[:, -1].argsort()][0, :])
    print(best_firefly)
    return best_firefly

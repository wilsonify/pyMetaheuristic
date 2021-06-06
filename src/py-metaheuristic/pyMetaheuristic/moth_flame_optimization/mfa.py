"""
Like a moth to a flame!
distracted by artificial lights moths lose their natural flying path in deadly spiral flying path.

a moth flies by maintaining a fixed angle with respect to the moon:
    * when the ligth source is far, a straight line.
    * when the light source is near, a spiral path.

"""
############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Moth Flame Algorithm

# Citation:
# PEREIRA, V. (2018). Project: Metaheuristic-Moth_Flame_Algorithm,
# File: Python-MH-Moth Flame Algorithm.py,
# GitHub repository: <https://github.com/Valdecy/Metaheuristic-Moth_Flame_Algorithm>

############################################################################

import math
import random

# Required Libraries
import numpy as np
# Function: Initialize Variables
from pyMetaheuristic import rando


def initial_moths(
        target_function, swarm_size=3, min_values=(-5, -5), max_values=(5, 5)
):
    position = np.zeros((swarm_size, len(min_values) + 1))
    for i in range(swarm_size):
        for j in range(len(min_values)):
            position[i, j] = random.uniform(min_values[j], max_values[j])
        position[i, -1] = target_function(position[i, 0: position.shape[1] - 1])
    return position


# Function: Update Flames
def update_flames(flames, position):
    population = np.vstack([flames, position])
    flames = np.copy(population[population[:, -1].argsort()][: flames.shape[0], :])
    return flames


# Function: Update Position
def update_position(
        target_function,
        position,
        flames,
        flame_number=1,
        b_constant=1,
        a_linear_component=1,
        min_values=(-5, -5),
        max_values=(5, 5),
):
    for i in range(position.shape[0]):
        for j in range(len(min_values)):
            if i <= flame_number:
                flame_distance = abs(flames[i, j] - position[i, j])
                rnd_1 = rando()
                rnd_2 = (a_linear_component - 1) * rnd_1 + 1
                position[i, j] = (
                        flame_distance
                        * math.exp(b_constant * rnd_2)
                        * math.cos(rnd_2 * 2 * math.pi)
                        + flames[i, j]
                )
            elif i > flame_number:
                flame_distance = abs(flames[i, j] - position[i, j])
                rnd_1 = rando()
                rnd_2 = (a_linear_component - 1) * rnd_1 + 1
                position[i, j] = np.clip(
                    (
                            flame_distance
                            * math.exp(b_constant * rnd_2)
                            * math.cos(rnd_2 * 2 * math.pi)
                            + flames[flame_number, j]
                    ),
                    min_values[j],
                    max_values[j],
                )
        position[i, -1] = target_function(position[i, 0: position.shape[1] - 1])
    return position


# MFA Function
def moth_flame_algorithm(
        target_function,
        swarm_size=3,
        min_values=(-5, -5),
        max_values=(5, 5),
        generations=50,
        b_constant=1,
):
    """

    :param target_function:
        # Target Function - It can be any function that needs to be minimize, However it has to have only one argument: 'variables_values'. This Argument must be a list of variables.

    :param swarm_size:
    :param min_values:
    :param max_values:
    :param generations:
    :param b_constant:
    :return:
    """
    count = 0
    position = initial_moths(
        target_function=target_function,
        swarm_size=swarm_size,
        min_values=min_values,
        max_values=max_values,
    )
    flames = np.copy(position[position[:, -1].argsort()][:, :])
    best_moth = np.copy(flames[0, :])
    while count <= generations:
        print("Generation: ", count, " of ", generations, " f(x) = ", best_moth[-1])
        flame_number = round(
            position.shape[0] - count * ((position.shape[0] - 1) / generations)
        )
        a_linear_component = -1 + count * ((-1) / generations)
        position = update_position(
            target_function=target_function,
            position=position,
            flames=flames,
            flame_number=flame_number,
            b_constant=b_constant,
            a_linear_component=a_linear_component,
            min_values=min_values,
            max_values=max_values,
        )
        flames = update_flames(flames, position)
        count = count + 1
        if best_moth[-1] > flames[0, -1]:
            best_moth = np.copy(flames[0, :])
    print(best_moth)
    return best_moth

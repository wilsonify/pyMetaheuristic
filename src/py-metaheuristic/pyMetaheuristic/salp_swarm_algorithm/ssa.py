"""

The Salp Swarm Algorithm (SSA) mimics salps to solve optimization problems.

Salps have transparent barrel-shaped body and propulsion movement similar to jellyfishes.

In deep oceans, salps often form a swarm called salp chain. Some believe
that this is done for achieving better locomotion using rapid coordinated changes and foraging.

"""
############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Salp Swarm Algorithm

# Citation:
# PEREIRA, V. (2018). Project: Metaheuristic-Salp_Swarm_Algorithm,
# File: Python-MH-Salp Swarm Algorithm.py,
# GitHub repository: <https://github.com/Valdecy/Metaheuristic-Salp_Swarm_Algorithm>

############################################################################

import math
import random

# Required Libraries
import numpy as np
# Function: Initialize Variables
from pyMetaheuristic import rando


def initial_position(
        target_function, swarm_size=5, min_values=(-5, -5), max_values=(5, 5)
):
    position = np.zeros((swarm_size, len(min_values) + 1))
    for i in range(swarm_size):
        for j in range(len(min_values)):
            position[i, j] = random.uniform(min_values[j], max_values[j])
        position[i, -1] = target_function(position[i, 0: position.shape[1] - 1])
    return position


# Function: Initialize Food Position
def food_position(target_function, dimension=2):
    food = np.zeros((1, dimension + 1))
    for j in range(dimension):
        food[0, j] = 0.0
    food[0, -1] = target_function(food[0, 0: food.shape[1] - 1])
    return food


# Function: Update Food Position by Fitness
def update_food(position, food):
    for i in range(position.shape[0]):
        if food[0, -1] > position[i, -1]:
            for j in range(position.shape[1]):
                food[0, j] = position[i, j]
    return food


# Function: Update Position
def update_position(
        target_function, position, food, c1=1.0, min_values=(-5, -5), max_values=(5, 5)
):
    for i in range(position.shape[0]):
        if i <= position.shape[0] / 2:
            for j in range(len(min_values)):
                c2 = rando()
                c3 = rando()
                if c3 >= 0.5:  # c3 < 0.5
                    position[i, j] = np.clip(
                        (
                                food[0, j]
                                + c1
                                * ((max_values[j] - min_values[j]) * c2 + min_values[j])
                        ),
                        min_values[j],
                        max_values[j],
                    )
                else:
                    position[i, j] = np.clip(
                        (
                                food[0, j]
                                - c1
                                * ((max_values[j] - min_values[j]) * c2 + min_values[j])
                        ),
                        min_values[j],
                        max_values[j],
                    )
        elif i > position.shape[0] / 2 and i < position.shape[0] + 1:
            for j in range(len(min_values)):
                position[i, j] = np.clip(
                    ((position[i - 1, j] + position[i, j]) / 2),
                    min_values[j],
                    max_values[j],
                )
        position[i, -1] = target_function(position[i, 0: position.shape[1] - 1])
    return position


# SSA Function
def salp_swarm_algorithm(
        target_function, swarm_size=5, min_values=(-5, -5), max_values=(5, 5), iterations=50
):
    """

    :param target_function:
    Target Function - It can be any function that needs to be minimize,
    However it has to have only one argument: 'variables_values'. This Argument must be a list of variables.

    :param swarm_size:
    :param min_values:
    :param max_values:
    :param iterations:
    :return:
    """
    count = 0
    position = initial_position(
        target_function=target_function,
        swarm_size=swarm_size,
        min_values=min_values,
        max_values=max_values,
    )
    food = food_position(target_function=target_function, dimension=len(min_values))
    while count <= iterations:
        print("Iteration = ", count, " f(x) = ", food[0, -1])
        c1 = 2 * math.exp(-((4 * (count / iterations)) ** 2))
        food = update_food(position, food)
        position = update_position(
            target_function=target_function,
            position=position,
            food=food,
            c1=c1,
            min_values=min_values,
            max_values=max_values,
        )
        count = count + 1
    print(food)
    return food

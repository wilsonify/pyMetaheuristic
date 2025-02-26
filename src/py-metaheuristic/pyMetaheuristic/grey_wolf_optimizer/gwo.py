"""
The Grey Wolf Optimizer (GWO) mimics the leadership hierarchy and hunting mechanism of grey wolves in nature.

Four types of grey wolves:
    * alpha
    * beta
    * delta
    * omega

Three steps to perform optimization.
    * searching
    * encircling
    * attacking
"""
############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Grey Wolf Optimizer

# Citation:
# PEREIRA, V. (2018). Project: Metaheuristic-Grey_Wolf_Optimizer,
# File: Python-MH-Grey Wolf Optimizer.py,
# GitHub repository: <https://github.com/Valdecy/Metaheuristic-Grey_Wolf_Optimizer>

############################################################################

import random

# Required Libraries
import numpy as np
# Function: Initialize Variables
from pyMetaheuristic import rando


def initial_position(
        target_function, pack_size=5, min_values=(-5, -5), max_values=(5, 5)
):
    position = np.zeros((pack_size, len(min_values) + 1))
    for i in range(pack_size):
        for j in range(len(min_values)):
            position[i, j] = random.uniform(min_values[j], max_values[j])
        position[i, -1] = target_function(position[i, 0: position.shape[1] - 1])
    return position


# Function: Initialize Alpha
def alpha_position(target_function, dimension=2):
    alpha = np.zeros((1, dimension + 1))
    for j in range(dimension):
        alpha[0, j] = 0.0
    alpha[0, -1] = target_function(alpha[0, 0: alpha.shape[1] - 1])
    return alpha


# Function: Initialize Beta
def beta_position(target_function, dimension=2):
    beta = np.zeros((1, dimension + 1))
    for j in range(dimension):
        beta[0, j] = 0.0
    beta[0, -1] = target_function(beta[0, 0: beta.shape[1] - 1])
    return beta


# Function: Initialize Delta
def delta_position(target_function, dimension=2):
    delta = np.zeros((1, dimension + 1))
    for j in range(dimension):
        delta[0, j] = 0.0
    delta[0, -1] = target_function(delta[0, 0: delta.shape[1] - 1])
    return delta


# Function: Update Pack by Fitness
def update_pack(position, alpha, beta, delta):
    updated_position = np.copy(position)
    for i in range(position.shape[0]):
        if updated_position[i, -1] < alpha[0, -1]:
            alpha[0, :] = np.copy(updated_position[i, :])
        if (
                updated_position[i, -1] > alpha[0, -1]
                and updated_position[i, -1] < beta[0, -1]
        ):
            beta[0, :] = np.copy(updated_position[i, :])
        if (
                updated_position[i, -1] > alpha[0, -1]
                and updated_position[i, -1] > beta[0, -1]
                and updated_position[i, -1] < delta[0, -1]
        ):
            delta[0, :] = np.copy(updated_position[i, :])
    return alpha, beta, delta


# Function: Update Position
def update_position(
        target_function,
        position,
        alpha,
        beta,
        delta,
        a_linear_component=2,
        min_values=(-5, -5),
        max_values=(5, 5),
):
    updated_position = np.copy(position)
    for i in range(updated_position.shape[0]):
        for j in range(len(min_values)):
            r1_alpha = rando()
            r2_alpha = rando()
            a_alpha = 2 * a_linear_component * r1_alpha - a_linear_component
            c_alpha = 2 * r2_alpha
            distance_alpha = abs(c_alpha * alpha[0, j] - position[i, j])
            x1 = alpha[0, j] - a_alpha * distance_alpha
            r1_beta = rando()
            r2_beta = rando()
            a_beta = 2 * a_linear_component * r1_beta - a_linear_component
            c_beta = 2 * r2_beta
            distance_beta = abs(c_beta * beta[0, j] - position[i, j])
            x2 = beta[0, j] - a_beta * distance_beta
            r1_delta = rando()
            r2_delta = rando()
            a_delta = 2 * a_linear_component * r1_delta - a_linear_component
            c_delta = 2 * r2_delta
            distance_delta = abs(c_delta * delta[0, j] - position[i, j])
            x3 = delta[0, j] - a_delta * distance_delta
            updated_position[i, j] = np.clip(
                ((x1 + x2 + x3) / 3), min_values[j], max_values[j]
            )
        updated_position[i, -1] = target_function(
            updated_position[i, 0: updated_position.shape[1] - 1]
        )
    return updated_position


def grey_wolf_optimizer(
        target_function, pack_size=5, min_values=(-5, -5), max_values=(5, 5), iterations=50
):
    """
    GWO Function

    :param target_function:
    Target Function - It can be any function that needs to be minimize, However it has to have only one argument: 'variables_values'. This Argument must be a list of variables.

    :param pack_size:
    :param min_values:
    :param max_values:
    :param iterations:
    :return:
    """
    count = 0
    alpha = alpha_position(target_function=target_function, dimension=len(min_values))
    beta = beta_position(target_function=target_function, dimension=len(min_values))
    delta = delta_position(target_function=target_function, dimension=len(min_values))
    position = initial_position(
        target_function=target_function,
        pack_size=pack_size,
        min_values=min_values,
        max_values=max_values,
    )
    while count <= iterations:
        print("Iteration = ", count, " f(x) = ", alpha[-1])
        a_linear_component = 2 - count * (2 / iterations)
        alpha, beta, delta = update_pack(position, alpha, beta, delta)
        position = update_position(
            target_function=target_function,
            position=position,
            alpha=alpha,
            beta=beta,
            delta=delta,
            a_linear_component=a_linear_component,
            min_values=min_values,
            max_values=max_values,
        )
        count = count + 1
    print(alpha[-1])
    return alpha

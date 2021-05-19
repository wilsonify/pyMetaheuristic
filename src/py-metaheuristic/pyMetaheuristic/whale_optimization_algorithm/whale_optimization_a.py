"""
Whale Optimization Algorithm (WOA)
This algorithm includes three operators to simulate
the bubble-net foraging behavior of humpback whales
 * searching
 * encircling
 * attacking
"""
############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Whale Optimization Algorithm

# Citation:
# PEREIRA, V. (2018). Project: Metaheuristic-Whale_Optimization_Algorithm,
# File: Python-MH-Whale Optimization Algorithm.py,
# GitHub repository: <https://github.com/Valdecy/Metaheuristic-Whale_Optimization_Algorithm>

############################################################################

import math
import os
import random

# Required Libraries
import numpy as np


def initial_position(
        target_function, hunting_party=5, min_values=(-5, -5), max_values=(5, 5)
):
    """
    Initialize Variables
    :param target_function:
    :param hunting_party:
    :param min_values:
    :param max_values:
    :return:
    """
    position = np.zeros((hunting_party, len(min_values) + 1))
    for i in range(hunting_party):
        for j, _ in enumerate(min_values):
            position[i, j] = random.uniform(min_values[j], max_values[j])
        position[i, -1] = target_function(position[i, 0: position.shape[1] - 1])
    return position


def leader_position(target_function, dimension=2):
    """
    Initialize Alpha
    :param target_function:
    :param dimension:
    :return:
    """
    leader = np.zeros((1, dimension + 1))
    for j in range(dimension):
        leader[0, j] = 0.0
    leader[0, -1] = target_function(leader[0, 0: leader.shape[1] - 1])
    return leader


def update_leader(position, leader):
    """
    Update Leader by Fitness
    :param position:
    :param leader:
    :return:
    """
    for i in range(position.shape[0]):
        if leader[0, -1] > position[i, -1]:
            for j in range(position.shape[1]):
                leader[0, j] = position[i, j]
    return leader


def update_position(
        target_function,
        position,
        leader,
        a_linear_component=2,
        b_linear_component=1,
        spiral_param=1,
        min_values=(-5, -5),
        max_values=(5, 5),
):
    """
    Update Position
    :param target_function:
    :param position:
    :param leader:
    :param a_linear_component:
    :param b_linear_component:
    :param spiral_param:
    :param min_values:
    :param max_values:
    :return:
    """
    for i in range(position.shape[0]):
        r1_leader = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
        r2_leader = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
        p_value = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)

        a_leader = 2 * a_linear_component * r1_leader - a_linear_component
        c_leader = 2 * r2_leader

        for j, _ in enumerate(min_values):
            if p_value < 0.5:
                if abs(a_leader) >= 1:
                    rand = int.from_bytes(os.urandom(8), byteorder="big") / (
                            (1 << 64) - 1
                    )
                    rand_leader_index = math.floor(position.shape[0] * rand)
                    x_rand = position[rand_leader_index, :]
                    distance_x_rand = abs(c_leader * x_rand[j] - position[i, j])
                    position[i, j] = np.clip(
                        x_rand[j] - a_leader * distance_x_rand,
                        min_values[j],
                        max_values[j],
                    )
                elif abs(a_leader) < 1:
                    distance_leader = abs(c_leader * leader[0, j] - position[i, j])
                    position[i, j] = np.clip(
                        leader[0, j] - a_leader * distance_leader,
                        min_values[j],
                        max_values[j],
                    )
            elif p_value >= 0.5:
                distance_leader = abs(leader[0, j] - position[i, j])
                rand = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
                m_param = (b_linear_component - 1) * rand + 1
                position[i, j] = np.clip(
                    (
                            distance_leader
                            * math.exp(spiral_param * m_param)
                            * math.cos(m_param * 2 * math.pi)
                            + leader[0, j]
                    ),
                    min_values[j],
                    max_values[j],
                )
        position[i, -1] = target_function(position[i, 0: position.shape[1] - 1])
    return position


def whale_optimization_algorithm(
        target_function,
        hunting_party=5,
        spiral_param=1,
        min_values=(-5, -5),
        max_values=(5, 5),
        iterations=50,
):
    """
    WOA Function

    :param target_function: It can be any function that needs to be minimize,
    However it has to have only one argument: 'variables_values'.
    This Argument must be a list of variables.

    :param hunting_party:
    :param spiral_param:
    :param min_values:
    :param max_values:
    :param iterations:
    :return:
    """
    count = 0
    position = initial_position(
        target_function=target_function,
        hunting_party=hunting_party,
        min_values=min_values,
        max_values=max_values,
    )
    leader = leader_position(target_function=target_function, dimension=len(min_values))
    while count <= iterations:
        print("Iteration = ", count, " f(x) = ", leader[0, -1])
        a_linear_component = 2 - count * (2 / iterations)
        b_linear_component = -1 + count * (-1 / iterations)
        leader = update_leader(position, leader)
        position = update_position(
            target_function=target_function,
            position=position,
            leader=leader,
            a_linear_component=a_linear_component,
            b_linear_component=b_linear_component,
            spiral_param=spiral_param,
            min_values=min_values,
            max_values=max_values,
        )
        count = count + 1
    print(leader)
    return leader

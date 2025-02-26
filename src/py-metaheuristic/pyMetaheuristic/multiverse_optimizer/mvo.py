"""
the Multi-Verse Optimizer (MVO) based on cosmological concepts:
 * white hole -> exploration
 * black hole -> exploitation
 * wormhole. -> local search
"""
############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Multi-Verse Optimizer

# Citation:
# PEREIRA, V. (2018). Project: Metaheuristic-Multi-Verse_Optimizer,
# File: Python-MH-Multi-Verse Optimizer.py,
# GitHub repository: <https://github.com/Valdecy/Metaheuristic-Multi-Verse_Optimizer>

############################################################################

import math
import os
import random

# Required Libraries
import numpy as np
from pyMetaheuristic import rando

# Function: Initialize Variables
def initial_universes(
        target_function, universes=5, min_values=(-5, -5), max_values=(5, 5)
):
    cosmos = np.zeros((universes, len(min_values) + 1))
    for i in range(universes):
        for j in range(len(min_values)):
            cosmos[i, j] = random.uniform(min_values[j], max_values[j])
        cosmos[i, -1] = target_function(cosmos[i, 0: cosmos.shape[1] - 1])
    return cosmos


# Function: Fitness
def fitness_function(cosmos):
    fitness = np.zeros((cosmos.shape[0], 2))
    for i in range(fitness.shape[0]):
        fitness[i, 0] = 1 / (1 + cosmos[i, -1] + abs(cosmos[:, -1].min()))
    fit_sum = fitness[:, 0].sum()
    fitness[0, 1] = fitness[0, 0]
    for i in range(1, fitness.shape[0]):
        fitness[i, 1] = fitness[i, 0] + fitness[i - 1, 1]
    for i in range(fitness.shape[0]):
        fitness[i, 1] = fitness[i, 1] / fit_sum
    return fitness


# Function: Selection
def roulette_wheel(fitness):
    ix = 0
    _random = rando()
    for i in range(fitness.shape[0]):
        if _random <= fitness[i, 1]:
            ix = i
            break
    return ix


# Function: Big Bang
def big_bang(
        target_function,
        cosmos,
        fitness,
        best_universe,
        wormhole_existence_probability,
        travelling_distance_rate,
        min_values=(-5, -5),
        max_values=(5, 5),
):
    for i in range(cosmos.shape[0]):
        for j in range(len(min_values)):
            r1 = rando()
            if r1 < fitness[i, 1]:
                white_hole_i = roulette_wheel(fitness)
                cosmos[i, j] = cosmos[white_hole_i, j]
            r2 = rando()
            if r2 < wormhole_existence_probability:
                r3 = rando()
                if r3 <= 0.5:
                    rand = int.from_bytes(os.urandom(8), byteorder="big") / (
                            (1 << 64) - 1
                    )
                    cosmos[i, j] = best_universe[j] + travelling_distance_rate * (
                            (max_values[j] - min_values[j]) * rand + min_values[j]
                    )
                elif r3 > 0.5:
                    rand = int.from_bytes(os.urandom(8), byteorder="big") / (
                            (1 << 64) - 1
                    )
                    cosmos[i, j] = np.clip(
                        (
                                best_universe[j]
                                - travelling_distance_rate
                                * ((max_values[j] - min_values[j]) * rand + min_values[j])
                        ),
                        min_values[j],
                        max_values[j],
                    )
        cosmos[i, -1] = target_function(cosmos[i, 0: cosmos.shape[1] - 1])
    return cosmos


def muti_verse_optimizer(
        target_function, universes=5, min_values=(-5, -5), max_values=(5, 5), iterations=50
):
    """
    MVO Function
    Multi-Verse Optimizer to Minimize Functions with Continuous Variables.


    :param target_function:
        target_function = Function to be minimized.

        It can be any function that needs to be minimize, However it has to have only one argument: 'variables_values'. This Argument must be a list of variables.

    :param universes:
        universes = The population size. The Default Value is 5.
    :param min_values:
        min_values = The minimum value that the variable(s) from a list can have. The default value is -5.
    :param max_values:
        max_values = The maximum value that the variable(s) from a list can have. The default value is 5.
    :param iterations:
        iterations = The total number of iterations. The Default Value is 50.

    :return:

    The function returns:
    1) An array containing the used value(s) for the target function and the output of the target function f(x).
    For example, if the function f(x1, x2) is used, then the array would be [x1, x2, f(x1, x2)].

    """
    count = 0
    cosmos = initial_universes(
        target_function=target_function,
        universes=universes,
        min_values=min_values,
        max_values=max_values,
    )
    fitness = fitness_function(cosmos)
    best_universe = np.copy(cosmos[cosmos[:, -1].argsort()][0, :])
    wormhole_existence_probability_max = 1.0
    wormhole_existence_probability_min = 0.2
    while count <= iterations:
        print("Iteration = ", count, " f(x) = ", best_universe[-1])
        wormhole_existence_probability = wormhole_existence_probability_min + count * (
                (wormhole_existence_probability_max - wormhole_existence_probability_min)
                / iterations
        )
        travelling_distance_rate = 1 - (
                math.pow(count, 1 / 6) / math.pow(iterations, 1 / 6)
        )
        cosmos = big_bang(
            target_function=target_function,
            cosmos=cosmos,
            fitness=fitness,
            best_universe=best_universe,
            wormhole_existence_probability=wormhole_existence_probability,
            travelling_distance_rate=travelling_distance_rate,
            min_values=min_values,
            max_values=max_values,
        )
        fitness = fitness_function(cosmos)
        value = np.copy(cosmos[cosmos[:, -1].argsort()][0, :])
        if best_universe[-1] > value[-1]:
            best_universe = np.copy(value)
        count = count + 1
    print(best_universe)
    return best_universe

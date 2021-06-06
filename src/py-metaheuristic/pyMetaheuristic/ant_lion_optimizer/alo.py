"""
The Ant Lion Optimizer (ALO) mimics the hunting mechanism of antlions.

Five main steps:
    * random walk of ants,
    * building traps,
    * entrapment of ants in traps,
    * catching preys,
    * re-building traps
"""
############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Ant Lion Optimizer

# Citation:
# PEREIRA, V. (2018). Project: Metaheuristic-Ant_Lion_Optimizer,
# File: Python-MH-Ant_Lion_Optimizer.py,
# GitHub repository: <https://github.com/Valdecy/Metaheuristic-Ant_Lion_Optimizer>

############################################################################

import random

# Required Libraries
import numpy as np
# Function: Initialize Variables
from pyMetaheuristic import rando


def initial_population(
        target_function, colony_size=5, min_values=(-5, -5), max_values=(5, 5)
):
    population = np.zeros((colony_size, len(min_values) + 1))
    for i in range(colony_size):
        for j in range(len(min_values)):
            population[i, j] = random.uniform(min_values[j], max_values[j])
        population[i, -1] = target_function(population[i, 0: population.shape[1] - 1])
    return population


# Function: Fitness
def fitness_function(population):
    fitness = np.zeros((population.shape[0], 2))
    for i in range(fitness.shape[0]):
        fitness[i, 0] = 1 / (1 + population[i, -1] + abs(population[:, -1].min()))
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


# Function: Random Walk
def random_walk(iterations):
    x_random_walk = [0] * (iterations + 1)
    x_random_walk[0] = 0
    for k in range(1, len(x_random_walk)):
        rand = rando()
        if rand > 0.5:
            rand = 1
        else:
            rand = 0
        x_random_walk[k] = x_random_walk[k - 1] + (2 * rand - 1)
    return x_random_walk


# Function: Combine Ants
def combine(population, antlions):
    combination = np.vstack([population, antlions])
    combination = combination[combination[:, -1].argsort()]
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            antlions[i, j] = combination[i, j]
            population[i, j] = combination[i + population.shape[0], j]
    return population, antlions


# Function: Update Antlion
def update_ants(
        target_function,
        population,
        antlions,
        count,
        iterations,
        min_values=(-5, -5),
        max_values=(5, 5),
):
    i_ratio = 1
    minimum_c_i = np.zeros((1, population.shape[1]))
    maximum_d_i = np.zeros((1, population.shape[1]))
    minimum_c_e = np.zeros((1, population.shape[1]))
    maximum_d_e = np.zeros((1, population.shape[1]))
    elite_antlion = np.zeros((1, population.shape[1]))
    if count > 0.10 * iterations:
        w_exploration = 2
        i_ratio = (10 ** w_exploration) * (count / iterations)
    elif count > 0.50 * iterations:
        w_exploration = 3
        i_ratio = (10 ** w_exploration) * (count / iterations)
    elif count > 0.75 * iterations:
        w_exploration = 4
        i_ratio = (10 ** w_exploration) * (count / iterations)
    elif count > 0.90 * iterations:
        w_exploration = 5
        i_ratio = (10 ** w_exploration) * (count / iterations)
    elif count > 0.95 * iterations:
        w_exploration = 6
        i_ratio = (10 ** w_exploration) * (count / iterations)
    for i in range(population.shape[0]):
        fitness = fitness_function(antlions)
        ant_lion = roulette_wheel(fitness)
        for j in range(population.shape[1] - 1):
            minimum_c_i[0, j] = antlions[antlions[:, -1].argsort()][0, j] / i_ratio
            maximum_d_i[0, j] = antlions[antlions[:, -1].argsort()][-1, j] / i_ratio
            elite_antlion[0, j] = antlions[antlions[:, -1].argsort()][0, j]
            minimum_c_e[0, j] = antlions[antlions[:, -1].argsort()][0, j] / i_ratio
            maximum_d_e[0, j] = antlions[antlions[:, -1].argsort()][-1, j] / i_ratio
            rand = rando()
            if rand < 0.5:
                minimum_c_i[0, j] = minimum_c_i[0, j] + antlions[ant_lion, j]
                minimum_c_e[0, j] = minimum_c_e[0, j] + elite_antlion[0, j]
            else:
                minimum_c_i[0, j] = -minimum_c_i[0, j] + antlions[ant_lion, j]
                minimum_c_e[0, j] = -minimum_c_e[0, j] + elite_antlion[0, j]

            rand = rando()
            if rand >= 0.5:
                maximum_d_i[0, j] = maximum_d_i[0, j] + antlions[ant_lion, j]
                maximum_d_e[0, j] = maximum_d_e[0, j] + elite_antlion[0, j]
            else:
                maximum_d_i[0, j] = -maximum_d_i[0, j] + antlions[ant_lion, j]
                maximum_d_e[0, j] = -maximum_d_e[0, j] + elite_antlion[0, j]
            x_random_walk = random_walk(iterations)
            e_random_walk = random_walk(iterations)
            min_x, max_x = min(x_random_walk), max(x_random_walk)
            x_random_walk[count] = (
                                           (
                                                   (x_random_walk[count] - min_x)
                                                   * (maximum_d_i[0, j] - minimum_c_i[0, j])
                                           )
                                           / (max_x - min_x)
                                   ) + minimum_c_i[0, j]
            min_e, max_e = min(e_random_walk), max(e_random_walk)
            e_random_walk[count] = (
                                           (
                                                   (e_random_walk[count] - min_e)
                                                   * (maximum_d_e[0, j] - minimum_c_e[0, j])
                                           )
                                           / (max_e - min_e)
                                   ) + minimum_c_e[0, j]
            population[i, j] = np.clip(
                (x_random_walk[count] + e_random_walk[count]) / 2,
                min_values[j],
                max_values[j],
            )
        population[i, -1] = target_function(population[i, 0: population.shape[1] - 1])
        return population, antlions


# ALO Function
def ant_lion_optimizer(
        target_function,
        colony_size=5,
        min_values=(-5, -5),
        max_values=(5, 5),
        iterations=50,
):
    """

    :param target_function:
        # Target Function - It can be any function that needs to be minimize, However it has to have only one argument: 'variables_values'. This Argument must be a list of variables.

    :param colony_size:
    :param min_values:
    :param max_values:
    :param iterations:
    :return:
    """
    count = 0
    population = initial_population(
        target_function=target_function,
        colony_size=colony_size,
        min_values=min_values,
        max_values=max_values,
    )
    antlions = initial_population(
        target_function=target_function,
        colony_size=colony_size,
        min_values=min_values,
        max_values=max_values,
    )
    elite = np.copy(antlions[antlions[:, -1].argsort()][0, :])
    while count <= iterations:
        print("Iteration = ", count, " f(x) = ", elite[-1])
        population, antlions = update_ants(
            target_function=target_function,
            population=population,
            antlions=antlions,
            count=count,
            iterations=iterations,
            min_values=min_values,
            max_values=max_values,
        )
        population, antlions = combine(population, antlions)
        value = np.copy(antlions[antlions[:, -1].argsort()][0, :])
        if elite[-1] > value[-1]:
            elite = np.copy(value)
        else:
            antlions[antlions[:, -1].argsort()][0, :] = np.copy(elite)
        count = count + 1
    print(elite)
    return elite

############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Genetic Algorithm

# Citation:
# PEREIRA, V. (2018). Project: Metaheuristic-Genetic_Algorithm,
# File: Python-MH-Genetic Algorithm.py,
# GitHub repository: <https://github.com/Valdecy/Metaheuristic-Genetic_Algorithm>

############################################################################

import os
import random

import numpy as np


# Function: Initialize Variables
from pyMetaheuristic import rando


def initial_population(
        target_function, population_size=5, min_values=(-5, -5), max_values=(5, 5)
):
    population = np.zeros((population_size, len(min_values) + 1))
    for i in range(population_size):
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


# Function: Offspring
def breeding(
        target_function,
        population,
        fitness,
        min_values=(-5, -5),
        max_values=(5, 5),
        mu=1,
        elite=0,
):
    offspring = np.copy(population)
    b_offspring = 0
    if elite > 0:
        preserve = np.copy(population[population[:, -1].argsort()])
        for i in range(elite):
            for j in range(offspring.shape[1]):
                offspring[i, j] = preserve[i, j]
    for i in range(elite, offspring.shape[0]):
        parent_1, parent_2 = roulette_wheel(fitness), roulette_wheel(fitness)
        while parent_1 == parent_2:
            parent_2 = np.random.choice(range(len(population) - 1), 1)[0]
            print(f"parent_2 = {parent_2}")
        for j in range(offspring.shape[1] - 1):
            rand = rando()
            rand_b = rando()
            if rand <= 0.5:
                b_offspring = 2 * rand_b
                b_offspring = b_offspring ** (1 / (mu + 1))
            elif rand > 0.5:
                b_offspring = 1 / (2 * (1 - rand_b))
                b_offspring = b_offspring ** (1 / (mu + 1))
            offspring[i, j] = np.clip(
                (
                        (1 + b_offspring) * population[parent_1, j]
                        + (1 - b_offspring) * population[parent_2, j]
                )
                / 2,
                min_values[j],
                max_values[j],
            )
            if i < population.shape[0] - 1:
                offspring[i + 1, j] = np.clip(
                    (
                            (1 - b_offspring) * population[parent_1, j]
                            + (1 + b_offspring) * population[parent_2, j]
                    )
                    / 2,
                    min_values[j],
                    max_values[j],
                )
        offspring[i, -1] = target_function(offspring[i, 0: offspring.shape[1] - 1])
    return offspring


# Function: Mutation
def mutation(
        target_function,
        offspring,
        mutation_rate=0.1,
        eta=1,
        min_values=(-5, -5),
        max_values=(5, 5),
):
    d_mutation = 0
    for i in range(offspring.shape[0]):
        for j in range(offspring.shape[1] - 1):
            probability = int.from_bytes(os.urandom(8), byteorder="big") / (
                    (1 << 64) - 1
            )
            if probability < mutation_rate:
                rand = rando()
                rand_d = int.from_bytes(os.urandom(8), byteorder="big") / (
                        (1 << 64) - 1
                )
                if rand <= 0.5:
                    d_mutation = 2 * rand_d
                    d_mutation = d_mutation ** (1 / (eta + 1)) - 1
                elif rand > 0.5:
                    d_mutation = 2 * (1 - rand_d)
                    d_mutation = 1 - d_mutation ** (1 / (eta + 1))
                offspring[i, j] = np.clip(
                    (offspring[i, j] + d_mutation), min_values[j], max_values[j]
                )
        offspring[i, -1] = target_function(offspring[i, 0: offspring.shape[1] - 1])
    return offspring


def genetic_algorithm(
        target_function,
        population_size=5,
        mutation_rate=0.1,
        elite=0,
        min_values=(-5, -5),
        max_values=(5, 5),
        eta=1,
        mu=1,
        generations=50,
):
    """
    GA Function

    :param target_function:
        # Target Function - It can be any function that needs to be minimize, However it has to have only one argument: 'variables_values'. This Argument must be a list of variables.

    :param population_size:
    :param mutation_rate:
    :param elite:
    :param min_values:
    :param max_values:
    :param eta:
    :param mu:
    :param generations:
    :return:
    """
    count = 0
    population = initial_population(
        target_function=target_function,
        population_size=population_size,
        min_values=min_values,
        max_values=max_values,
    )
    fitness = fitness_function(population)
    elite_ind = np.copy(population[population[:, -1].argsort()][0, :])
    while count <= generations:
        print("Generation = ", count, " f(x) = ", elite_ind[-1])
        offspring = breeding(
            target_function=target_function,
            population=population,
            fitness=fitness,
            min_values=min_values,
            max_values=max_values,
            mu=mu,
            elite=elite,
        )
        population = mutation(
            target_function=target_function,
            offspring=offspring,
            mutation_rate=mutation_rate,
            eta=eta,
            min_values=min_values,
            max_values=max_values,
        )
        fitness = fitness_function(population)
        value = np.copy(population[population[:, -1].argsort()][0, :])
        if elite_ind[-1] > value[-1]:
            elite_ind = np.copy(value)
        count = count + 1
    print(elite_ind)
    return elite_ind

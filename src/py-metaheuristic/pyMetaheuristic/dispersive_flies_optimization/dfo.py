"""
Dispersive flies optimisation (DFO) is a bare-bones swarm intelligence algorithm which is
inspired by the swarming behaviour of flies hovering over food sources.

DFO is a simple optimiser which works by iteratively trying to improve a candidate solution
with regard to a numerical measure that is calculated by a fitness function.

Each member of the population, a fly or an agent, holds a candidate solution whose suitability
can be evaluated by their fitness value.

Optimisation problems are often formulated as either minimisation or maximisation problems.

DFO [2] was introduced with the intention of analysing a simplified swarm intelligence algorithm
with the fewest tunable parameters and components.

In the first work on DFO, this algorithm was compared against a few other existing swarm intelligence
techniques using error, efficiency and diversity measures.
It is shown that despite the simplicity of the algorithm, which only uses agents’ position vectors at time t
to generate the position vectors for time t + 1, it exhibits a competitive performance.

"""
############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Dispersive Flies Optimization

# Citation:
# PEREIRA, V. (2018). Project: Metaheuristic-Dispersive_Flies_Optimization,
# File: Python-MH-Dispersive Flies Optimization.py,
# GitHub repository: <https://github.com/Valdecy/Metaheuristic-Dispersive_Flies_Optimization>

############################################################################

import os

# Required Libraries
import numpy as np


def initial_flies(target_function, swarm_size=3, min_values=(-5, -5), max_values=(5, 5)):
    """Initialize Variables"""
    position = np.zeros((swarm_size, len(min_values) + 1))
    for i in range(swarm_size):
        for j, _ in enumerate(min_values):
            random_int = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            position[i, j] = min_values[j] + random_int * (max_values[j] - min_values[j])
        position[i, -1] = target_function(position[i, 0: position.shape[1] - 1])
    return position


def update_position(target_function, position, neighbour_best, swarm_best, min_values=(-5, -5), max_values=(5, 5),
                    fly=0):
    """Update Position"""
    for j in range(position.shape[1] - 1):
        random_int = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
        position[fly, j] = np.clip(
            (neighbour_best[j] + random_int * (swarm_best[j] - position[fly, j])),
            min_values[j],
            max_values[j],
        )
    position[fly, -1] = target_function(position[fly, 0: position.shape[1] - 1])
    return position


def dispersive_fly_optimization(
        target_function,
        swarm_size=3,
        min_values=(-5, -5),
        max_values=(5, 5),
        generations=50,
        thresh=0.2
):
    """
    # DFO Function

    :param swarm_size:

    It can be any function that needs to be minimize,
    However it has to have only one argument: 'variables_values'.
    This Argument must be a list of variables.

    :param min_values:
    :param max_values:
    :param generations:
    :param thresh:
    :param target_function:
    :return:
    """
    count = 0
    population = initial_flies(target_function=target_function, swarm_size=swarm_size, min_values=min_values,
                               max_values=max_values)
    neighbour_best = np.copy(population[population[:, -1].argsort()][0, :])
    swarm_best = np.copy(population[population[:, -1].argsort()][0, :])
    while count <= generations:
        print("Generation: ", count, " of ", generations, " f(x) = ", swarm_best[-1])
        for i in range(swarm_size):
            population = update_position(
                target_function=target_function,
                position=population,
                neighbour_best=neighbour_best,
                swarm_best=swarm_best,
                min_values=min_values,
                max_values=max_values,
                fly=i
            )
            random_number = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            if random_number < thresh:
                for j, _ in enumerate(min_values):
                    random_number = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
                    population[i, j] = min_values[j] + random_number * (
                            max_values[j] - min_values[j]
                    )
                population[i, -1] = target_function(
                    population[i, 0: population.shape[1] - 1]
                )
        neighbour_best = np.copy(population[population[:, -1].argsort()][0, :])
        if swarm_best[-1] > neighbour_best[-1]:
            swarm_best = np.copy(neighbour_best)
        count = count + 1
    print(swarm_best)
    return swarm_best

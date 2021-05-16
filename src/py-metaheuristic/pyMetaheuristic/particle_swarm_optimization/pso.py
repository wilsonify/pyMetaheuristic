############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Particle Swarm Optimization

# Citation:
# PEREIRA, V. (2018). Project: Metaheuristic-Particle_Swarm_Optimization, File: Python-MH-Particle Swarm Optimization.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Particle_Swarm_Optimization>

############################################################################

import os
import random

# Required Libraries
import numpy as np


# Function: Initialize Variables
def initial_position(
        target_function, swarm_size=3, min_values=(-5, -5), max_values=(5, 5)
):
    position = np.zeros((swarm_size, len(min_values) + 1))
    for i in range(0, swarm_size):
        for j in range(0, len(min_values)):
            position[i, j] = random.uniform(min_values[j], max_values[j])
        position[i, -1] = target_function(position[i, 0: position.shape[1] - 1])
    return position


# Function: Initialize Velocity
def initial_velocity(position, min_values=(-5, -5), max_values=(5, 5)):
    init_velocity = np.zeros((position.shape[0], len(min_values)))
    for i in range(0, init_velocity.shape[0]):
        for j in range(0, init_velocity.shape[1]):
            init_velocity[i, j] = random.uniform(min_values[j], max_values[j])
    return init_velocity


# Function: Individual Best
def individual_best_matrix(position, i_b_matrix):
    for i in range(0, position.shape[0]):
        if i_b_matrix[i, -1] > position[i, -1]:
            for j in range(0, position.shape[1]):
                i_b_matrix[i, j] = position[i, j]
    return i_b_matrix


# Function: Velocity
def velocity_vector(
        position, init_velocity, i_b_matrix, best_global, w=0.5, c1=2, c2=2
):
    r1 = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
    r2 = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
    velocity = np.zeros((position.shape[0], init_velocity.shape[1]))
    for i in range(0, init_velocity.shape[0]):
        for j in range(0, init_velocity.shape[1]):
            velocity[i, j] = (
                    w * init_velocity[i, j]
                    + c1 * r1 * (i_b_matrix[i, j] - position[i, j])
                    + c2 * r2 * (best_global[j] - position[i, j])
            )
    return velocity


# Function: Updtade Position
def update_position(
        target_function, position, velocity, min_values=(-5, -5), max_values=(5, 5)
):
    for i in range(0, position.shape[0]):
        for j in range(0, position.shape[1] - 1):
            position[i, j] = np.clip(
                (position[i, j] + velocity[i, j]), min_values[j], max_values[j]
            )
        position[i, -1] = target_function(position[i, 0: position.shape[1] - 1])
    return position


# PSO Function
def particle_swarm_optimization(
        target_function,
        swarm_size=3,
        min_values=(-5, -5),
        max_values=(5, 5),
        iterations=50,
        decay=0,
        w=0.9,
        c1=2,
        c2=2,
):
    count = 0
    position = initial_position(
        target_function=target_function,
        swarm_size=swarm_size,
        min_values=min_values,
        max_values=max_values,
    )
    init_velocity = initial_velocity(
        position, min_values=min_values, max_values=max_values
    )
    i_b_matrix = np.copy(position)
    best_global = np.copy(position[position[:, -1].argsort()][0, :])
    while count <= iterations:
        print("Iteration = ", count, " f(x) = ", best_global[-1])
        position = update_position(
            target_function=target_function, position=position, velocity=init_velocity
        )
        i_b_matrix = individual_best_matrix(position, i_b_matrix)
        value = np.copy(i_b_matrix[i_b_matrix[:, -1].argsort()][0, :])
        if best_global[-1] > value[-1]:
            best_global = np.copy(value)
        if decay > 0:
            n = decay
            w = w * (1 - ((count - 1) ** n) / (iterations ** n))
            c1 = (1 - c1) * (count / iterations) + c1
            c2 = (1 - c2) * (count / iterations) + c2
        init_velocity = velocity_vector(
            position, init_velocity, i_b_matrix, best_global, w=w, c1=c1, c2=c2
        )
        count = count + 1
    print(best_global)
    return best_global

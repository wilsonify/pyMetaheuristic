############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Random Search

# Citation:
# PEREIRA, V. (2018). Project: Metaheuristic-Random_Search, File: Python-MH-Random Search.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Random_Search>

############################################################################

import os
import random

# Required Libraries
import numpy as np


# Function: Initialize Variables
def initial_position(
        target_function, solutions=3, min_values=(-5, -5), max_values=(5, 5)
):
    position = np.zeros((solutions, len(min_values) + 1))
    for i in range(0, solutions):
        for j in range(0, len(min_values)):
            position[i, j] = random.uniform(min_values[j], max_values[j])
        position[i, -1] = target_function(position[i, 0: position.shape[1] - 1])
    return position


# Function: Updtade Position
def update_position(target_function, position, min_values=(-5, -5), max_values=(5, 5)):
    updated_position = np.copy(position)
    for i in range(0, updated_position.shape[0]):
        for j in range(0, len(min_values)):
            rand = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            updated_position[i, j] = np.clip(
                min_values[j] + (max_values[j] - min_values[j]) * rand,
                min_values[j],
                max_values[j],
            )
        updated_position[i, -1] = target_function(
            updated_position[i, 0: updated_position.shape[1] - 1]
        )
    return updated_position


# RS Function
def random_search(
        target_function, solutions=5, min_values=(-5, -5), max_values=(5, 5), iterations=50
):
    count = 0
    position = initial_position(
        target_function=target_function,
        solutions=solutions,
        min_values=min_values,
        max_values=max_values,
    )
    best_solution = np.copy(position[position[:, -1].argsort()][0, :])
    while count <= iterations:
        print("Iteration = ", count, " f(x) = ", best_solution[-1])
        position = update_position(
            target_function=target_function,
            position=position,
            min_values=min_values,
            max_values=max_values,
        )
        if best_solution[-1] > position[position[:, -1].argsort()][0, :][-1]:
            best_solution = np.copy(position[position[:, -1].argsort()][0, :])
        count = count + 1
    print(best_solution)
    return best_solution

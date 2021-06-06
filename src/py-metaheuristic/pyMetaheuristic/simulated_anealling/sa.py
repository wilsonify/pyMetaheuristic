"""
Simulated annealing is an optimization algoirthm for solving
unconstrained optimization problems.
The method models the physical process of heating a material and
then slowly lowering the temperature
to decrease defects, thus minimizing the system energy.
"""
############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Simulated Annealing

# Citation:
# PEREIRA, V. (2018). Project: Metaheuristic-Simulated_Annealing,
# File: Python-MH-Simulated Annealing.py,
# GitHub repository: <https://github.com/Valdecy/Metaheuristic-Simulated_Annealing>

############################################################################
import logging
import random

# Required Libraries
import numpy as np
from pyMetaheuristic import rando


def initial_guess(target_function, min_values=(-5, -5), max_values=(5, 5)):
    """
    Initialize Variables
    """
    guess = np.zeros((1, len(min_values) + 1))
    for j, min_values_j in enumerate(min_values):
        guess[0, j] = random.uniform(min_values_j, max_values[j])
    guess[0, -1] = target_function(guess[0, 0: guess.shape[1] - 1])
    return guess


def epson_vector(guess, mu=0, sigma=1):
    """
    Epson Vector
    :param guess:
    :param mu:
    :param sigma:
    :return:
    """
    epson = np.zeros((1, guess.shape[1] - 1))
    for j in range(guess.shape[1] - 1):
        epson[0, j] = float(np.random.normal(mu, sigma, 1))
    return epson


def update_solution(
        target_function, guess, epson, min_values=(-5, -5), max_values=(5, 5)
):
    """
    Update Solution

    :param target_function:
    :param guess:
    :param epson:
    :param min_values:
    :param max_values:
    :return:
    """
    updated_solution = np.copy(guess)
    for j in range(guess.shape[1] - 1):
        if guess[0, j] + epson[0, j] > max_values[j]:
            updated_solution[0, j] = random.uniform(min_values[j], max_values[j])
        elif guess[0, j] + epson[0, j] < min_values[j]:
            updated_solution[0, j] = random.uniform(min_values[j], max_values[j])
        else:
            updated_solution[0, j] = guess[0, j] + epson[0, j]
    updated_solution[0, -1] = target_function(
        updated_solution[0, 0: updated_solution.shape[1] - 1]
    )
    return updated_solution


def simulated_annealing(
        target_function,
        min_values=(-5, -5),
        max_values=(5, 5),
        mu=0,
        sigma=1,
        initial_temperature=1.0,
        temperature_iterations=1000,
        final_temperature=0.0001,
        alpha=0.9,
):
    """
    SA Function

    :param target_function:
    Target Function - It can be any function that needs to be minimize,
    However it has to have only one argument: 'variables_values'. This Argument must be a list of variables.

    :param min_values:
    :param max_values:
    :param mu:
    :param sigma:
    :param initial_temperature:
    :param temperature_iterations:
    :param final_temperature:
    :param alpha:
    :return:
    """
    guess = initial_guess(
        target_function=target_function,
        min_values=min_values,
        max_values=max_values
    )
    epson = epson_vector(guess, mu=mu, sigma=sigma)
    logging.debug(f"epson = {epson}")
    best = np.copy(guess)
    fx_best = guess[0, -1]
    temperature = float(initial_temperature)
    while temperature > final_temperature:
        for repeat in range(temperature_iterations):
            print(
                "Temperature = ",
                round(temperature, 4),
                " ; iteration = ",
                repeat,
                " ; f(x) = ",
                round(best[0, -1], 4),
            )
            fx_old = guess[0, -1]
            epson = epson_vector(guess, mu=mu, sigma=sigma)
            new_guess = update_solution(
                target_function=target_function,
                guess=guess,
                epson=epson,
                min_values=min_values,
                max_values=max_values,
            )
            fx_new = new_guess[0, -1]
            delta = fx_new - fx_old
            rand = rando()
            p_value = np.exp(-delta / temperature)
            if delta < 0 or rand <= p_value:
                guess = np.copy(new_guess)
            if fx_new < fx_best:
                fx_best = fx_new
                best = np.copy(guess)
        temperature = alpha * temperature
    print(best)
    return best

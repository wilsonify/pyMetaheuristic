############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Cross Entropy Method

# Citation:
# PEREIRA, V. (2018). Project: Metaheuristic-Cross_Entropy_Method,
# File: Python-MH-Cross Entropy Method.py,
# GitHub repository: <https://github.com/Valdecy/Metaheuristic-Cross_Entropy_Method>

############################################################################

import random

# Required Libraries
import numpy as np


# Function: Initialize Variables
def initial_guess(target_function, n=5, min_values=(-5, -5), max_values=(5, 5)):
    guess = np.zeros((n, len(min_values) + 1))
    for i in range(0, n):
        for j in range(0, len(min_values)):
            guess[i, j] = random.uniform(min_values[j], max_values[j])
        guess[i, -1] = target_function(guess[i, 0: guess.shape[1] - 1])
    return guess


# Function: Variables Mean
def guess_mean_calc(guess):
    guess_mean = np.zeros((1, guess.shape[1] - 1))
    for j in range(0, guess_mean.shape[1]):
        guess_mean[0, j] = guess[:, j].mean()
    return guess_mean


# Function: Variables Standard Deviation
def guess_std_calc(guess):
    guess_std = np.zeros((1, guess.shape[1] - 1))
    for j in range(0, guess_std.shape[1]):
        guess_std[0, j] = guess[:, j].std()
    return guess_std


# Function: Generate Samples
def generate_samples(
        target_function,
        guess,
        guess_mean,
        guess_std,
        min_values=(-5, -5),
        max_values=(5, 5),
        k_samples=2,
):
    guess_ = np.copy(guess)
    guess_sample = guess_[guess_[:, -1].argsort()]
    for i in range(k_samples, guess.shape[0]):
        for j in range(0, len(min_values)):
            guess_sample[i, j] = np.clip(
                np.random.normal(guess_mean[0, j], guess_std[0, j], 1)[0],
                min_values[j],
                max_values[j],
            )
        guess_sample[i, -1] = target_function(
            guess_sample[i, 0: guess_sample.shape[1] - 1]
        )
    return guess_sample


# Function: Update Samples
def update_distribution(guess, guess_mean, guess_std, learning_rate=0.7, k_samples=2):
    guess = guess[guess[:, -1].argsort()]
    for j in range(0, guess_mean.shape[1]):
        guess_mean[0, j] = (
                learning_rate * guess_mean[0, j]
                + (1 - learning_rate) * guess[0:k_samples, j].mean()
        )
        guess_std[0, j] = (
                learning_rate * guess_std[0, j]
                + (1 - learning_rate) * guess[0:k_samples, j].std()
        )
        if guess_std[0, j] < 0.005:
            guess_std[0, j] = 3
    return guess_mean, guess_std


# CEM Function
def cross_entropy_method(
        target_function,
        n=5,
        min_values=(-5, -5),
        max_values=(5, 5),
        iterations=1000,
        learning_rate=0.7,
        k_samples=2,
):
    guess = initial_guess(
        target_function=target_function,
        n=n,
        min_values=min_values,
        max_values=max_values,
    )
    guess_mean = guess_mean_calc(guess)
    guess_std = guess_std_calc(guess)
    best = np.copy(guess[guess[:, -1].argsort()][0, :])
    count = 0
    while count < iterations:
        print("Iteration = ", count, " f(x) = ", best[-1])
        guess = generate_samples(
            target_function=target_function,
            guess=guess,
            guess_mean=guess_mean,
            guess_std=guess_std,
            min_values=min_values,
            max_values=max_values,
            k_samples=k_samples,
        )
        guess_mean, guess_std = update_distribution(
            guess,
            guess_mean,
            guess_std,
            learning_rate=learning_rate,
            k_samples=k_samples,
        )
        if best[-1] > guess[guess[:, -1].argsort()][0, :][-1]:
            best = np.copy(guess[guess[:, -1].argsort()][0, :])
        count = count + 1
    print(best)
    return best

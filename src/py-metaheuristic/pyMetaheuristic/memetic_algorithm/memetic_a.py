"""
Inspired by both Darwinian principles of natural evolution and Dawkins' notion of a meme,
the term memetic algorithm (MA) was introduced by Pablo Moscato in his technical report in 1989
where he viewed MA as being close to a form of population-based hybrid genetic algorithm (GA)
coupled with an individual learning procedure capable of performing local refinements.

The metaphorical parallels, on the one hand, to Darwinian evolution and,
on the other hand, between memes and domain specific (local search) heuristics are
captured within memetic algorithms thus rendering a methodology that
balances well between generality and problem specificity.
This two-stage nature makes them a special case of dual-phase evolution.

In a more diverse context,
memetic algorithms are now used under various names including hybrid evolutionary algorithms,
Baldwinian evolutionary algorithms, Lamarckian evolutionary algorithms, cultural algorithms, or genetic local search.
In the context of complex optimization,
many different instantiations of memetic algorithms have been reported across a wide range of application domains,
in general, converging to high-quality solutions more efficiently than their conventional evolutionary counterparts.
"""

############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Memetic Algorithm

# Citation:
# PEREIRA, V. (2018). Project: Metaheuristic-Memetic_Algorithm,
# File: Python-MH-Memetic Algorithm.py,
# GitHub repository: <https://github.com/Valdecy/Metaheuristic-Memetic_Algorithm>

############################################################################
import math
import os
import random

import numpy as np
from pyMetaheuristic import rando


class Memetic:
    """
    using the ideas of memetics within a computational framework is called memetic computing
    the traits of universal Darwinism are more appropriately captured.
    Viewed in this perspective, MA is a more constrained notion of MC. More specifically, MA covers one area of MC,
    in particular dealing with areas of evolutionary algorithms that marry other deterministic refinement
    techniques for solving optimization problems.
    MC extends the notion of memes to cover conceptual entities of knowledge-enhanced procedures or representations.
    """

    def __init__(
            self,
            target_function,
            population_size=5,
            mutation_rate=0.1,
            elite=0,
            min_values=(-5, -5),
            max_values=(5, 5),
            eta=1,
            mu=1,
            std=0.1,
            generations=50,
    ):
        """
        :param target_function:
            # Target Function - It can be any function that needs to be minimize,
            However it has to have only one argument: 'variables_values'.
            This Argument must be a list of variables.

        :param population_size:
        :param mutation_rate:
        :param elite:
        :param min_values:
        :param max_values:
        :param eta:
        :param mu:
        :param std:
        :param generations:
        :return:
        """
        self.target_function = target_function
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite = elite
        self.min_values = min_values
        self.max_values = max_values
        self.eta = eta
        self.mu = mu
        self.std = std
        self.generations = generations
        self.population = np.zeros((self.population_size, len(self.min_values) + 1))
        self.fitness = np.zeros((self.population.shape[0], 2))
        self.offspring_xhc = np.zeros((2, len(self.min_values) + 1))
        self.b_offspring = 0

        self.offspring = self.breeding()

    def initial_population(self):
        """initialize population"""

        for i in range(self.population_size):
            for j in range(len(self.min_values)):
                self.population[i, j] = random.uniform(self.min_values[j], self.max_values[j])
            self.population[i, -1] = self.target_function(self.population[i, 0: self.population.shape[1] - 1])
        return self.population

    def fitness_function(self):
        """Fitness"""
        for i in range(self.fitness.shape[0]):
            self.fitness[i, 0] = 1 / (1 + self.population[i, -1] + abs(self.population[:, -1].min()))
        fit_sum = self.fitness[:, 0].sum()
        self.fitness[0, 1] = self.fitness[0, 0]
        for i in range(1, self.fitness.shape[0]):
            self.fitness[i, 1] = self.fitness[i, 0] + self.fitness[i - 1, 1]
        for i in range(self.fitness.shape[0]):
            self.fitness[i, 1] = self.fitness[i, 1] / fit_sum
        return self.fitness

    # Function: Selection
    def roulette_wheel(self):
        """Selection"""
        ix = 0
        _random = rando()
        for i in range(self.fitness.shape[0]):
            if _random <= self.fitness[i, 1]:
                ix = i
                break
        return ix

    def breeding(self):
        """Offspring"""
        offspring = np.copy(self.population)
        b_offspring = 0
        if self.elite > 0:
            preserve = np.copy(self.population[self.population[:, -1].argsort()])
            for i in range(self.elite):
                for j in range(offspring.shape[1]):
                    offspring[i, j] = preserve[i, j]
        for i in range(self.elite, offspring.shape[0]):
            parent_1, parent_2 = self.roulette_wheel(), self.roulette_wheel()
            while parent_1 == parent_2:
                parent_2 = np.random.choice(range(len(self.population) - 1), 1)[0]
            for j in range(offspring.shape[1] - 1):
                rand = rando()
                rand_b = rando()
                if rand <= 0.5:
                    b_offspring = 2 * rand_b
                    b_offspring = b_offspring ** (1 / (self.mu + 1))
                elif rand > 0.5:
                    b_offspring = 1 / (2 * (1 - rand_b))
                    b_offspring = b_offspring ** (1 / (self.mu + 1))
                offspring[i, j] = np.clip(
                    (
                            (1 + b_offspring) * self.population[parent_1, j]
                            + (1 - b_offspring) * self.population[parent_2, j]
                    )
                    / 2,
                    self.min_values[j],
                    self.max_values[j],
                )
                if i < self.population.shape[0] - 1:
                    offspring[i + 1, j] = np.clip(
                        (
                                (1 - b_offspring) * self.population[parent_1, j]
                                + (1 + b_offspring) * self.population[parent_2, j]
                        )
                        / 2,
                        self.min_values[j],
                        self.max_values[j],
                    )
            offspring[i, -1] = self.target_function(offspring[i, 0: offspring.shape[1] - 1])
        return offspring

    def xhc(self):
        """Crossover Hill Clibing"""

        for _ in range(self.offspring.shape[0]):
            parent_1, parent_2 = self.roulette_wheel(), self.roulette_wheel()
            while parent_1 == parent_2:
                parent_2 = np.random.choice(range(len(self.offspring) - 1), 1)[0]
            for j in range(self.offspring.shape[1] - 1):
                rand = rando()
                rand_b = rando()
                b_offspring = 1 / (2 * (1 - rand_b))
                if rand <= 0.5:
                    b_offspring = 2 * rand_b
                b_offspring = b_offspring ** (1 / (self.mu + 1))
                self.offspring_xhc[0, j] = np.clip(
                    (
                            (1 + b_offspring) * self.offspring[parent_1, j]
                            + (1 - b_offspring) * self.offspring[parent_2, j]
                    )
                    / 2,
                    self.min_values[j],
                    self.max_values[j],
                )
                self.offspring_xhc[1, j] = np.clip(
                    (
                            (1 - b_offspring) * self.offspring[parent_1, j]
                            + (1 + b_offspring) * self.offspring[parent_2, j]
                    )
                    / 2,
                    self.min_values[j],
                    self.max_values[j],
                )
            self.offspring_xhc[0, -1] = self.target_function(
                self.offspring_xhc[0, 0: self.offspring_xhc.shape[1] - 1]
            )
            self.offspring_xhc[1, -1] = self.target_function(
                self.offspring_xhc[1, 0: self.offspring_xhc.shape[1] - 1]
            )
            xhc1_less_xhc0 = self.offspring_xhc[1, -1] < self.offspring_xhc[0, -1]
            if xhc1_less_xhc0:
                for k in range(self.offspring.shape[1]):
                    self.offspring_xhc[0, k] = self.offspring_xhc[1, k]
            p1_less_2 = self.offspring[parent_1, -1] < self.offspring[parent_2, -1]
            p2_less_p1 = (self.offspring[parent_2, -1] < self.offspring[parent_1, -1])

            xhc0_less_p1 = self.offspring_xhc[0, -1] < self.offspring[parent_1, -1]
            xhc0_less_p2 = (self.offspring_xhc[0, -1] < self.offspring[parent_2, -1])
            if p1_less_2 and xhc0_less_p1:
                for k in range(self.offspring.shape[1]):
                    self.offspring[parent_1, k] = self.offspring_xhc[0, k]
            elif p2_less_p1 and xhc0_less_p2:
                for k in range(self.offspring.shape[1]):
                    self.offspring[parent_2, k] = self.offspring_xhc[0, k]
        return self.offspring

    def mutation(self):
        """
        Function: Mutation
        """
        d_mutation = 0
        for i in range(self.offspring.shape[0]):
            for j in range(self.offspring.shape[1] - 1):
                probability = int.from_bytes(os.urandom(8), byteorder="big") / (
                        (1 << 64) - 1
                )
                if probability < self.mutation_rate:
                    rand = rando()
                    rand_d = int.from_bytes(os.urandom(8), byteorder="big") / (
                            (1 << 64) - 1
                    )
                    if rand <= 0.5:
                        d_mutation = 2 * rand_d
                        d_mutation = d_mutation ** (1 / (self.eta + 1)) - 1
                    elif rand > 0.5:
                        d_mutation = 2 * (1 - rand_d)
                        d_mutation = 1 - d_mutation ** (1 / (self.eta + 1))
                    self.offspring[i, j] = np.clip(
                        (self.offspring[i, j] + d_mutation), self.min_values[j], self.max_values[j]
                    )
            self.offspring[i, -1] = self.target_function(self.offspring[i, 0: self.offspring.shape[1] - 1])
        return self.offspring

    def minimize(self):
        """ minimize target function """
        count = 0
        self.population = self.initial_population()

        elite_ind = np.copy(self.population[self.population[:, -1].argsort()][0, :])
        while count <= self.generations:
            print("Generation = ", count, " f(x) = ", round(elite_ind[-1], 4))
            self.offspring = self.breeding()
            self.population = self.mutation()
            self.population = self.xhc()
            if (self.population[:, 0: self.population.shape[1] - 1].std()) / len(self.min_values) < self.std:
                print("Reinitializing Population")
                self.population = self.initial_population()
            self.fitness = self.fitness_function()
            if elite_ind[-1] > self.population[self.population[:, -1].argsort()][0, :][-1]:
                elite_ind = np.copy(self.population[self.population[:, -1].argsort()][0, :])
            count = count + 1
        print(elite_ind)
        return elite_ind

    def plot_target(self, front):
        from matplotlib import pyplot as plt
        # Target Function - Values
        front_1 = front[:, 0]
        front_2 = front[:, 1]
        func_1_values = front[:, -1]

        # Target Function - Plot
        plt.style.use("bmh")
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("$x_1$", fontsize=25, labelpad=20)
        ax.set_ylabel("$x_2$", fontsize=25, labelpad=20)
        ax.set_zlabel("$f(x_1, x_2)$", fontsize=25, labelpad=20)
        ax.scatter(front_1, front_2, func_1_values, c=func_1_values, s=50, alpha=0.3)
        ax.scatter(
            math.pi, math.pi, -1, c="red", s=100, alpha=1, edgecolors="k", marker="o"
        )
        ax.text(
            math.pi - 1.0,
            math.pi - 1.5,
            -1,
            "$x_1 = $" + str(round(math.pi, 2)) + " ; $x_2 = $" + str(round(math.pi, 2)),
            size=15,
            zorder=1,
            color="k",
        )
        ax.text(
            math.pi + 0.5,
            math.pi - 2.5,
            -1,
            "$f(x_1;x_2) = $" + str(-1),
            size=15,
            zorder=1,
            color="k",
        )
        plt.savefig(f"{os.path.basename(__file__)}.png")

    def plot_solution(self, front_1, front_2, func_1_values, variables, minimum):
        from matplotlib import pyplot as plt
        # MA - Plot Solution
        plt.style.use("bmh")
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("$x_1$", fontsize=25, labelpad=20)
        ax.set_ylabel("$x_2$", fontsize=25, labelpad=20)
        ax.set_zlabel("$f(x_1, x_2)$", fontsize=25, labelpad=20)
        ax.scatter(front_1, front_2, func_1_values, c=func_1_values, s=50, alpha=0.3)
        ax.scatter(
            variables[0],
            variables[1],
            minimum,
            c="b",
            s=150,
            alpha=1,
            edgecolors="k",
            marker="s",
        )
        ax.text(
            math.pi - 1.0,
            math.pi - 1.5,
            -1,
            "$x_1 = $"
            + str(round(variables[0], 2))
            + " ; $x_2 = $"
            + str(round(variables[1], 2)),
            size=15,
            zorder=1,
            color="k",
        )
        ax.text(
            math.pi + 0.5,
            math.pi - 2.5,
            -1,
            "$f(x_1;x_2) = $" + str(round(minimum, 4)),
            size=15,
            zorder=1,
            color="k",
        )
        plt.savefig(f"{os.path.basename(__file__)}.png")

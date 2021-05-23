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

import numpy as np


class WOA:
    """
    Whale Optimization Algorithm (WOA)
    This algorithm includes three operators to simulate
    the bubble-net foraging behavior of humpback whales
     * searching
     * encircling
     * attacking
    """

    def __init__(
        self,
        target_function,
        hunting_party=5,
        spiral_param=1,
        min_values=(-5, -5),
        max_values=(5, 5),
        iterations=50,
    ):
        """

        :param target_function:
        It can be any function that needs to be minimize,
        However it has to have only one argument: 'variables_values'.
        This Argument must be a list of variables.

        :param hunting_party:
        :param spiral_param:
        :param min_values:
        :param max_values:
        :param iterations:
        """
        self.target_function = target_function
        self.hunting_party = hunting_party
        self.spiral_param = spiral_param
        self.min_values = min_values
        self.max_values = max_values
        self.iterations = iterations
        self.dimension = len(min_values)
        self.leader = np.zeros((1, self.dimension + 1))
        self.position = np.zeros((self.hunting_party, self.dimension + 1))

        self.position = self.initial_position()
        self.leader = self.initial_leader_position()

    def initial_position(self):
        """Initialize Variables"""
        for i in range(self.hunting_party):
            for j, _ in enumerate(self.min_values):
                self.position[i, j] = random.uniform(
                    self.min_values[j], self.max_values[j]
                )
            self.position[i, -1] = self.target_function(
                self.position[i, 0 : self.position.shape[1] - 1]
            )
        return self.position

    def initial_leader_position(self):
        """
        Initialize Alpha
        :param dimension:
        :return:
        """
        for j in range(self.dimension):
            self.leader[0, j] = 0.0
        self.leader[0, -1] = self.target_function(
            self.leader[0, 0 : self.leader.shape[1] - 1]
        )
        return self.leader

    def update_leader(self, position):
        """
        Update Leader by Fitness
        :param position:
        :return:
        """
        for i in range(position.shape[0]):
            if self.leader[0, -1] > position[i, -1]:
                for j in range(position.shape[1]):
                    self.leader[0, j] = position[i, j]
        return self.leader

    def update_position(self, a_linear_component=2, b_linear_component=1):
        """
        Update Position
        :param a_linear_component:
        :param b_linear_component:
        :return:
        """
        for i in range(self.position.shape[0]):
            r1_leader = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            r2_leader = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            p_value = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)

            a_leader = 2 * a_linear_component * r1_leader - a_linear_component
            c_leader = 2 * r2_leader

            for j, _ in enumerate(self.min_values):
                if p_value < 0.5:
                    if abs(a_leader) >= 1:
                        rand = int.from_bytes(os.urandom(8), byteorder="big") / (
                            (1 << 64) - 1
                        )
                        rand_leader_index = math.floor(self.position.shape[0] * rand)
                        x_rand = self.position[rand_leader_index, :]
                        distance_x_rand = abs(
                            c_leader * x_rand[j] - self.position[i, j]
                        )
                        self.position[i, j] = np.clip(
                            x_rand[j] - a_leader * distance_x_rand,
                            self.min_values[j],
                            self.max_values[j],
                        )
                    elif abs(a_leader) < 1:
                        distance_leader = abs(
                            c_leader * self.leader[0, j] - self.position[i, j]
                        )
                        self.position[i, j] = np.clip(
                            self.leader[0, j] - a_leader * distance_leader,
                            self.min_values[j],
                            self.max_values[j],
                        )
                elif p_value >= 0.5:
                    distance_leader = abs(self.leader[0, j] - self.position[i, j])
                    rand = int.from_bytes(os.urandom(8), byteorder="big") / (
                        (1 << 64) - 1
                    )
                    m_param = (b_linear_component - 1) * rand + 1
                    self.position[i, j] = np.clip(
                        (
                            distance_leader
                            * math.exp(self.spiral_param * m_param)
                            * math.cos(m_param * 2 * math.pi)
                            + self.leader[0, j]
                        ),
                        self.min_values[j],
                        self.max_values[j],
                    )
            self.position[i, -1] = self.target_function(
                self.position[i, 0 : self.position.shape[1] - 1]
            )
        return self.position

    def minimize(self):
        """
        WOA minimizer
        :return:
        """
        count = 0
        while count <= self.iterations:
            print("Iteration = ", count, " f(x) = ", self.leader[0, -1])
            a_linear_component = 2 - count * (2 / self.iterations)
            b_linear_component = -1 + count * (-1 / self.iterations)
            leader = self.update_leader(self.position)
            self.position = self.update_position(
                a_linear_component=a_linear_component,
                b_linear_component=b_linear_component,
            )
            count = count + 1
        print(leader)
        return leader

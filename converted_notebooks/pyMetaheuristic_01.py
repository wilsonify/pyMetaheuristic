# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + id="D-uLyemUEytr"
# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Lesson: pyMetaheuristic - Adaptive Random Search
 
# GitHub Repository: <https://github.com/Valdecy>

# + colab={"base_uri": "https://localhost:8080/"} id="veR4VfsvJi0-" outputId="ae0e5f4c-c3b7-4c3c-8b81-49e2736d58f4"
# Build Dir
import os
os.chdir('/content')
CODE_DIR = 'code'

# Clone Github Repository
# !git clone https://github.com/Valdecy/pyMetaheuristic.git $CODE_DIR
os.chdir(f'./{CODE_DIR}')

# + id="ga2m0WWSJsEY"
# Required Libraries
import math
import matplotlib.pyplot as plt
import numpy as np

# ARS
from pyMetaheuristic.adaptive_random_search.ars import adaptive_random_search


# + id="b1Srha3tMAGL"
# Target Function - It can be any function that needs to be minimize, However it has to have only one argument: 'variables_values'. This Argument must be a list of variables.
# For Instance, suppose that our Target Function is the Easom Function (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

# Target Function: Easom Function
def easom(variables_values = [0, 0]):
    return -math.cos(variables_values[0])*math.cos(variables_values[1])*math.exp(-(variables_values[0] - math.pi)**2 - (variables_values[1] - math.pi)**2)


# + colab={"base_uri": "https://localhost:8080/", "height": 846} id="NocDHsTrOiFh" outputId="62b4bc46-a10f-4190-f3b6-015f50a70f75"
# Target Function - Values
x     = np.arange(-1, 7, 0.1)
front = np.zeros((len(x)**2, 3))
count = 0
for j in range (0, len(x)):
    for k in range (0, len(x)):
            front[count, 0] = x[j]
            front[count, 1] = x[k]
            count           = count + 1        
for i in range (0, front.shape[0]):
    front[i, 2] = easom(variables_values = [front[i, 0], front[i, 1]])
front_1       = front[:, 0]
front_2       = front[:, 1]
func_1_values = front[:,-1]

# Target Function - Plot 
plt.style.use('bmh')
fig = plt.figure(figsize = (15, 15))
ax  = fig.add_subplot(111, projection = '3d')
ax.set_xlabel('$x_1$', fontsize = 25, labelpad = 20)
ax.set_ylabel('$x_2$', fontsize = 25, labelpad = 20)
ax.set_zlabel('$f(x_1, x_2)$', fontsize = 25, labelpad = 20)
ax.scatter(front_1, front_2, func_1_values, c = func_1_values, s = 50, alpha = 0.3)
ax.scatter(math.pi, math.pi, -1, c = 'red', s = 100, alpha = 1, edgecolors = 'k', marker = 'o')
ax.text(math.pi-1.0, math.pi-1.5, -1,  '$x_1 = $' + str(round(math.pi, 2)) + ' ; $x_2 = $' + str(round(math.pi, 2)), size = 15, zorder = 1, color = 'k')
ax.text(math.pi+0.5, math.pi-2.5, -1,  '$f(x_1;x_2) = $' + str(-1), size = 15, zorder = 1, color = 'k')
plt.show()

# + id="IKeFda4WL12d"
# ARS - Parameters
n_sols = 100
minv   = [-5, -5]
maxv   = [ 5,  5]
ssf    = 0.05
f1     = 3
f2     = 1.5
iter   = 1000
lst    = 15
ipt    = 25
tgt    = easom

# + colab={"base_uri": "https://localhost:8080/"} id="rj6_0UpaL8xx" outputId="fb4bf8ff-228b-4ba1-dcc6-b3fd31b384e3"
# ARS - Parameters
ars = adaptive_random_search(solutions = n_sols, min_values = minv, max_values = maxv, step_size_factor = ssf, factor_1 = f1, factor_2 = f2, iterations = iter, large_step_threshold = lst, improvement_threshold = ipt, target_function = tgt)

# + colab={"base_uri": "https://localhost:8080/"} id="g3M5-49hWQbV" outputId="0b0c419a-c929-45c2-ce0c-4d40fea4d609"
# ARS - Solution
variables = ars[0][:-1]
minimum   = ars[0][ -1]
print('Variables: ', np.around(variables, 4) , ' Minimum Value Found: ', round(minimum, 4) )

# + colab={"base_uri": "https://localhost:8080/", "height": 846} id="YDQbMmLJW8it" outputId="daac69ea-ae3b-42d7-f8cc-fc9d3e4c6de5"
# ARS - Plot Solution
plt.style.use('bmh')
fig = plt.figure(figsize = (15, 15))
ax  = fig.add_subplot(111, projection = '3d')
ax.set_xlabel('$x_1$', fontsize = 25, labelpad = 20)
ax.set_ylabel('$x_2$', fontsize = 25, labelpad = 20)
ax.set_zlabel('$f(x_1, x_2)$', fontsize = 25, labelpad = 20)
ax.scatter(front_1, front_2, func_1_values, c = func_1_values, s = 50, alpha = 0.3)
ax.scatter(variables[0], variables[1], minimum, c = 'b', s = 150, alpha = 1, edgecolors = 'k', marker = 's')
ax.text(math.pi-1.0, math.pi-1.5, -1,  '$x_1 = $' + str(round(variables[0], 2)) + ' ; $x_2 = $' + str(round(variables[1], 2)), size = 15, zorder = 1, color = 'k')
ax.text(math.pi+0.5, math.pi-2.5, -1,  '$f(x_1;x_2) = $' + str(round(minimum, 4)), size = 15, zorder = 1, color = 'k')
plt.show()

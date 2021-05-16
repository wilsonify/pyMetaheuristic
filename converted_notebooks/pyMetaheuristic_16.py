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

# + id="aaas55nx6CuV"
# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Lesson: pyMetaheuristic - Random Search
 
# GitHub Repository: <https://github.com/Valdecy>

# + colab={"base_uri": "https://localhost:8080/"} id="9R0Nd7xz6H1x" outputId="acdc76e8-89ea-4310-d54b-7d00b0321dde"
# Build Dir
import os
os.chdir('/content')
CODE_DIR = 'code'

# Clone Github Repository
# !git clone https://github.com/Valdecy/pyMetaheuristic.git $CODE_DIR
os.chdir(f'./{CODE_DIR}')

# + id="pi8elZCN6Lqp"
# Required Libraries
import math
import matplotlib.pyplot as plt
import numpy as np

# RS
from pyMetaheuristic.random_search.random_s import random_search


# + id="9CBkyhh26gmk"
# Target Function - It can be any function that needs to be minimize, However it has to have only one argument: 'variables_values'. This Argument must be a list of variables.
# For Instance, suppose that our Target Function is the Easom Function (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

# Target Function: Easom Function
def easom(variables_values = [0, 0]):
    return -math.cos(variables_values[0])*math.cos(variables_values[1])*math.exp(-(variables_values[0] - math.pi)**2 - (variables_values[1] - math.pi)**2)


# + colab={"base_uri": "https://localhost:8080/", "height": 846} id="MD6PZa0q6jaE" outputId="95e2d786-aee3-4df6-b178-a4dd00368f1b"
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

# + id="J0rccYw96kKX"
# RS - Parameters
n_sols = 150
minv   = [-5, -5]
maxv   = [ 5,  5]
iter   = 1000
tgt    = easom

# + colab={"base_uri": "https://localhost:8080/"} id="b-ivrb886a5R" outputId="083d6104-2185-495a-95ac-c63a051a362b"
# RS - Algorithm
rs = random_search(solutions = n_sols, min_values = minv, max_values = maxv, iterations = iter, target_function = tgt)

# + colab={"base_uri": "https://localhost:8080/"} id="FOVQAI3g6m0V" outputId="a2905eb7-d386-4b74-a1fa-aa5a804a3184"
# RS - Solution
variables = rs[:-1]
minimum   = rs[ -1]
print('Variables: ', np.around(variables, 4) , ' Minimum Value Found: ', round(minimum, 4) )

# + colab={"base_uri": "https://localhost:8080/", "height": 846} id="O2cY-PZ46pYJ" outputId="3a96ed17-926e-49eb-8daf-8783c4a2f430"
# RS - Plot Solution
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

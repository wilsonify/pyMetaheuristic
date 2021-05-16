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

# + id="M7Adm3wVqHEt"
# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Lesson: pyMetaheuristic - Differential Evolution 
 
# GitHub Repository: <https://github.com/Valdecy>

# + colab={"base_uri": "https://localhost:8080/"} id="yGIKeqIqqN3q" outputId="67c84ed7-3021-46c9-b227-f2e8a563b925"
# Build Dir
import os
os.chdir('/content')
CODE_DIR = 'code'

# Clone Github Repository
# !git clone https://github.com/Valdecy/pyMetaheuristic.git $CODE_DIR
os.chdir(f'./{CODE_DIR}')

# + id="r7nLwqqrqQMD"
# Required Libraries
import math
import matplotlib.pyplot as plt
import numpy as np

# DE
from pyMetaheuristic.differential_evolution.de import differential_evolution


# + id="ARGAtLpmqfnd"
# Target Function - It can be any function that needs to be minimize, However it has to have only one argument: 'variables_values'. This Argument must be a list of variables.
# For Instance, suppose that our Target Function is the Easom Function (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

# Target Function: Easom Function
def easom(variables_values = [0, 0]):
    return -math.cos(variables_values[0])*math.cos(variables_values[1])*math.exp(-(variables_values[0] - math.pi)**2 - (variables_values[1] - math.pi)**2)


# + colab={"base_uri": "https://localhost:8080/", "height": 846} id="6qvfLMDKqiLa" outputId="0c2bcbf8-ab2e-4292-893a-ecfea98acbbf"
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

# + id="hBdHJ6rCqjAy"
# DE - Parameters
n_sols = 500
minv   = [-5, -5]
maxv   = [ 5,  5]
iter   = 100
par_f  = 0.9
par_c  = 0.2
tgt    = easom

# + colab={"base_uri": "https://localhost:8080/"} id="joBnErKMqn-Q" outputId="f9cf510b-28e8-4076-affc-30ebc298f258"
# DE - Algorithm
de = differential_evolution(n = n_sols, min_values = minv, max_values = maxv, iterations = iter, F = par_f, Cr = par_c, target_function = tgt)

# + colab={"base_uri": "https://localhost:8080/"} id="_wK_5hGZrEGn" outputId="616b2f1a-7d94-4f5d-803c-1293295fc90d"
# DE - Solution
variables = de[:-1]
minimum   = de[ -1]
print('Variables: ', np.around(variables, 4) , ' Minimum Value Found: ', round(minimum, 4) )

# + colab={"base_uri": "https://localhost:8080/", "height": 846} id="CLgutP_PrIbw" outputId="50fe1b41-eb2f-44fc-a62d-ade60a361245"
# DE - Plot Solution
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

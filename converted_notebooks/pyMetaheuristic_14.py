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

# + id="J1bIOOTe3Pbi"
# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Lesson: pyMetaheuristic - Multiverse Optimizer
 
# GitHub Repository: <https://github.com/Valdecy>

# + colab={"base_uri": "https://localhost:8080/"} id="jjaWhdjr3Vqq" outputId="865714d7-f902-4221-9c50-5f0f1b29c8c2"
# Build Dir
import os
os.chdir('/content')
CODE_DIR = 'code'

# Clone Github Repository
# !git clone https://github.com/Valdecy/pyMetaheuristic.git $CODE_DIR
os.chdir(f'./{CODE_DIR}')

# + id="sZdYjKEZ3YMA"
# Required Libraries
import math
import matplotlib.pyplot as plt
import numpy as np

# MVO
from pyMetaheuristic.multiverse_optimizer.mvo import  muti_verse_optimizer


# + id="bUIZqQbl3str"
# Target Function - It can be any function that needs to be minimize, However it has to have only one argument: 'variables_values'. This Argument must be a list of variables.
# For Instance, suppose that our Target Function is the Easom Function (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

# Target Function: Easom Function
def easom(variables_values = [0, 0]):
    return -math.cos(variables_values[0])*math.cos(variables_values[1])*math.exp(-(variables_values[0] - math.pi)**2 - (variables_values[1] - math.pi)**2)


# + colab={"base_uri": "https://localhost:8080/", "height": 846} id="NFao-tmK3vcQ" outputId="1d42b482-8ce5-4dad-e379-6ef26d561ec6"
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

# + id="0SC_E4Wu3zFe"
# MVO - Parameters
ss   = 50
minv = [-5, -5]
maxv = [ 5,  5]
iter = 100
tgt  = easom

# + colab={"base_uri": "https://localhost:8080/"} id="N-UFd9aZ3l0X" outputId="51f0b2d8-d026-4d02-8a0b-bb35e4daf7e8"
# MVO - Algorithm
mvo = muti_verse_optimizer(universes = ss, min_values = minv, max_values = maxv, iterations = iter, target_function = tgt)

# + colab={"base_uri": "https://localhost:8080/"} id="_ZM4sOXg32YD" outputId="2ffb232e-b8d3-47e1-a56d-09084b8a5b14"
# MVO - Solution
variables = mvo[:-1]
minimum   = mvo[ -1]
print('Variables: ', np.around(variables, 4) , ' Minimum Value Found: ', round(minimum, 4) )

# + colab={"base_uri": "https://localhost:8080/", "height": 846} id="SD5yuSqq38HV" outputId="3a36d827-23d1-42d7-c4a4-8389f0c1401d"
# MVO - Plot Solution
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

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 23:13:03 2023

@author: shinle
"""

import matplotlib.pyplot as plt
import numpy as np

# Define the function for the curve to be approximated
def f(x, 𝛃0, 𝛃1, 𝛃2, 𝛃3):
    return 𝛃0*x**3 + 𝛃1*x**2 + 𝛃2*x +𝛃3

# Define the design matrix for the linear model
def design_matrix(x):
    x = np.array(x)
    X = np.column_stack((x**3, x**2, x , np.ones_like(x)))
    return X

# Define the data
'''
data = [(4, 1.58), (6, 2.08), (8, 2.5), (10, 2.8),
        (12, 3.1), (14, 3.4), (16, 3.8), (18, 4.32),
        (20, 4.8),(22, 5.2),(24, 5.6),(26, 6.1),(28, 6.6),
        (30, 7.1),(32, 7.7),(34, 8.4),(36, 9.1),(38, 10)]

data = [(2, 1.2),(4, 1.58),(6, 2.08),(8, 2.5),(10, 2.8),(12, 3.1),(14, 3.4),
(16, 3.8),(18, 4.32),(20, 5),(22, 6),(24, 7.2),(26, 8.8),(28, 10.6),
(30, 12.8),(32, 15.4),(34, 18.4),(36, 21.8),(38, 25.6),(40, 29.8)  ]

data=[(-4, 2),(-2, 1.5),(0, 0),(2, 1.5),(4, 2.5),(6, 3),(8, 3.5),(10, 4),(12, 4.5),
(14, 4.8),(16, 5),(18, 5.1),(20, 5.2),(22, 5.15),(24, 5.1),(26, 5.05),(28, 5),
(30, 4.95),(32, 4.9),(34, 4.85),(36, 4.8),(38, 4.75)]
'''
#data= [(1, 0.05), (2, 0.1), (3, 0.25), (4, 0.5), (5, 0.1), (10, 0.25), (15, 0.5), (20, 0.75), (25, 0.9), (30, 0.95), (35, 0.98), (40, 0.99), (45, 0.995), (50, 0.997), (55, 0.998), (60, 0.999)]
#data= [(5, 0.1), (10, 0.25), (15, 0.5), (20, 0.75), (25, 0.9), (30, 0.95), (35, 0.97), (40, 0.98), (45, 0.99), (50, 0.995)]
#data= [(5, 0.1), (10, 0.25), (15, 0.5), (20, 0.75)]
data= [(1, 0.05), (2, 0.1), (3, 0.25), (4, 0.5), (5, 0.1),(6, 0.12),(7, 0.2),(8, 0.19),(9, 0.11), (10, 0.25), (15, 0.5), (20, 0.75), (25, 0.9), (30, 0.95), (35, 0.98), (40, 0.99), (45, 0.995), (50, 0.997), (55, 0.998), (60, 0.999)]



x = [point[0] for point in data]
y = [point[1] for point in data]

# Print the dataset
for i in range(len(data)):
    print("(", x[i], ",", y[i], ")")

# Print the design matrix with float entries
np.set_printoptions(precision=2, suppress=True)
A = design_matrix(x)
print("\nX= \n",A)

# Define the modified Gram-Schmidt process to compute orthonormal basis
def gram_schmidt_process(A):
    # Convert the input array to a float array
    A = np.array(A, dtype=float)

    # Initialize Q and R
    Q = np.zeros_like(A)
    R = np.zeros((A.shape[1], A.shape[1]))

    # Iterate over basis vectors
    for j in range(A.shape[1]):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    print("\nR =\n",R)
    print("\nQ = \n",Q)
    return Q, R


# Compute the least-squares solution
Q, R = gram_schmidt_process(A)
R_inv = np.linalg.inv(R)
Q_T = Q.T
params = np.dot(np.dot(R_inv, Q_T), y)

# Print the parameters
print('parameters are:',params)
print(len(params))

# Generate the best fit curve
x_plot = np.linspace(min(x), max(x), 100)
y_plot = f(x_plot, *params)

# Plot the data points and the best fit curve
plt.scatter(x, y, color='blue')
plt.plot(x_plot, y_plot, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Best Fit Curve')
plt.show()

'''

import numpy as np
import matplotlib.pyplot as plt

# Define the input and output variables
x = np.array([1, 2, 4, 5])
y = np.array([0, 1, 2, 3])

# Calculate the coefficients of the line of best fit
n = len(x)
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_x_squared = np.sum(x**2)

a = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x**2)
b = (sum_y - a * sum_x) / n

# Print the equation of the line of best fit
eq = f"y = {a:.2f}x + {b:.2f}"
print(f"The equation of the line of best fit is {eq}")

# Plot the data points and the line of best fit
plt.scatter(x, y, color='blue')
plt.plot(x, a * x + b, color='red', label='Line of Best Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear')
plt.text(2, 2.2, eq, fontsize=12, color='green')
plt.legend()
plt.show()

'''












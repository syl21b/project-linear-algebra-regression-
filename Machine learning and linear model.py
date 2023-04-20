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
data = [(1, 0.0179),(2, 0.0279), (3, 0.0305), (4, 0.0401), (5, 0.0525), (6, 0.0555), (7.0, 0.0647), (8, 0.0702), (9, 0.0803), (10, 0.0908)]

#data=[(1, 0.0179),(2, 0.0279), (3, 0.0305), (4, 0.0401), (5, 0.0545), (6, 0.0555), (7.0, 0.0647), (8, 0.0702), (9, 0.0803), (10, 0.0908), (11.89, 0.1163), (13.19, 0.1732), (13.74, 0.1768), (14.43, 0.2009), (15.7, 0.2446), (15.7, 0.2462), (16.06, 0.2578), (16.94, 0.2798), (18.75, 0.3206), (19.92, 0.3578), (20.18, 0.3711), (20.29, 0.3749), (22.93, 0.4541), (23.63, 0.3934), (24.13, 0.4261), (26.01, 0.2501), (26.13, 0.2501), (26.64, 0.2762), (27.06, 0.2831), (27.1, 0.2831), (28.35, 0.3086), (28.61, 0.3159), (28.94, 0.3195), (29.2, 0.3249), (29.47, 0.3354), (30.42, 0.3553), (30.63, 0.3714), (31.14, 0.3855), (33.14, 0.4586), (35.26, 0.5156), (35.3, 0.5159), (36.08, 0.5395), (36.33, 0.5488), (37.1, 0.5806), (37.34, 0.5861), (37.84, 0.6084), (38.04, 0.5743), (38.31, 0.6082), (39.23, 0.6397), (42.52, 0.7056), (43.54, 0.6923), (44.19, 0.7562), (44.4, 0.7616), (44.44, 0.7643), (49.02, 0.9686), (50.33, 0.9902), (51.06, 0.9955), (51.64, 0.9984), (52.15, 0.9966), (52.91, 0.998), (52.99, 0.9981), (56.32, 0.9988), (57.23, 0.9992), (60, 0.9993)]

#data = [(1.00, 0.31), (1.17, 0.25), (1.33, 0.22), (1.50, 0.19), (1.67, 0.17), (1.83, 0.15), (2.00, 0.13), (2.17, 0.12), (2.33, 0.10), (2.50, 0.09), (2.67, 0.08), (2.83, 0.07), (3.00, 0.06), (3.17, 0.06), (3.33, 0.05), (3.50, 0.04), (3.67, 0.04), (3.83, 0.04), (4.00, 0.03), (4.17, 0.03), (4.33, 0.03), (4.50, 0.03), (4.67, 0.02), (4.83, 0.02), (5.00, 0.02), (5.17, 0.02), (5.33, 0.02), (5.50, 0.01), (5.67, 0.01), (5.83, 0.01), (6.00, 0.01), (6.17, 0.01), (6.33, 0.01), (6.50, 0.01), (6.67, 0.01), (6.83, 0.01), (7.00, 0.01), (7.17, 0.01), (7.33, 0.01), (7.50, 0.00), (7.67, 0.00), (7.83, 0.00), (8.00, 0.00), (8.17, 0.00), (8.33, 0.00), (8.50, 0.00), (8.67, 0.00), (8.83, 0.00), (9.00, 0.00), (9.17, 0.00), (9.33, 0.00), (9.50, 0.00), (9.67, 0.00), (9.83, 0.00), (10.00, 0.00), (10.17, 0.00), (10.33, 0.00), (10.50, 0.00), (10.67, 0.00), (10.83, 0.00)]

#data= [(1, 0.05), (2, 0.1), (3, 0.25), (4, 0.5), (5, 0.1),(6, 0.12),(7, 0.2),(8, 0.19),(9, 0.11), (10, 0.25), (15, 0.5), (20, 0.75), (25, 0.9), (30, 0.95), (35, 0.98), (40, 0.99), (45, 0.995), (50, 0.997), (55, 0.998), (60, 0.999)]


x = [point[0] for point in data]
y = [point[1] for point in data]

# Print the dataset
for i in range(len(data)):
    print("(", x[i], ",", y[i], ")")

# Print the design matrix with float entries
np.set_printoptions(precision=4, suppress=True)         #rounded four decimal places
X = design_matrix(x)
print("\nX= \n",X)

# Define the modified Gram-Schmidt process to compute orthonormal basis
def gram_schmidt_process(X):
    # Convert the input array to a float array
    X = np.array(X, dtype=float)

    # Initialize Q and R
    Q = np.zeros_like(X)
    R = np.zeros((X.shape[1], X.shape[1]))

    # Iterate over basis vectors
    for j in range(X.shape[1]):
        v = X[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], X[:, j])
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    print("\nR =\n",R)
    print("\nQ = \n",Q)
    return Q, R


np.set_printoptions(precision=4, suppress=True)

# Compute the least-squares solution
Q, R = gram_schmidt_process(X)
R_inv = np.linalg.inv(R)
Q_T = Q.T


params = np.dot(np.dot(R_inv, Q_T), y)

# Print the parameters
print('coefficient are:',params)
#print(len(params))

# Generate the best fit curve
x_plot = np.linspace(min(x), max(x), 100)
y_plot = f(x_plot, *params)

# Plot the data points and the best fit curve
eq = r'$y = {:.4f}x^3 + {:.4f}x^2 + {:.4f}x + {:.4f}$'.format(*params)

# Plot the data points and the best fit curve
plt.scatter(x, y, color='blue')
plt.plot(x_plot, y_plot, color='red')
plt.xlabel('minutes')
plt.ylabel('probability')
plt.title('Best Fit Curve')
plt.text(min(x), max(y), eq, fontsize=12, color='green')
plt.show()

params = np.round(params, decimals=4)
x=18
prob = params[0]*x**3 + params[1]*x**2 + params[2]*x + params[3]
print(f"Probability that a customer will make a purchase after spending {x} minutes on the website is: {prob *100} %")











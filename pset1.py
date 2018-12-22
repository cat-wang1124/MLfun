"""
Problem Set 1
Wesley Xiao     A14674605
    problems 7 - 12

Catherine Wang  A14394510
    problems 1 - 6
Collaborated together on all of the problems

10/12/18
"""

import numpy as np
import matplotlib.pyplot as plt

# problem 1
def compute_slope_estimator(x, y):
    avg_x = np.mean(x, dtype = np.float64)
    avg_y = np.mean(y, dtype = np.float64)
    n = x.size

    sum_products = np.sum(np.dot(x, y.T), dtype = np.float64)
    sum_x_squared = np.sum(np.dot(x, x.T), dtype = np.float64)

    a = (sum_products - (n * avg_x * avg_y)) / (sum_x_squared - (n * avg_x**2))
    return a

# problem 2
def compute_intercept_estimator(x, y):
    avg_x = np.mean(x)
    avg_y = np.mean(y)
    a = compute_slope_estimator(x, y)

    b = avg_y - (a * avg_x)
    return b

# problem 3
def train_model(x, y):
    a = compute_slope_estimator(x, y)
    b = compute_intercept_estimator(x, y)

    return (a, b)

# problem 4
def  sample_linear_model(x_vals, a, b, sd):
    sampled = []
    for x in x_vals:
        epsilon_i = np.random.normal(0, sd)
        y = a * x + b + epsilon_i
        sampled.append(y)

    return np.array(sampled)

# problem 5
def sample_datasets(x_vals, a, b, sd, n):
    total_sampled = []
    for i in range(0, n):
        total_sampled.append(sample_linear_model(x_vals, a, b, sd))

    return np.array(total_sampled)

# problem 6
def compute_average_estimated_slope(x_vals):
    sample = sample_datasets(x_vals, 1, 1, 1, 1000)

    total_a = 0
    for y_vals in sample:
        total_a += compute_slope_estimator(x_vals, y_vals)

    return total_a / len(x_vals)

# problem 7
"""
print("PROBLEM 7")
print("-----------------------------")
x_vals = np.linspace(0, 1, 5)
print("n = 5")
print("avg est slope = ", compute_average_estimated_slope(x_vals), "\n")
x_vals = np.linspace(0, 1, 100)
print("n = 100")
print("avg est slope = ", compute_average_estimated_slope(x_vals), "\n")
x_vals = np.linspace(0, 1, 1000)
print("n = 1000")
print("avg est slope = ", compute_average_estimated_slope(x_vals), "\n")
"""

# Response
"""
As n gets bigger, the average estimated slope gets siginificantly smaller/closer to 1.
When n = 5, the average estimated slope is around 200. When n = 100, the average estimated
slope is around 10. When n = 1000, the average estimated slope is around 1.
"""

# problem 8
def compute_estimated_slope_error(x_vals):
    sample = sample_datasets(x_vals, 1, 1, 1, 1000)

    total_error = 0
    for y_vals in sample:
        total_error += (1 - compute_slope_estimator(x_vals, y_vals))**2

    return total_error / len(x_vals)
"""
print("PROBLEM 8")
print("-----------------------------")
x_vals = np.linspace(0, 1, 5)
print("n = 5")
print("est slope error = ", compute_estimated_slope_error(x_vals), "\n")
x_vals = np.linspace(0, 1, 100)
print("n = 100")
print("est slope error = ", compute_estimated_slope_error(x_vals), "\n")
x_vals = np.linspace(0, 1, 1000)
print("n = 1000")
print("est slope error = ", compute_estimated_slope_error(x_vals), "\n")
"""

# Response
"""
The average squared error also get significantly smaller/closer to 0. When n = 5,
the average estimated slope error is around 350. When n = 100, the average estimated
slope error is around 1. When n = 1000, the average estimated slope error is around 0.01.
"""

# problem 9
def generate_plot(x_vals):
    y_vals = sample_datasets(x_vals, 1, 1, 1, 1000)

    s = []
    for y in y_vals:
        a = compute_slope_estimator(x_vals, y)
        s.append(a)

    slopes = np.array(s)
    plt.hist(slopes)
    plt.ylabel("estimated slope")
    plt.show()
"""
print("PROBLEM 9")
print("-----------------------------")
x_vals = np.linspace(0, 1, 5)
print("n = 5")
generate_plot(x_vals)
x_vals = np.linspace(0, 1, 100)
print("n = 100")
generate_plot(x_vals)
x_vals = np.linspace(0, 1, 1000)
print("n = 1000")
generate_plot(x_vals)
"""

# Reponse
"""
As the number of x_vals increases, the range of slope values gets smaller: starting
from a range of about -4 to 4 and going onto a range of about 0.6 to 1.4. Also, the
standard deviation gets smaller as the number of x_vals get larger. In other words,
more of the estimated slopes are closer to the actual.
"""

# problem 10
def calculate_prediction_error(y, y_hat):
    diff = y - y_hat
    squared_sum = np.sum(np.dot(y, y_hat.T))
    return squared_sum / y.size       # avg error

# problem 11
def average_training_set_error(x_vals):
    #import pdb; pdb.set_trace()
    y_vals = sample_datasets(x_vals, 1, 1, 1, 1000)

    error = 0
    for i in range(0, 1000):
        y = y_vals[i]

        slope_intercept = train_model(x_vals, y)
        a = slope_intercept[0]
        b = slope_intercept[1]

        y_hat = []
        for x in x_vals:
            y_hat.append(a * x + b)

        error += calculate_prediction_error(y, np.array(y_hat))
    return error / 1000
"""
print("\nPROBLEM 11")
print("-----------------------------")
x_vals = np.linspace(0, 1, 5)
print("n = 5")
print("avg error = ", average_training_set_error(x_vals), "\n")
x_vals = np.linspace(0, 1, 100)
print("n = 100")
print("avg error = ", average_training_set_error(x_vals), "\n")
x_vals = np.linspace(0, 1, 1000)
print("n = 1000")
print("avg error = ", average_training_set_error(x_vals), "\n")
"""

# Response
"""
As the number of elements in x_vals increases, the average perdiction error does
not change significantly. For n=5, n=100, and n=1000, the average error stays closer
to 2, though the average error does decrease slightly as n increases: 2.78 for n=5,
2.35 for n=100, and 2.33 for n=1000
"""

# problem 12
def average_test_set_error(x_vals):
    #import pdb; pdb.set_trace()
    y_vals = sample_linear_model(x_vals, 1, 1, 1)

    slope_intercept = train_model(x_vals, y_vals)
    a = slope_intercept[0]
    b = slope_intercept[1]

    y_hat = []
    for x in x_vals:
        y_hat.append(a * x + b)

    error = calculate_prediction_error(y_vals, np.array(y_hat))
    return error / 1000

"""
print("\nPROBLEM 12")
print("-----------------------------")
x_vals = np.linspace(0, 1, 5)
print("n = 5")
print("avg test set error = ", average_test_set_error(x_vals), "\n")
x_vals = np.linspace(0, 1, 100)
print("n = 100")
print("avg test set error = ", average_test_set_error(x_vals), "\n")
x_vals = np.linspace(0, 1, 1000)
print("n = 1000")
print("avg test set error = ", average_test_set_error(x_vals), "\n")
"""

# Response
"""
The average value of the test set prediction error is about 1/1000 of the average
value of the test set prediction error found in problem 11. As the number of elements
in x_vals increases, the average perdiction error does not change significantly. For n=5,
n=100, and n=1000, the average error stays close to about 0.25, though the average error
does decrease slightly as n increases: 0.0034 for n=5, 0.0022 for n=100, and 0.0023 for
n=1000.
"""
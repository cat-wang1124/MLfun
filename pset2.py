"""
Problem Set 2
Wesley Xiao     A14674605
    problems 7, 10 - 15

Catherine Wang  A14394510
    problems 1 - 6, 8, 9
Collaborated together on all of the problems
"""

import numpy as np
import matplotlib.pyplot as plt


################################ BEGIN STARTER CODE ################################################
def sigmoid(x):
	#Numerically stable sigmoid function.
	#Taken from: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
	if x >= 0:
		z = np.exp(-x)
		return 1 / (1 + z)
	else:
		# if x is less than zero then z will be small, denom can't be
		# zero because it's 1+z.
		z = np.exp(x)
		return z / (1 + z)


def sample_logistic_distribution(x,a):
	#np.random.seed(1)
	num_samples = len(x)
	y = np.empty(num_samples)
	for i in range(num_samples):
		y[i] = np.random.binomial(1,logistic_positive_prob(x[i],a))
	return y

def create_input_values(dim,num_samples):
	#np.random.seed(100)
	x_inputs = []
	for i in range(num_samples):
		x = 10*np.random.rand(dim)-5
		x_inputs.append(x)
	return x_inputs


def create_dataset():
	x= create_input_values(2,100)
	a=np.array([10,10])
	y=sample_logistic_distribution(x,a)

	return x,y

################################ END STARTER CODE ################################################


# NOTICE: DO NOT EDIT FUNCTION SIGNATURES 
# PLEASE FILL IN FREE RESPONSE AND CODE IN THE PROVIDED SPACES


# PROBLEM 1
def logistic_positive_prob(x,a):
    # returns probability of P(y_i = 1 | x_i, a, b) w/ b=0
    return sigmoid(np.dot(x, a))

# PROBLEM 2
def logistic_derivative_per_datapoint(y_i,x_i,a,j):
	prob = logistic_positive_prob(x_i, a)
	return -(y_i - prob) * x_i[j]

# PROBLEM 3
def logistic_partial_derivative(y,x,a,j):
    # partial derivative of the loss function
    sum = 0;
    for i in range(0, len(y)):
    	sum += logistic_derivative_per_datapoint(y[i], x[i], a, j)
    return sum

# PROBLEM 4
def compute_logistic_gradient(a,y,x):
    # returns gradient: a vector of all the partial derivatives
    gradient = []
    for j in range(0, len(a)):
    	partial_derv = logistic_partial_derivative(y, x, a, j)
    	gradient.append(partial_derv)
    return np.array(gradient)

# PROBLEM 5
def gradient_update(a,lr,gradient):
    # update the values in a using the gradient and learning curve
    # relation: new_weight = old_weight - learning_rate * gradient
    # helpful website: https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10

    updated = []
    for i in range(0, len(a)):
    	updated.append(a[i] - lr * gradient[i])
    return np.array(updated)

# PROBLEM 6
def gradient_descent_logistic(initial_a,lr,num_iterations,y,x):
    # runs the gradient and gradient update algs multiple times
    new_a = initial_a
    for i in range(num_iterations):
    	gradient = compute_logistic_gradient(new_a, y, x)
    	new_a = gradient_update(new_a, lr, gradient)
    return new_a


# PROBLEM 7
# Please include your generated graphs in this zipped folder when you submit.
# Comment out your calls to matplotlib (e.g. plt.show()) before submitting
x, y = create_dataset()
x_coord = []
y_coord = []
for i in x:
	x_coord.append(i[0])
	y_coord.append(i[1])

x_coord = np.array(x_coord)
y_coord = np.array(y_coord)

plt.scatter(x_coord, y_coord, c=y)
plt.show()



# PROBLEM 8
# Free Response Answer Here: 
"""
The value for a that we got was [5.74698155 6.02726583]. When we ran it many times, we got results
that are reasonably close: [5.0137432  5.25899451], [5.51233035 5.46731376], [6.20173137 6.26133519]. 

When we start off with an initial gradient of [-1, -1], the gradient descent algorithm adjusts the gradient
1000 times, each time getting closer and closer to the lowest error [5.75, 6.027]. The updated gradients seems to us
to be converging to the same values. During the last 100 or so iterations, the intermediary gradients begin
to become very close to the final value.


"""
#x, y = create_dataset()

print("PROBLEM 8 -----------------------")
a = np.array([-1, -1])
lr = 0.01
num_iterations = 1000
a = gradient_descent_logistic(a, lr, num_iterations, y, x)
print(a)

x, y = create_dataset()
num_iterations = 1000
lr = 0.01
a = np.array([-1, -1])
a = gradient_descent_logistic(a, lr, num_iterations, y, x)
print(a)

x, y = create_dataset()
num_iterations = 1000
lr = 0.01
a = np.array([-1, -1])
a = gradient_descent_logistic(a, lr, num_iterations, y, x)
print(a)



# PROBLEM 9
# Free Response Answer Here: 
"""
When initial_a is (1, 0) or (0, -1) the updated a is close to the value of updated a from initial_a
being (-1, -1). (-1, -1) returns an a of [5.6828792  5.62023506]. (1, 0) returns an a of 
[4.90184416 5.15700184]. (0, -1) returns [6.08275106 5.78927605]. These outputs all remain in the 
5 to 6ish range. 
This suggests that the solution found by our function is reasonably optimal, telling
us that with gradient descent on logisitc regression, our starting gradient does not really
matter. Every data set has an optimal gradient to be calculated.
"""

"""
x, y = create_dataset()
a = np.array([0, -1])
lr = 0.01
num_iterations = 1000
a = gradient_descent_logistic(a, lr, num_iterations, y, x)
print("(0, -1): ", a)

x, y = create_dataset()
a = np.array([0, 0])
lr = 0.01
num_iterations = 1000
a = gradient_descent_logistic(a, lr, num_iterations, y, x)
print("(0, 0): ", a)

a = np.array([1, 0])
lr = 0.01
num_iterations = 1000
a = gradient_descent_logistic(a, lr, num_iterations, y, x)
print("(1, 0): ", a)

x, y = create_dataset()
a = np.array([0, 1])
lr = 0.01
num_iterations = 1000
a = gradient_descent_logistic(a, lr, num_iterations, y, x)
print("(0, 1): ", a)

x, y = create_dataset()
a = np.array([1, 1])
lr = 0.01
num_iterations = 1000
a = gradient_descent_logistic(a, lr, num_iterations, y, x)
print("(1, 1): ", a)
"""




# PROBLEM 10
# Free Response Answer Here: 
"""
Output (learning rate is the first, gradient is second):

0.001 [2.84413192 2.73726676]
0.01 [4.81458738 5.10468078]
0.1 [17.2894362 18.2692157]
1 [139.25899239 153.62918682]


The learning rate plays a siginificant role when using gradient descent on
logistic regression. The learning rate determines the "steps" we take when calculating the gradient
on different points of the cost function. A optimal learning rate can provide youw with a very 
good gradient; a poor learning rate can give you a bad gradient
"""

"""
print("PROBLEM 10 -----------------------")
x, y = create_dataset()
a = np.array([-1, -1])
lr = 0.001
num_iterations = 1000
a = gradient_descent_logistic(a, lr, num_iterations, y, x)
print(lr, a)

x, y = create_dataset()
a = np.array([-1, -1])
lr = 0.01
num_iterations = 1000
a = gradient_descent_logistic(a, lr, num_iterations, y, x)
print(lr, a)

x, y = create_dataset()
num_iterations = 1000
lr = 0.1
a = np.array([-1, -1])
a = gradient_descent_logistic(a, lr, num_iterations, y, x)
print(lr, a)

x, y = create_dataset()
num_iterations = 1000
lr = 1
a = np.array([-1, -1])
a = gradient_descent_logistic(a, lr, num_iterations, y, x)
print(lr, a)
"""


# PROBLEM 11
def logistic_l2_partial_derivative(y,x,a,j):
    penalty = 0.1
    sum = 0;
    for i in range(0, len(y)):
    	sum += logistic_derivative_per_datapoint(y[i], x[i], a, j) + (penalty * 2 * a[j])
    return sum

# PROBLEM 12
def compute_logistic_l2_gradient(a,y,x):
    gradient = []
    for j in range(0, len(a)):
    	partial_derv = logistic_l2_partial_derivative(y, x, a, j)
    	gradient.append(partial_derv)
    return np.array(gradient)

# PROBLEM 13
def gradient_descent(initial_a,lr,num_iterations,y,x,gradient_fn):
    new_a = initial_a
    for i in range(num_iterations):
    	gradient = gradient_fn(new_a, y, x)
    	new_a = gradient_update(new_a, lr, gradient)

    return new_a

# PROBLEM 14
# Free Response Answer Here: 
"""
output:

PROBLEM 8 -----------------------
[5.54405523 5.64540487]
[5.71469461 6.43042568]
[5.70912472 5.91490117]
PROBLEM 14 -----------------------
[0.62509774 0.65046991]
[0.64872145 0.62612489]
[0.6698015  0.63274522]

The values found in this problem are about 1/10 the size of the values found in problem 8. The 
great difference between the two values is due to the penalty term in l2-regularization. It makes
large values more siginificant, while making smaller values less siginificant, which helps prevent
overfitting.

"""

"""
print("PROBLEM 14 -----------------------")
x, y = create_dataset()
a = np.array([-1, -1])
lr = 0.01
num_iterations = 1000
a = gradient_descent(a, lr, num_iterations, y, x, compute_logistic_l2_gradient)
print(a)

x, y = create_dataset()
a = np.array([-1, -1])
lr = 0.01
num_iterations = 1000
a = gradient_descent(a, lr, num_iterations, y, x, compute_logistic_l2_gradient)
print(a)

x, y = create_dataset()
a = np.array([-1, -1])
lr = 0.01
num_iterations = 1000
a = gradient_descent(a, lr, num_iterations, y, x, compute_logistic_l2_gradient)
print(a)
"""



# PROBLEM 15
# Free Response Answer Here: 
"""
Output for penalty = 0.01:
PROBLEM 15 -----------------------
[1.5448273  1.40900727]
[1.49161112 1.43996096]
[1.4579255  1.45655418]

The output values where larger than those found in problem 14.

output for penalty = 1:
PROBLEM 15 -----------------------
[   57.53213621 -1417.19155515]
[   96.82027102 -1326.33879762]
[-955.5520208  -922.72069884]

output for penalty = 10:
PROBLEM 15 -----------------------
ps2.py:258: RuntimeWarning: overflow encountered in double_scalars
  sum += logistic_derivative_per_datapoint(y[i], x[i], a, j) + (penalty * 2 * a[j])
[nan nan]
[nan nan]
[nan nan]

As lambda gets larger, the values returned become much larger or much smaller (the absolute
values of the returned values are larger). Lambda seems to be controlling the significance of 
each of the data points. When lambda is higher, every point is considered important. When lambda
is lower, some points are given less signficiance.
"""

"""
print("PROBLEM 15 -----------------------")
x, y = create_dataset()
a = np.array([-1, -1])
lr = 0.01
num_iterations = 1000
a = gradient_descent(a, lr, num_iterations, y, x, compute_logistic_l2_gradient)
print(a)

x, y = create_dataset()
a = np.array([-1, -1])
lr = 0.01
num_iterations = 1000
a = gradient_descent(a, lr, num_iterations, y, x, compute_logistic_l2_gradient)
print(a)

x, y = create_dataset()
a = np.array([-1, -1])
lr = 0.01
num_iterations = 1000
a = gradient_descent(a, lr, num_iterations, y, x, compute_logistic_l2_gradient)
print(a)
"""
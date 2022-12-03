from numpy import exp, array, random, dot
import numpy as np

def sigmoid(x):
    return 1/(1+exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

data = np.array([[0,0,1,0],[1,1,1,1],[1,0,1,1],[0,1,1,0]])

x = data[:,0:3]
y = np.array([data[:,3]]).T

random.seed(1)
synaptic_weights = 2 * random.random((3, 1)) - 1

num_iterations = 10000

for iteration in range(num_iterations):
    
    dot_prod = dot(x,synaptic_weights)
    output = sigmoid(dot_prod)
    error = y - output
    adjustment = dot(x.T, error*sigmoid_derivative(output))
    synaptic_weights += adjustment

#testing the results
test = [1,0,0] #expected result is
result = sigmoid(dot(test,synaptic_weights))
print('Result = ',result)




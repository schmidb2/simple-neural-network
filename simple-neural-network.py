from numpy import exp, array, random, dot
import numpy as np

def sigmoid(x):
    return 1/(1+exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

data = np.array([[0,0,1,0],[1,1,1,1],[1,0,1,1],[0,1,1,0]])

x = data[:,0:3] #input values
y = np.array([data[:,3]]).T #output values

random.seed(1) #this saves the randomly generated weights so the starting
               #weights are the same everytime the program runs
synaptic_weights = 2 * random.random((3, 1)) - 1 #starting weights range between -1 and 1

num_iterations = 10000

for iteration in range(num_iterations):
    
    #input value is adjusted by weight using the dot product
    dot_prod = dot(x,synaptic_weights)
    #the values are normalized with the sigmoid function so the result is between
    #0 and 1
    output = sigmoid(dot_prod)
    #comparing the calculated output to the expected output
    error = y - output
    #the error is multiplied by the gradient of the sigmoid curve then
    #applied back to the input
    adjustment = dot(x.T, error*sigmoid_derivative(output))
    #applying our adjustment to update the synaptic weights
    synaptic_weights += adjustment

#testing the results
test = [1,0,0] #expected result is
result = sigmoid(dot(test,synaptic_weights))
print('Result = ',result)




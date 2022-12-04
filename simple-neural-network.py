from numpy import exp, array, random, dot
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

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
results_table = np.empty([num_iterations+1,7])


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
    #saving results to a table
    weight0 = synaptic_weights[0]
    delta0 = weight0 - results_table[iteration,1]
    weight1 = synaptic_weights[1]
    delta1 = weight1 - results_table[iteration,3]
    weight2 = synaptic_weights[2]
    delta2 = weight2 - results_table[iteration,5]
    results_table[iteration+1] = [iteration,weight0,delta0,weight1,delta1,weight2,delta2]

#testing the results
test = [1,0,0] #expected result is 1
result = sigmoid(dot(test,synaptic_weights))
print('Result = ',result)
print_table = results_table.tolist()
print_table[0] = ['Iteration', 'Weight 1','Delta1','Weight 2','Delta2','Weight 3','Delta3']
print(tabulate(print_table,headers='firstrow',tablefmt='fancy_grid'))

#plotting data
x_axis = results_table[:,0]
plot_weight0 = results_table[:,1]
plot_delta0 = results_table[:,2]
plot_weight1 = results_table[:,3]
plot_delta1 = results_table[:,4]
plot_weight2 = results_table[:,5]
plot_delta2 = results_table[:,6]

line1, = plt.plot(x_axis,plot_weight0,color='green')
line1.set_label('Weight 1')
line2, = plt.plot(x_axis,plot_delta0,color='green',linestyle='dashed')
line2.set_label('Delta of Weight 1')
line3, = plt.plot(x_axis,plot_weight1,color='blue')
line3.set_label('Weight 2')
line4, = plt.plot(x_axis,plot_delta1,color='blue',linestyle='dashed')
line4.set_label('Delta of Weight2')
line5, = plt.plot(x_axis,plot_weight2,color='pink')
line5.set_label('Weight 3')
line6, = plt.plot(x_axis,plot_delta2,color='pink',linestyle='dashed')
line6.set_label('Delta of Weight 3')

plt.xlabel('Number of Iterations')
plt.ylabel('Weight/Delta Values')
plt.legend()
plt.show()



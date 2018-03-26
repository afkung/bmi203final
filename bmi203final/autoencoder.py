"""
Andrew Kung
BMI203: Algorithms - W18

Using our neural network framework to make an autoencoder
"""

import numpy as np
from bmi203final import algs

"""
Training set: identity matrix
[[1,0,0,0,0,0,0,0]
 [0,1,0,0,0,0,0,0]
 [0,0,1,0,0,0,0,0]
 [0,0,0,1,0,0,0,0]
 [0,0,0,0,1,0,0,0]
 [0,0,0,0,0,1,0,0]
 [0,0,0,0,0,0,1,0]
 [0,0,0,0,0,0,0,1]]
"""

# function run_autoencoder: trains a 8x3x8 neural network to return a 8 digit input (1x1 + 7x0) exactly as the output, with 3 neurons in the hidden layer

def run_autoencoder():
	# building 8x8 identity matrix
	input = np.zeros((8,8))
	for index in range(8):
		input[index][index] = 1
	# creating network Layers, network
	hidden_layer = algs.Layer(3,8)
	output_layer = algs.Layer(8,3)
	auto8x3x8 = algs.Network(hidden_layer, output_layer)
	# training on identity matrix
	auto8x3x8.train_network(input, input, 50000, 0.5) # function syntax: input, output, iterations, learning rate
	output = auto8x3x8.feedforward(input)
	print("INPUT:")
	print(input)
	print("Hidden Layer Weights")
	print(auto8x3x8.hidden_layer.weights)
	print("Output Layer Weights")
	print(auto8x3x8.output_layer.weights)
	print("OUTPUT:")
	print(np.round(output,2))
	return np.round(output,2)
	

run_autoencoder()
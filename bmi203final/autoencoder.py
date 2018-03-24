"""
Andrew Kung
BMI203: Algorithms - F18
Last modified: 3/23/18

Using our neural network framework to make an autoencoder
"""

import numpy as np
import algs

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
	input = np.zeros((8,8))
	for index in range(8):
		input[index][index] = 1
	hidden_weights, output_weights = algs.trainNetwork(input, input, 3, 100) # function syntax: input, output, number hidden nodes, iterations
	value = algs.predictValue(input, hidden_weights, output_weights)
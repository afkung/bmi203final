"""
Andrew Kung
BMI203: Algorithms - F18
Last modified: 3/23/18

Algorithms and framework for generating, training, and running a 3-layer artificial neural network

Partially inspired by https://medium.com/technology-invention-and-more/how-to-build-a-multi-layered-neural-network-in-python-53ec3d1d326a
"""

import numpy as np
import matplotlib.pyplot as plt


# object Layer: one layer in ANN, input number of neurons in layer and number of inputs from previous layer
class Layer():
	
	# initializing object values
	def __init__(self, number_neurons, number_inputs):
		self.neurons = number_neurons
		self.inputs = number_inputs
		self.weights = 2*np.random.random([number_inputs, number_neurons])-1 # randomizing initial weights between -1 and 1
		self.activation = np.random.random([number_inputs, number_neurons]) # randomizing intial sigmoid activations between 0 and 1
		self.bias = np.random.random([number_inputs, number_neurons]) # randomizing initial biases between 0 and 1

	# function sigmoid: activation function that values to a range from 0 to 1
	# Input: float value
	# Ouput: float value from 0 to 1
	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	# function sigDeriv: derivative of sigmoid function inputs value [0,1] and outputs sigmoid derivative
	# Input: float value from 0 to 1
	# Ouput: derivative evaluation of sigmoid float value
	def sigDeriv(self, x):
		return x * (1-x)


	# function feedforward: running through network with given values
	def feedforward(self, input):
		self.weights = np.dot(input, self.weights) + self.bias
		self.activation = self.sigmoid(self.weights)
		return output

	# function backpropagate: changing values in network based on results, gradient descent
	def backpropagate(self, input, output):
		error = output - self.activation
		grad = self.sigDeriv(self.activation)
		self.weights += np.dot(input.T,error * grad)
		self.bias += np.sum(error * grad) * 0.5 # learning rate


# function trainNetwork: builds and trains a neural network with the given parameters
# input: test input, test output, # of hidden nodes, # of iterations to run
# output: weights of hidden and output layers of trained network
def trainNetwork(test_input, test_output, hidden_nodes, iterations):
	hidden_layer = Layer(hidden_nodes, len(test_input)) # building hidden layer
	output_layer = Layer(len(test_input), hidden_nodes) # building output layer
	for iter in range(iterations): # running feed-forward + back-propagation for n iterations
		hidden_forward = hidden_layer.feedforward(test_input)
		output_forward = output_layer.feedforward(hidden_forward)
		output_layer.backpropagate(hidden_forward, test_output)
		hidden_layer.backpropagate(test_input, output_layer.weights)
	return hidden_layer.weights, output_layer.weights

# sigmoid function for non-layer objects
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# function predictValue: run input through the weight functions of the hidden and output layers of a neural network
# input: input, weight matrix for hidden layer, weight matrix for output layer
# output: predicted value based on neural network
def predictValue(input, hidden_weights, output_weights):
	return sigmoid(np.dot(sigmoid(np.dot(input, hidden.weights)), output_weights))

# Function oneHot: converting DNA sequence (A,C,G,T) into binary sequence using one-hot (each DNA base = 4 binary digits with 1x1 and 3x0, order maps to base as below)
# input: DNA sequence of length L
# output: list of binary digits (4 * L long) representing sequence
# A: 1 0 0 0
# C: 0 1 0 0 
# G: 0 0 1 0
# T: 0 0 0 1
def oneHot(tf):
	binary = np.zeros(4*len(tf))
	base_dict = {'A':0,'C':1,'G':2,'T':3}
	for index in range(len(tf)):
		binary[4*index + base_dict[tf[index]]] = 1 
	return binary

# function findFP: find the false positive rate based on known positive and negative pairs
# Input: true positive threshold as float, scores for known +/- pairs as lists of floats
# Output: false positive rate from known negative pairs
def findFP(TP_threshold, pos_scores, neg_scores):
	pos_scores.sort()
	score_cutoff = pos_scores[int(TP_threshold * len(pos_scores))-1]
	# counting how many negative pairs scored above the threshold
	FP_count = 0
	for item in neg_scores:
		if item > score_cutoff:
			FP_count += 1
	return FP_count / len(neg_scores)

# function findTP: find the true positive rate based on known positive and negative pairs
# Input: false positive threshold as float, scores for known +/- pairs as lists of floats
# Output: true positive rate from known positive pairs
def findTP(FP_threshold, pos_scores, neg_scores):
	neg_scores.sort()
	score_cutoff = neg_scores[int((1-FP_threshold) * len(neg_scores))-1]
	# counting how many positive pairs scored above the threshold
	TP_count = 0
	for item in pos_scores:
		if item > score_cutoff:
			TP_count += 1
	return TP_count / len(pos_scores)

# function rocCurve: plot a ROC curve, with false positive values on the x-axis and false negative values on the y-axis
# Input: list of FP values, list of TP values, name of matrix as string
# Output: ROC plot, displayed in interface as saved as .png
def rocCurve(FP_values, TP_values, name):
	area = 0.0
	for item in TP_values:
		area += item * (1 / len(FP_values))  # area under curve estimate using rectangle method
	fig, ax = plt.subplots()
	ax.set(xlabel = "False Positives", ylabel = "True Positives", title = "ROC " + name + " Area = " + str(area))
	ax.grid()
	plt.ylim(0,1)
	plt.xlim(0,1)
	ax.plot(FP_values, TP_values)
	fig.savefig(name+"ROC.png")
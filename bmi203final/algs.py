"""
Andrew Kung
BMI203: Algorithms - W18

Algorithms and framework for generating, training, and running a 3-layer artificial neural network

Structure partially inspired by https://medium.com/technology-invention-and-more/how-to-build-a-multi-layered-neural-network-in-python-53ec3d1d326a
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# Layer object: one layer in ANN, input number of neurons in layer and number of inputs from previous layer
class Layer():
	# initializing object values
	def __init__(self, number_neurons, number_inputs):
		self.neurons = number_neurons
		self.inputs = number_inputs
#		self.weights = np.random.random([number_inputs, number_neurons]) # randomizing initial weights between 0 and 1
		self.weights = 2*np.random.random([number_inputs, number_neurons])-1 # randomizing initial weights between -1 and 1
		self.activation = self.sigmoid(self.weights) # initial activation from random weights
		self.bias = np.random.random([1, number_neurons]) # randomizing initial biases between 0 and 1

	# function sigmoid: activation function that values to a range from 0 to 1
	# Input: float value
	# Ouput: float value from 0 to 1
	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	# function sig_deriv: derivative of sigmoid function inputs value [0,1] and outputs sigmoid derivative
	# Input: float value from 0 to 1
	# Ouput: derivative evaluation of sigmoid float value
	def sig_deriv(self, x):
		return x * (1 - x)

# Network object: Neural network with two Layer components (hidden and output), functions for training network
class Network():
	def __init__(self, hidden_layer, output_layer):
		self.hidden_layer = hidden_layer
		self.output_layer = output_layer
	
	# function feedforward: running through network with given values, changing activation values of layers
	# input: training data
	# output: outputs from both layers of network at the current iteration
	def feedforward(self, test_input):
		self.hidden_layer.activation = self.hidden_layer.sigmoid(np.dot(test_input, self.hidden_layer.weights) + self.hidden_layer.bias)
		self.output_layer.activation = self.output_layer.sigmoid(np.dot(self.hidden_layer.activation, self.output_layer.weights) + self.output_layer.bias)

		return self.output_layer.activation

	# function backpropagate: changing weights in network layers based on output, applying gradient descent
	# input: test data input and output, feed forward results, learning rate
	# output: no return, but weights, activations, biases are updated
	def backpropagate(self, test_input, test_output, learning_rate):
		output_error = test_output - self.output_layer.activation # deviation from expected output
		output_grad = self.output_layer.sig_deriv(self.output_layer.activation) #gradient calculation
		output_delta = output_error * output_grad # magnitude of change

		hidden_error = np.dot(output_delta, self.output_layer.weights.T) # error associated with neurons in hidden layer
		hidden_grad = self.hidden_layer.sig_deriv(self.hidden_layer.activation) # gradient calculation
		hidden_delta = hidden_error * hidden_grad # magnitude of change

		self.hidden_layer.weights += np.dot(test_input.T, hidden_delta) * learning_rate # adjusting weights
		self.output_layer.weights += np.dot(self.hidden_layer.activation.T, output_delta) * learning_rate
		self.hidden_layer.bias += np.sum(hidden_delta) * learning_rate # adjusting biases based on sum of deltas
		self.output_layer.bias += np.sum(output_delta) * learning_rate

	# function trainNetwork: builds and trains a neural network with the given parameters
	# input: test input, test output, # of iterations to run
	# output: weights of hidden and output layers of trained network
	def train_network(self, test_input, test_output, iterations, learning_rate):
		for iter in range(iterations): # running feed-forward + back-propagation for n iterations
			self.feedforward(test_input)
			self.backpropagate(test_input, test_output, learning_rate)


# function one_hot: converting DNA sequence (A,C,G,T) into binary sequence using one-hot
# 					each DNA base = 4 binary digits with 1x1 and 3x0, order maps to base as below
# input: DNA sequence of length L
# output: list of binary digits (4 * L long) representing sequence
# A: 1 0 0 0
# C: 0 1 0 0 
# G: 0 0 1 0
# T: 0 0 0 1
def one_hot(tf):
	binary = np.zeros(4*len(tf))
	base_dict = {'A':0,'C':1,'G':2,'T':3}
	for index in range(len(tf)):
		binary[4*index + base_dict[tf[index]]] = 1 
	return binary

# function subset_data: subsetting into training data and test data
# input: list of positive samples, list of negative samples, size of desired test set, seed of RNG
# output: partitioned training and test sets
def subset_data(pos_list, neg_list, size_test, seed):
	training_input = []
	training_output = []
	test_input = []
	test_output = []
	np.random.seed(seed) # change seed to have different training vs. test sets
	test_indices = np.random.choice(len(pos_list),len(pos_list)-size_test, replace = False)
	for index in range(len(pos_list)):
		if index in test_indices: # if part of 87 randomly selected, add to training set
			training_input.append(one_hot(pos_list[index]))
			training_output.append([1])
			training_input.append(one_hot(neg_list[index]))
			training_output.append([0])
		else: # otherwise add to test set
			test_input.append(one_hot(pos_list[index]))
			test_output.append([1])
			test_input.append(one_hot(neg_list[index]))
			test_output.append([0])
	return np.asarray(training_input), np.asarray(training_output), np.asarray(test_input), np.asarray(test_output)

# function build_network: creates and trains a network with given inputs
# input: training data input and output, # of hidden neurons, iterations to run, grad descent learning rate
# output: trained Network object
def build_network(training_input, training_output, length_hidden, iterations, learning_rate):
	hidden_layer = Layer(length_hidden, len(training_input.T))
	output_layer = Layer(1, length_hidden)
	new_network = Network(hidden_layer, output_layer)
	new_network.train_network(training_input, training_output, iterations, learning_rate) # variable number of iterations
	return new_network

# function score_network: feeds values through a trained network, returns outputs
# input: trained network, test set
# output: list of scores for test set
def score_network(trained_network, test_input):
	test_scores = []
	for item in test_input:
		output_output = trained_network.feedforward(item)
		test_scores.append(float(output_output))
	return test_scores

# function plotROC: plot a ROC curve using sklearn
# input: list of binary numbers as identities (either 1-positive or 0-negative), list of corresponding values (floats)
# output: plotted ROC curve, return AUC
def plotROC(identities, values):
	fpr, tpr, thresholds = metrics.roc_curve(identities, values, pos_label = 1) # sklearn does all the work
	ROC_auc = metrics.auc(fpr, tpr)
	# plotting figure
	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % ROC_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.show()
	return ROC_auc
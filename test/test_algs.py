"""
Andrew Kung
BMI203: Algorithms - W18

Basic testing of functions
"""

from bmi203final import algs
import numpy as np

# testing one-hot encoding from k-long nucleotide sequence to 4*k-long binary
def test_onehot():
	input = "ACGT"
	output = np.array([1,0,0,0] + [0,1,0,0] + [0,0,1,0] + [0,0,0,1])
	assert np.array_equal(algs.one_hot(input), output)


# testing neural network architecture by creating an autoencoder, more documentation details in parent file
from bmi203final import autoencoder
def test_autoencoder():
	input = np.zeros((8,8)) # identity matrix
	for index in range(8):
		input[index][index] = 1
	output = autoencoder.run_autoencoder() # should return identity matrix
	for index1 in range(8):
		for index2 in range(8):
			assert (input[index1][index2] - output[index1][index2]) < 0.05 # deviates by less than 0.05
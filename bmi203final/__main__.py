"""
Andrew Kung
BMI203: Algorithms - F18
Last modified: 3/23/18

Using a 3-layer artificial neural network to identify RAP1 transcription factor binding sites

We input positive and negative test data from the local directory, train a 3-layer feed-forward ANN using the test data,
then run a sample dataset against the trained ANN, saving the output as <Predictions.txt>.
"""

import algs
import numpy as np

# input positive site list
pos_list = []
f = open('rap1-lieb-positives.txt', 'r')
for line in f:
	pos_list.append(str(line).strip())
f.close()

# input yeast genome (negative)
yeast_genome = ""
g = open('yeast-upstream-1k-negative.fa','r')
for line in g:
	if line[0] != '>':	
		yeast_genome += str(line).strip()
g.close()

# create negative file from yeast genome
neg_list_size = 137
np.random.seed(1) # change initial seed to have different negative values
yeast_length = len(yeast_genome)
neg_list = []
for counter in range(neg_list_size):
	index = np.random.randint(0,yeast_length-17) # randomly selecting 17-bp sequence from yeast genome
	neg_seq = yeast_genome[index:index+17] 
	if neg_seq not in pos_list: # making sure doesn't overlap with positive site list
		neg_list.append(neg_seq)

# subsetting the training data and test data
training_input = []
training_output = []
test_input = []
test_output = []
np.random.seed(1) # change initial seed to have different training vs. test sets
test_indices = np.random.choice(137,87, replace = False)
for index in range(137):
	if index in test_indices:
		training_input.append(algs.oneHot(pos_list[index]))
		training_output.append([1])
		training_input.append(algs.oneHot(neg_list[index]))
		training_output.append([0])
	else:
		test_input.append(algs.oneHot(pos_list[index]))
		test_output.append([1])
		test_input.append(algs.oneHot(neg_list[index]))
		test_output.append([0])

hidden_weights, output_weights = algs.trainNetwork(test_input, test_output, 100, 1000) # variable length of hidden layer, number of iterations

test_list = []
t = open('rap1-lieb-test.txt','r')
for item in t:
	test_list.append(item.strip())
t.close()

for index in range(len(test_input)):
	score = algs.predict_Value(oneHot(test_input[index]), hidden_weights, output_weights)
	if test_output[index] == 1:
		pos_scores.append(score)
	else:
		neg_scores.append(score)
algs.rocCurve(pos_scores, neg_scores, "Title Here")

w = open('Predictions.txt','w')
for item in test_list:
	w.write(item + '\t' + algs.predict_Value(oneHot(item), hidden_weights, output_weights) + '\n')
w.close()
"""
Andrew Kung
BMI203: Algorithms - W18

Using a 3-layer artificial neural network to identify RAP1 transcription factor binding sites

We input positive and negative test data from the local directory, train a 3-layer feed-forward ANN using the test data,
then run a sample dataset against the trained ANN, saving the output as <Predictions.txt>.
"""

from bmi203final import algs
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

# create negative file from yeast genome, same size as positive file
np.random.seed(1) # change seed to have different negative values
yeast_length = len(yeast_genome)
neg_list = []
while len(neg_list) < len(pos_list):
	index = np.random.randint(0,yeast_length-17) # randomly selecting 17-bp sequence from yeast genome
	neg_seq = yeast_genome[index:index+17] 
	if neg_seq not in pos_list: # making sure doesn't overlap with positive site list
		neg_list.append(neg_seq)

# splitting data into training and test sets
training_input, training_output, test_input, test_output = algs.subset_data(pos_list, neg_list, 50, 1)


# adjustable parameters for network, currently set to optimized values
length_hidden = 50
iterations = 1000
learning_rate = 0.5

# building network
rap1_network = algs.build_network(training_input, training_output, length_hidden, iterations, learning_rate)

# scoring network
test_scores = algs.score_network(rap1_network, test_input)

# plotting AUC
AUC = algs.plotROC(test_output, test_scores)


# running trained network against novel set
test_list = []
t = open('rap1-lieb-test.txt','r')
for item in t:
	test_list.append(item.strip())
t.close()

w = open('Predictions.txt','w')
for item in test_list:
	output_output = rap1_network.feedforward(algs.one_hot(item))
	w.write(item + '\t' + str(float(output_output)) + '\n')
w.close()
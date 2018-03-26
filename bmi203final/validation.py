"""
Andrew Kung
BMI203: Algorithms - W18

Functions for finding the optimal parameters for training network
"""

import algs
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

# testing different random subsets
def test_seeds(pos_list, neg_list, seeds):
	fpr_list = []
	tpr_list = []
	auc_list = []
	for seed in seeds:
		print(seed)
		training_input, training_output, test_input, test_output = algs.subset_data(pos_list, neg_list, 50, seed) #
		rap1_network = algs.build_network(training_input, training_output, 50, 1000, 0.5)
		test_scores = algs.score_network(rap1_network, test_input)
		fpr, tpr, thresholds = metrics.roc_curve(test_output, test_scores, pos_label = 1) # sklearn ROC calculations
		AUC = metrics.auc(fpr, tpr)
		fpr_list.append(fpr)
		tpr_list.append(tpr)
		auc_list.append(AUC)
		print(str(seed) + " as random seed: AUC = " + str(AUC))
	plt.figure()
	lw = 1
	plt.plot(fpr_list[0], tpr_list[0], color='orange', lw=lw, label='Subset 1 (area = %0.2f)' % auc_list[0])
	plt.plot(fpr_list[1], tpr_list[1], color='red', lw=lw, label='Subset 2 (area = %0.2f)' % auc_list[1])
	plt.plot(fpr_list[2], tpr_list[2], color='yellow', lw=lw, label='Subset 3 (area = %0.2f)' % auc_list[2])
	plt.plot(fpr_list[2], tpr_list[2], color='green', lw=lw, label='Subset 4 (area = %0.2f)' % auc_list[3])
	plt.plot(fpr_list[2], tpr_list[2], color='purple', lw=lw, label='Subset 5 (area = %0.2f)' % auc_list[4])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC: Different Test/Training Subsets')
	plt.legend(loc="lower right")
	plt.show()

# testing hidden layer size (structure/methodology same as above)
def test_hidden_size(pos_list, neg_list, hidden_sizes):
	fpr_list = []
	tpr_list = []
	auc_list = []
	training_input, training_output, test_input, test_output = algs.subset_data(pos_list, neg_list, 50, 2)
	for size in hidden_sizes:
		test_scores = []
		rap1_network = algs.build_network(training_input, training_output, size, 100, 0.5)
		test_scores = algs.score_network(rap1_network, test_input)
		fpr, tpr, thresholds = metrics.roc_curve(test_output, test_scores, pos_label = 1) # sklearn ROC calculations
		AUC = metrics.auc(fpr, tpr)
		fpr_list.append(fpr)
		tpr_list.append(tpr)
		auc_list.append(AUC)
		print(str(size) + " hidden neurons: AUC = " + str(AUC))
	plt.figure()
	lw = 1
	plt.plot(fpr_list[0], tpr_list[0], color='orange', lw=lw, label='20 hidden neurons (area = %0.2f)' % auc_list[0])
	plt.plot(fpr_list[1], tpr_list[1], color='red', lw=lw, label='40 hidden neurons (area = %0.2f)' % auc_list[1])
	plt.plot(fpr_list[2], tpr_list[2], color='yellow', lw=lw, label='60 hidden neurons (area = %0.2f)' % auc_list[2])
	plt.plot(fpr_list[3], tpr_list[3], color='green', lw=lw, label='80 hidden neurons (area = %0.2f)' % auc_list[3])
	plt.plot(fpr_list[4], tpr_list[4], color='purple', lw=lw, label='100 hidden neurons (area = %0.2f)' % auc_list[4])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC: Different Hidden Layer Sizes')
	plt.legend(loc="lower right")
	plt.show()

# testing number of iterations of feed-forward + back-propagate (structure/methodology same as above)
def test_iterations(pos_list, neg_list, iteration_lengths):
	fpr_list = []
	tpr_list = []
	auc_list = []
	training_input, training_output, test_input, test_output = algs.subset_data(pos_list, neg_list, 50, 1)
	for length in iteration_lengths:
		rap1_network = algs.build_network(training_input, training_output, 1000, length, 0.5)
		test_scores = algs.score_network(rap1_network, test_input)
		fpr, tpr, thresholds = metrics.roc_curve(test_output, test_scores, pos_label = 1) # sklearn ROC calculations
		AUC = metrics.auc(fpr, tpr)
		fpr_list.append(fpr)
		tpr_list.append(tpr)
		auc_list.append(AUC)
		print(str(length) + " iterations: AUC = " + str(AUC))
	plt.figure()
	lw = 2
	plt.plot(fpr_list[0], tpr_list[0], color='orange', lw=lw, label='500 iterations (area = %0.2f)' % auc_list[0])
	plt.plot(fpr_list[1], tpr_list[1], color='red', lw=lw, label='1000 iterations (area = %0.2f)' % auc_list[1])
	plt.plot(fpr_list[2], tpr_list[2], color='yellow', lw=lw, label='1500 iterations (area = %0.2f)' % auc_list[2])
	plt.plot(fpr_list[3], tpr_list[3], color='green', lw=lw, label='2000 iterations (area = %0.2f)' % auc_list[3])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC: Iteration Lengths')
	plt.legend(loc="lower right")
	plt.show()
	
# testing learning rate for gradient descent (structure/methodology same as above)
def test_learning_rate(pos_list, neg_list, learning_rates):
	fpr_list = []
	tpr_list = []
	auc_list = []
	training_input, training_output, test_input, test_output = algs.subset_data(pos_list, neg_list, 50, 1)

	for rate in learning_rates:
		rap1_network = algs.build_network(training_input, training_output, 100, 1000, rate)
		test_scores = algs.score_network(rap1_network, test_input)
		fpr, tpr, thresholds = metrics.roc_curve(test_output, test_scores, pos_label = 1) # sklearn ROC calculations
		AUC = metrics.auc(fpr, tpr)
		fpr_list.append(fpr)
		tpr_list.append(tpr)
		auc_list.append(AUC)
		print(str(rate) + " learning rate: AUC = " + str(AUC))
	plt.figure()
	lw = 2
	plt.plot(fpr_list[0], tpr_list[0], color='orange', lw=lw, label='Learning Rate 0.1 (area = %0.2f)' % auc_list[0])
	plt.plot(fpr_list[1], tpr_list[1], color='red', lw=lw, label='Learning Rate 0.5 (area = %0.2f)' % auc_list[1])
	plt.plot(fpr_list[2], tpr_list[2], color='yellow', lw=lw, label='Learning Rate 1.0 (area = %0.2f)' % auc_list[2])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC: Learning Rates')
	plt.legend(loc="lower right")
	plt.show()

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

test_seeds(pos_list, neg_list, [1,2,3,4,5])
test_hidden_size(pos_list, neg_list, [20,40,60,80,100])
test_iterations(pos_list, neg_list, [500,1000,1500,2000])
test_learning_rate(pos_list, neg_list, [0.1,0.5,1])
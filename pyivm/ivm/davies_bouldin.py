from sklearn.metrics import davies_bouldin_score
import numpy as np
from . import utils

def davies_bouldin_sklearn(X, label):
	return davies_bouldin_score(X, label)

def davies_bouldin_original(X, label):

	n_clusters = len(np.unique(label))
	n_samples = X.shape[0]
	n_features = X.shape[1]

	centroids = np.zeros((n_clusters, n_features))
	for i in range(n_clusters):
		centroids[i, :] = utils.centroid(X[label == i, :])
	
	sum_score = 0
	for i in range(n_clusters):
		max_score = -1
		for j in range(n_clusters):
			if i == j:
				continue
			compactness  = np.sum(np.sqrt(np.sum(np.square(X[label == i, :] - centroids[i, :]), axis=1))) / X[label == i, :].shape[0]
			compactness += np.sum(np.sqrt(np.sum(np.square(X[label == j, :] - centroids[j, :]), axis=1))) / X[label == j, :].shape[0]
			separability = np.sqrt(np.sum(np.square(centroids[i, :] - centroids[j, :])))
			curr_score = compactness / separability
			if curr_score > max_score:
				max_score = curr_score
		sum_score += max_score
	
	## note that davies bouldin praises clustering with lower score
	## should be inversed to be consistent with other metrics
	return (sum_score / n_clusters) ** (-1)


	

def davies_bouldin_exp(X, label):
	return 0

def davies_bouldin_range(X, labels, k= 1.1583908441240243):
	orig = davies_bouldin(X, labels) 
	orig_logistic = 1 / (1 + np.exp(-k * orig))

	e_val = davies_bouldin_exp(X, labels)
	e_val_logistic = 1 / (1 + np.exp(-k * e_val))
	return (orig_logistic - e_val_logistic) / (1 - e_val_logistic)

def davies_bouldin_shift(X, labels):
	n_clusters = len(np.unique(labels))
	n_samples = X.shape[0]
	n_features = X.shape[1]

	std = np.std(np.sum(np.square(X - utils.centroid(X)), axis=1))

	centroids = np.zeros((n_clusters, n_features))
	for i in range(n_clusters):
		centroids[i, :] = utils.centroid(X[labels == i, :])
	
	entire_centroid = utils.centroid(X)
	
	sum_score = 0
	for i in range(n_clusters):
		max_score = -1
		for j in range(n_clusters):
			if i == j:
				continue
			compactness = np.exp(np.sum(np.square(X[labels == i, :] - centroids[i, :])) / (X[labels == i, :].shape[0] * std))
			compactness += np.exp(np.sum(np.square(X[labels == j, :] - centroids[j, :])) / (X[labels == j, :].shape[0] * std))
			separability = (np.sum(np.square(centroids[i, :] - centroids[j, :]))) / std
			invariance_term = np.sum(np.square(X - entire_centroid)) / (n_samples * std)
			curr_score = compactness / (separability * invariance_term)
			if curr_score > max_score:
				max_score = curr_score
		sum_score += max_score
	
	return (sum_score / n_clusters)  ** (-1)

def davies_bouldin_shift_exp(X, labels):
	return 0


def davies_bouldin_shift_range(X, labels, k=2.4831185988914117):
	orig = davies_bouldin_shift(X, labels)
	orig_logistic = 1 / (1 + np.exp(-k * orig))
	e_val = davies_bouldin_shift_exp(X, labels)
	e_val_logistic = 1 / (1 + np.exp(-k * e_val))
	return (orig_logistic - e_val_logistic) / (1 - e_val_logistic)

def davies_bouldin_shift_range_class(X, labels, k):
	return utils.pairwise_computation_k(X, labels,  k, davies_bouldin_shift_range)


def davies_bouldin_adjusted(X, labels, k):
	return davies_bouldin_shift_range_class(X, labels, k)

def davies_bouldin(X, labels, adjusted=False, k=2.4831185988914117):
	labels = utils.change_label_to_int(labels)
	utils.sanity_check(X, labels)
	if adjusted:
		return davies_bouldin_adjusted(X, labels, k)
	else:
		return davies_bouldin_original(X, labels)
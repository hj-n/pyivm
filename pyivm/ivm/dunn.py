from sklearn.metrics import euclidean_distances
import numpy as np
from . import utils


def dunn_original(X, labels):
	"""
	Calculate the Dunn index of a clustering.
		:param X: The data matrix.
		:param label: The cluster labels.
		:return: The Dunn index
	"""
	n_clusters = len(np.unique(labels))

	max_intra_dist = -np.inf
	for i in range(n_clusters):
		curr_cluster = X[labels == i, :]
		curr_dists = euclidean_distances(curr_cluster)
		curr_min_dist = np.max(curr_dists)
		max_intra_dist = max(max_intra_dist, curr_min_dist)
	
	inter_dists = euclidean_distances(X[labels==0,:], X[labels==1,:])
	min_inter_dist = np.min(inter_dists)

	result = min_inter_dist / max_intra_dist
	return result




def dunn_range(X, labels, k=2980.573789948166):
	np.random.shuffle(labels)
	orig = dunn(X, labels)
	orig_logistic = 1 / (1 + np.exp(-k * orig))
	eval_logistic = 0.5
	return (orig_logistic - eval_logistic) / (1 - eval_logistic)


def dunn_shift(X, labels):
	n_clusters = len(np.unique(labels))

	std = np.std(np.sqrt(np.sum(np.square(X - utils.centroid(X)), axis=1)))

	max_intra_dist = -np.inf
	for i in range(n_clusters):
		curr_cluster = X[labels == i, :]
		curr_dists = euclidean_distances(curr_cluster)
		curr_min_dist = np.max(curr_dists)
		max_intra_dist = max(max_intra_dist, curr_min_dist)
	
	inter_dists = euclidean_distances(X[labels==0,:], X[labels==1,:])
	min_inter_dist = np.min(inter_dists)

	max_intra_dist_exp = np.exp(max_intra_dist / std)
	min_inter_dist_exp = np.exp(min_inter_dist / std)

	result = min_inter_dist_exp / max_intra_dist_exp
	return result


def dunn_shift_range(X, labels, k=18039.038395458672):
	orig = dunn_shift(X, labels)
	orig_logistic = 1 / (1 + np.exp(-k * orig))
	e_val_logistic = 0.5
	return (orig_logistic - e_val_logistic) / (1 - e_val_logistic)


def dunn_shift_range_class(X, labels):
	return utils.pairwise_computation(X, labels, dunn_shift_range)

def dunn_dcal(X, labels):
	n_clusters = len(np.unique(labels))

	max_avg_intra_dist = -np.inf
	for i in range(n_clusters):
		curr_cluster = X[labels == i, :]
		curr_dists = euclidean_distances(curr_cluster)
		curr_avg_dist = np.sum(curr_dists) / (curr_cluster.shape[0] * (curr_cluster.shape[0] - 1))
		max_avg_intra_dist = max(max_avg_intra_dist, curr_avg_dist)
	
	inter_dists = euclidean_distances(X[labels==0,:], X[labels==1,:])
	min_avg_inter_dist = np.mean(inter_dists)

	result = min_avg_inter_dist / max_avg_intra_dist
	return result


def dunn_dcal_exp(X):
	pairwise_dist = euclidean_distances(X)
	pairwise_dist_avg = np.sum(pairwise_dist) / (X.shape[0] * (X.shape[0] - 1))

	median = utils.geometric_median(X)
	centroid_dist = np.sqrt(np.sum(np.square(X - median), axis=1))
	centroid_dist_avg = np.sum(centroid_dist) / X.shape[0]

	return centroid_dist_avg / pairwise_dist_avg

def dunn_dcal_range(X, labels, k =1.269):
	orig = dunn_dcal(X, labels)
	orig_logistic = 1 / (1 + np.exp(-k * orig))
	e_val = dunn_dcal_exp(X)
	e_val_logistic = 1 / (1 + np.exp(-k * e_val))
	return (orig_logistic - e_val_logistic) / (1 - e_val_logistic)

def dunn_dcal_shift(X, labels):
	n_clusters = len(np.unique(labels))

	std = np.std(np.sqrt(np.sum(np.square(X - utils.centroid(X)), axis=1)))

	max_avg_intra_dist = -np.inf
	for i in range(n_clusters):
		curr_cluster = X[labels == i, :]
		curr_dists = euclidean_distances(curr_cluster)
		curr_avg_dist = np.sum(curr_dists) / (curr_cluster.shape[0] * (curr_cluster.shape[0] - 1))
		max_avg_intra_dist = max(max_avg_intra_dist, curr_avg_dist)
	
	inter_dists = euclidean_distances(X[labels==0,:], X[labels==1,:])
	min_avg_inter_dist = np.mean(inter_dists)

	max_avg_intra_dist = np.exp(max_avg_intra_dist / std)
	min_avg_inter_dist = np.exp(min_avg_inter_dist / std)

	result = min_avg_inter_dist / max_avg_intra_dist
	return result



def dunn_dcal_shift_exp(X):
	std = np.std(np.sqrt(np.sum(np.square(X - utils.centroid(X)), axis=1)))
	pairwise_dist = euclidean_distances(X)
	pairwise_dist_avg = np.sum(pairwise_dist) / (X.shape[0] * (X.shape[0] - 1))

	median = utils.geometric_median(X)
	centroid_dist = np.sqrt(np.sum(np.square(X - median), axis=1))
	centroid_dist_avg = np.sum(centroid_dist) / X.shape[0]

	centroid_dist_avg = np.exp(centroid_dist_avg / std)
	pairwise_dist_avg = np.exp(pairwise_dist_avg / std)

	return centroid_dist_avg / pairwise_dist_avg

def dunn_dcal_shift_range(X, labels, k ):
	orig = dunn_dcal_shift(X, labels)
	orig_logistic = 1 / (1 + np.exp(-k * orig))

	e_val = dunn_dcal_shift_exp(X)
	e_val_logistic = 1 / (1 + np.exp(-k * e_val))

	
	return (orig_logistic - e_val_logistic) / (1 - e_val_logistic)

def dunn_dcal_shift_range_class(X, labels, k):
	return utils.pairwise_computation_k(X, labels, k, dunn_dcal_shift_range)



def dunn_adjusted(X, labels, k):
	return dunn_dcal_shift_range_class(X, labels, k)


def dunn(X, labels, adjusted=False, k=0.40019810656179045):
	labels = utils.change_label_to_int(labels)
	utils.sanity_check(X, labels)
	if adjusted:
		return dunn_adjusted(X, labels, k)
	else:
		return dunn_original(X, labels)
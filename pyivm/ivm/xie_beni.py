import numpy as np
from . import utils
from .i_index import i_index_adjusted

def xie_beni_original(X, labels):
	'''
	Calculate the Xie Beni index of a clustering.
		:param X: The data matrix. 
		:param label: The cluster labels.
		:return: The Xie Beni index
	'''
	n_clusters = len(np.unique(labels))
	n_samples = X.shape[0]
	n_features = X.shape[1]
	
	"""
	compute the compacteness of each cluster as 
	the sum of the squared distances of each point to its centroid
	"""

	## compute the centroids of each cluster
	centroids = np.zeros((n_clusters, n_features))
	for i in range(n_clusters):
		centroids[i, :] = utils.centroid(X[labels == i, :])
	
	## compute the sum of squared distances of each point to its centroid
	compactness = 0
	for i in range(n_clusters):
		compactness += np.sum(np.square(X[labels == i, :] - centroids[i, :]))

	"""
	Compute the separability of each cluster as
	the sqaured minimum distance between centroids
	"""


	## compute the minimum distance between centroids
	min_dist = np.inf
	for i in range(n_clusters):
		for j in range(i + 1, n_clusters):
			dist = utils.euc_dis(centroids[i, :], centroids[j, :])
			if dist < min_dist:
				min_dist = dist
	
	min_dist_squared = min_dist * min_dist
	separability = n_samples * min_dist_squared


	return compactness / separability

def xie_beni(X, labels, adjusted=False, k=0.29289310158571386):
	labels = utils.change_label_to_int(labels)
	utils.sanity_check(X, labels)
	if adjusted:
		return i_index_adjusted(X, labels, k)
	else:
		return xie_beni_original(X, labels)


import numpy as np
from . import utils
from sklearn.metrics.pairwise import euclidean_distances

def calinski_harabasz_template(X, label, normalize):



	n_clusters = len(np.unique(label))
	n_samples = X.shape[0]
	n_features = X.shape[1]
	centroids = np.zeros((n_clusters, n_features))
	for i in range(n_clusters):
		centroids[i, :] = utils.centroid(X[label == i, :])
	entire_centroid = utils.centroid(X)
	compactness = 0
	separability = 0	
	for i in range(n_clusters):
		compactness += np.sum(np.square(X[label == i, :] - centroids[i, :]))
		if (normalize):
			separability += np.sum(np.square(centroids[i, :] - entire_centroid)) * (X[label == i, :].shape[0] / n_samples)
		else:
			separability += np.sum(np.square(centroids[i, :] - entire_centroid)) * (X[label == i, :].shape[0])
	result =  (separability *  (n_samples - n_clusters)) /(compactness * (n_clusters - 1)) 
	return result


def calinski_harabasz_shift_template(X, label, normalize):
	n_clusters = len(np.unique(label))
	n_samples = X.shape[0]
	n_features = X.shape[1]

	std = np.std(np.sum(np.square(X - utils.centroid(X)), axis=1))

	centroids = np.zeros((n_clusters, n_features))
	for i in range(n_clusters):
		centroids[i, :] = utils.centroid(X[label == i, :])

	entire_centroid = utils.centroid(X)

	compactness = 0
	separability = 0	
	for i in range(n_clusters):
		compactness += np.sum((np.sum(np.square(X[label == i, :] - centroids[i, :]), axis=1)))
		separability += np.sum(np.square(centroids[i, :] - entire_centroid)) * X[label == i, :].shape[0]

	compactness  /= n_samples

	separability /= n_clusters
	compactness   = np.exp(compactness / std)
	
	separability = separability / std
	invariance_term = np.exp(np.mean(np.sum(np.square(X - entire_centroid), axis=1)) / std)
	
	if (normalize):
		result = (separability * invariance_term) / (compactness * n_samples)
	else:
		result = (separability * invariance_term) / compactness 
	return result

def calinski_harabasz_shift_exp_template(X, normalize=False):
	n_samples = X.shape[0]
	std = np.std(np.sum(np.square(X - utils.centroid(X)), axis=1))
	entire_centroid = utils.centroid(X)

	compactness = np.sum((np.sum(np.square(X - entire_centroid), axis=1))) / n_samples
	compactness = np.exp(compactness / std)
	separability = 0
	invariance_term = (np.exp(np.mean(np.sum(np.square(X - entire_centroid), axis=1)) / std))


	if normalize:
		return (separability * invariance_term) / (compactness * n_samples)
	else: 
		return (separability * invariance_term) / compactness



def calinski_harabasz_dcal(X, labels):
	return calinski_harabasz_template(X, labels, normalize=True)

def calinski_harabasz_range(X, label, k=0.0015492533420772602):
	orig = calinski_harabasz(X, label)
	orig_logistic = 1 / (1 + np.exp(-k * orig))
	e_val_logistic = 0.5
	return (orig_logistic - e_val_logistic) / (1 - e_val_logistic)

def calinski_harabasz_shift(X, labels):
	return calinski_harabasz_shift_template(X, labels, normalize=False)

def calinski_harabasz_dcal_shift(X, labels):
	return calinski_harabasz_shift_template(X, labels, normalize=True)

def calinski_harabasz_dcal_range(X, labels, k=1.5498422867781867):
	orig = calinski_harabasz_dcal(X, labels)
	orig_logistic = 1 / (1 + np.exp(-k * orig))
	e_val_logistic = 0.5
	return (orig_logistic - e_val_logistic) / (1 - e_val_logistic)

def calinski_harabasz_shift_range(X, labels, k=0.004393626999113602):
	orig = calinski_harabasz_shift(X, labels)
	orig_logistic = 1 / (1 + np.exp(-k * orig))
	e_val = 0
	e_val_logistic = 1 / (1 + np.exp(-k * e_val))

	if (e_val_logistic == 1):
		return 0
	return (orig_logistic - e_val_logistic) / (1 - e_val_logistic)

def calinski_harabasz_dcal_shift_range(X, labels, k):

	orig = calinski_harabasz_dcal_shift(X, labels)
	orig_logistic = 1 / (1 + np.exp(-k * orig))
	e_val = 0
	e_val_logistic = 1 / (1 + np.exp(-k * e_val))

	if (e_val_logistic == 1):
		return 0

	return (orig_logistic - e_val_logistic) / (1 - e_val_logistic)


def calinski_harabasz_shift_range_class(X, label, k):
	return utils.pairwise_computation_k(X, label, k, calinski_harabasz_dcal_shift_range)



def calinski_harabasz_adjusted(X, labels, k):
	return calinski_harabasz_shift_range_class(X, labels, k)


def calinski_harabasz(X, labels, adjusted=False, k=4.432010535838295):
	labels = utils.change_label_to_int(labels)
	utils.sanity_check(X, labels)
	if adjusted:
		return calinski_harabasz_adjusted(X, labels, k)
	else:
		return calinski_harabasz_template(X, labels, normalize=False)
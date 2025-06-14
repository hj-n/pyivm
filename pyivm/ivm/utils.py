import numpy as np
from scipy.spatial.distance import cdist, euclidean

def centroid(X):
	return np.mean(X, axis=0)

def euc_dis(vec_1, vec_2):
	return np.linalg.norm(vec_1 - vec_2)

def min_max_pairwise_distance(entire_dist):
	np.fill_diagonal(entire_dist, -np.inf)
	max_entire_dist = np.max(entire_dist)
	np.fill_diagonal(entire_dist, np.inf)
	min_entire_dist = np.min(entire_dist)

	return min_entire_dist, max_entire_dist

def min_max_dist(dist):
	min_dist = np.min(dist)
	max_dist = np.max(dist)
	return min_dist, max_dist

def pairwise_computation_k(X, labels, k, measure):
	class_num = len(np.unique(labels))
	result_pairwise = []
	for label_a in range(class_num):
		for label_b in range(label_a + 1, class_num):
			X_pair      = X[((labels == label_a) | (labels == label_b))]
			labels_pair = labels[((labels == label_a) | (labels == label_b))]

			unique_labels = np.unique(labels_pair)
			label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
			labels_pair = np.array([label_map[old_label] for old_label in labels_pair], dtype=np.int32)

			score = measure(X_pair, labels_pair, k)
			result_pairwise.append(score)
	
	return np.mean(result_pairwise)


def pairwise_computation(X, labels, measure):
	class_num = len(np.unique(labels))
	result_pairwise = []
	for label_a in range(class_num):
		for label_b in range(label_a + 1, class_num):
			X_pair      = X[((labels == label_a) | (labels == label_b))]
			labels_pair = labels[((labels == label_a) | (labels == label_b))]

			unique_labels = np.unique(labels_pair)
			label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
			labels_pair = np.array([label_map[old_label] for old_label in labels_pair], dtype=np.int32)

			score = measure(X_pair, labels_pair)
			result_pairwise.append(score)
	
	return np.mean(result_pairwise)

def geometric_median(X, eps=1e-5):
	y = np.mean(X, 0)

	while True:
		D = cdist(X, [y]) 
		nonzeros = (D != 0)[:, 0]

		Dinv = 1 / D[nonzeros]
		Dinvs = np.sum(Dinv)
		W = Dinv / Dinvs
		T = np.sum(W * X[nonzeros], 0)

		num_zeros = len(X) - np.sum(nonzeros)
		if num_zeros == 0:
			y1 = T
		elif num_zeros == len(X):
			return y
		else:
			R = (T - y) * Dinvs
			r = np.linalg.norm(R)
			rinv = 0 if r == 0 else num_zeros/r
			y1 = max(0, 1-rinv)*T + min(1, rinv)*y

		if euclidean(y, y1) < eps:
			return y1

		y = y1

def change_label_to_int(labels):
	
	unique_labels = np.unique(labels)
	label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
	labels_int = np.array([label_map[old_label] for old_label in labels], dtype=np.int32)
	return labels_int

def sanity_check(X, labels):
	"""
	Comprehensive sanity check for clustering data and labels
	
	Parameters:
	-----------
	X : array-like, shape (n_samples, n_features)
		Data points
	labels : array-like, shape (n_samples,)
		Cluster labels for each data point
		
	Returns:
	--------
	bool : True if all checks pass
	
	Raises:
	-------
	ValueError : If any sanity check fails
	"""
	X = np.asarray(X)
	labels = np.asarray(labels)
	
	# Check if arrays are empty
	if X.size == 0:
		raise ValueError("Data array X is empty")
	if labels.size == 0:
		raise ValueError("Labels array is empty")
	
	# Check dimensions
	if X.ndim != 2:
		raise ValueError(f"X should be 2D array, got {X.ndim}D array")
	if labels.ndim != 1:
		raise ValueError(f"Labels should be 1D array, got {labels.ndim}D array")
	
	# Check if lengths match
	if X.shape[0] != len(labels):
		raise ValueError(f"Number of samples in X ({X.shape[0]}) does not match number of labels ({len(labels)})")
	
	# Check for minimum samples
	if X.shape[0] < 3:
		raise ValueError(f"Need at least 3 samples for clustering metrics, got {X.shape[0]}")
	
	# Check for valid data (no NaN or infinite values)
	if np.any(np.isnan(X)):
		raise ValueError("Data X contains NaN values")
	if np.any(np.isinf(X)):
		raise ValueError("Data X contains infinite values")
	
	# Get unique labels and their counts
	unique_labels, counts = np.unique(labels, return_counts=True)
	
	# Check if there are at least 2 clusters
	if len(unique_labels) < 2:
		raise ValueError("Number of unique labels is 1. Need at least 2 clusters for clustering metrics.")
	
	# Check for single-point clusters
	single_point_clusters = unique_labels[counts == 1]
	if len(single_point_clusters) > 0:
		raise ValueError(f"Found clusters with only 1 data point: {single_point_clusters}. "
						f"All clusters must have at least 2 data points for meaningful clustering metrics.")
	
	# Check for too many clusters (more clusters than samples is suspicious)
	if len(unique_labels) >= X.shape[0]:
		raise ValueError(f"Number of clusters ({len(unique_labels)}) should be less than number of samples ({X.shape[0]})")
	
	# Check for NaN values in labels
	if np.any(np.isnan(labels)):
		raise ValueError("Labels contain NaN values")
	
	# Check label data type (should be convertible to integers)
	try:
		labels_int = labels.astype(int)
		if not np.array_equal(labels.astype(float), labels_int.astype(float)):
			raise ValueError("Labels should be integers or convertible to integers")
	except (ValueError, OverflowError):
		raise ValueError("Labels should be numeric and convertible to integers")
	
	return True

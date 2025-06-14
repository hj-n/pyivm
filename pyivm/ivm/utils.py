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
# PyIVM - Python Library for Clustering Quality Metrics

A comprehensive Python library for computing clustering quality metrics with both **original** and **adjusted** forms, making clustering evaluation more reliable and meaningful.

## Why Adjusted Metrics?

Traditional clustering metrics have a fundamental problem: they're biased and unreliable when comparing different clustering solutions. For example, some metrics artificially favor clusterings with more clusters, while others behave inconsistently across different datasets. This makes it nearly impossible to fairly evaluate which clustering is actually better.

PyIVM solves this with **adjusted metrics** (TPAMI 2025). The adjusted versions remove these biases and provide consistent, reliable scores that enable fair comparison across different numbers of clusters and datasets. Moreover, all adjusted metrics use a simple "higher = better" interpretation, making clustering evaluation straightforward.

## Features

- **Six essential clustering metrics**: Calinski-Harabasz, Davies-Bouldin, Dunn, I-Index, Silhouette, and Xie-Beni
- **Original + Adjusted forms**: Traditional metrics plus improved adjusted variants
- **Easy to use**: Simple API that works with scikit-learn and numpy arrays

## Installation

```bash
pip install pyivm
```

## Quick Start

```python
import numpy as np
import pyivm

# Your clustering data
X = np.random.rand(100, 2)  # Features
labels = np.random.randint(0, 3, 100)  # Cluster assignments

# Compute metrics (higher = better clustering)
calinski_score = pyivm.calinski_harabasz(X, labels)
silhouette_score = pyivm.silhouette(X, labels)

# For better cluster comparison, use adjusted forms
calinski_adj = pyivm.calinski_harabasz(X, labels, adjusted=True)
silhouette_adj = pyivm.silhouette(X, labels, adjusted=True)

print(f"Calinski-Harabasz: {calinski_score:.3f} (adjusted: {calinski_adj:.3f})")
print(f"Silhouette Score: {silhouette_score:.3f} (adjusted: {silhouette_adj:.3f})")
```

## When to Use Adjusted Metrics

Use **adjusted=True** when:
- Comparing clusterings with different numbers of clusters
- Selecting optimal number of clusters
- Evaluating clustering algorithms fairly
- Publishing research results

Use **original forms** when:
- You have fixed number of clusters
- Comparing with existing literature
- Following specific benchmarking protocols

## 📋 API Reference

All metrics follow the same simple pattern:

```python
score = pyivm.metric_name(X, labels, adjusted=False, **kwargs)
```

### Parameters
- **`X`** *(array-like, shape (n_samples, n_features))*: Data points
- **`labels`** *(array-like, shape (n_samples,))*: Cluster labels for each data point
- **`adjusted`** *(bool, default=False)*: Whether to use adjusted form for fair comparison
- **`**kwargs`**: Additional metric-specific parameters

### Supported Metrics

| Function | Description | Original Form | Adjusted Form |
|----------|-------------|---------------|---------------|
| `pyivm.calinski_harabasz(X, labels, adjusted=False)` | Calinski-Harabasz Index | ✅ Higher better | ✅ Higher better |
| `pyivm.davies_bouldin(X, labels, adjusted=False)` | Davies-Bouldin Index | ❌ Lower better | ✅ Higher better |
| `pyivm.dunn(X, labels, adjusted=False)` | Dunn Index | ✅ Higher better | ✅ Higher better |
| `pyivm.i_index(X, labels, adjusted=False)` | I Index | ✅ Higher better | ✅ Higher better |
| `pyivm.silhouette(X, labels, adjusted=False)` | Silhouette Coefficient | ✅ Higher better | ✅ Higher better |
| `pyivm.xie_beni(X, labels, adjusted=False)` | Xie Beni Index | ❌ Lower better | ✅ Higher better |

**✅ Adjusted Benefits**: All adjusted metrics are consistently "higher = better", making them easy to interpret and compare across different metrics and datasets.

### Example Usage

```python
import numpy as np
import pyivm

# Your clustering data
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
labels = np.array([0, 0, 0, 1, 1, 1])

# Compute all metrics
results = {
    'calinski_harabasz': pyivm.calinski_harabasz(X, labels),
    'calinski_harabasz_adj': pyivm.calinski_harabasz(X, labels, adjusted=True),
    'davies_bouldin': pyivm.davies_bouldin(X, labels),
    'davies_bouldin_adj': pyivm.davies_bouldin(X, labels, adjusted=True),
    'dunn': pyivm.dunn(X, labels),
    'dunn_adj': pyivm.dunn(X, labels, adjusted=True),
    'i_index': pyivm.i_index(X, labels),
    'i_index_adj': pyivm.i_index(X, labels, adjusted=True),
    'silhouette': pyivm.silhouette(X, labels),
    'silhouette_adj': pyivm.silhouette(X, labels, adjusted=True),
    'xie_beni': pyivm.xie_beni(X, labels),
    'xie_beni_adj': pyivm.xie_beni(X, labels, adjusted=True),
}

for metric, score in results.items():
    print(f"{metric}: {score:.4f}")
```

### Quick Comparison Helper

```python
def evaluate_clustering(X, labels, adjusted=True):
    """Evaluate clustering with all metrics"""
    metrics = {}
    
    # Higher is better metrics
    for metric_name in ['calinski_harabasz', 'dunn', 'i_index', 'silhouette']:
        metric_func = getattr(pyivm, metric_name)
        metrics[metric_name] = metric_func(X, labels, adjusted=adjusted)
    
    # Lower is better metrics
    for metric_name in ['davies_bouldin', 'xie_beni']:
        metric_func = getattr(pyivm, metric_name)
        metrics[metric_name] = metric_func(X, labels, adjusted=adjusted)
    
    return metrics

# Usage
scores = evaluate_clustering(X, labels, adjusted=True)
print(scores)
```

## Requirements

- Python ^3.9
- NumPy ^1.20.0
- SciPy ^1.7.0
- scikit-learn ^1.0.0



## Citation

If you use this library in your research, please cite:

```bibtex
@ARTICLE{10909451,
  author={Jeon, Hyeon and Aupetit, Michaël and Shin, DongHwa and Cho, Aeri and Park, Seokhyeon and Seo, Jinwook},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Measuring the Validity of Clustering Validation Datasets}, 
  year={2025},
  volume={47},
  number={6},
  pages={5045-5058},
  keywords={Reliability;Benchmark testing;Protocols;Training;Standards;Size measurement;Mutual information;Indexes;Electronic mail;Data mining;Clustering;clustering validation;internal clustering validation;external clustering validation;clustering benchmark},
  doi={10.1109/TPAMI.2025.3548011}
}
```
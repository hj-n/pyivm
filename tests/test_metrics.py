import pytest
import numpy as np
import requests
import os
import pyivm
import io
import json
from .config import DATASETS, METRICS

@pytest.fixture(scope="session", params=DATASETS)
def dataset_data(request):
    """Download and cache various datasets for testing"""
    dataset_name = request.param
    
    # Create cache directory
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    data_file = os.path.join(cache_dir, f"{dataset_name}_data.npy")
    label_file = os.path.join(cache_dir, f"{dataset_name}_labels.npy")
    
    data_url = f"https://github.com/hj-n/labeled-datasets/raw/master/npy/{dataset_name}/data.npy"
    label_url = f"https://github.com/hj-n/labeled-datasets/raw/master/npy/{dataset_name}/label.npy"
    
    # Download and cache data if not exists
    if not os.path.exists(data_file):
        print(f"Downloading {dataset_name} data...")
        data_response = requests.get(data_url)
        data = np.load(io.BytesIO(data_response.content))
        np.save(data_file, data)
        print(f"Data cached to {data_file}")
    else:
        print(f"Loading cached data from {data_file}")
        data = np.load(data_file)
    
    # Download and cache labels if not exists
    if not os.path.exists(label_file):
        print(f"Downloading {dataset_name} labels...")
        label_response = requests.get(label_url)
        labels = np.load(io.BytesIO(label_response.content))
        np.save(label_file, labels)
        print(f"Labels cached to {label_file}")
    else:
        print(f"Loading cached labels from {label_file}")
        labels = np.load(label_file)
    
    data_flatten = data.flatten() 
    data_global_mean = np.mean(data_flatten)
    data_global_std = np.std(data_flatten)
    data = (data - data_global_mean) / data_global_std
    
    return data, labels, dataset_name


class TestMultipleDatasets:
    """Test all clustering quality metrics across multiple datasets"""
    
    @pytest.mark.parametrize("metric_name,metric_params", METRICS)
    def test_all_metrics_all_datasets(self, dataset_data, metric_name, metric_params, request):
        """Test each metric with each dataset combination"""
        data, labels, dataset_name = dataset_data
        adjusted_str = "adjusted" if metric_params.get("adjusted", False) else "original"
        print(f"Testing {metric_name} ({adjusted_str}) on {dataset_name}")
        
        # Get the metric function from pyivm
        metric_func = getattr(pyivm, metric_name)
        
        # Call the metric with parameters
        score = metric_func(data, labels, **metric_params)
        
        # Validate the score
        assert isinstance(score, (int, float, np.number)), f"Score should be numeric, got {type(score)}"
        assert not np.isnan(score), f"Score should not be NaN for {metric_name} on {dataset_name}"
        assert np.isfinite(score), f"Score should be finite for {metric_name} on {dataset_name}"
        
        # Store score for HTML report
        metric_key = f"{metric_name}_{adjusted_str}"
        score_info = {
            "dataset": dataset_name,
            "metric": metric_key,
            "score": float(score),
            "data_shape": list(data.shape),
            "n_classes": int(len(np.unique(labels)))
        }
        
        # Store in global results for HTML report
        if not hasattr(pytest, '_score_results_global'):
            pytest._score_results_global = []
        pytest._score_results_global.append(score_info)
        
        # Also add as pytest extra for individual test
        request.node.add_report_section("call", "score_info", json.dumps(score_info, indent=2))
    
    def test_dataset_properties(self, dataset_data):
        """Test that dataset properties are valid"""
        data, labels, dataset_name = dataset_data
        print(f"Testing dataset properties for {dataset_name}")
        
        assert data.ndim == 2, f"Data should be 2D, got {data.ndim}D for {dataset_name}"
        assert labels.ndim == 1, f"Labels should be 1D, got {labels.ndim}D for {dataset_name}"
        assert data.shape[0] == labels.shape[0], f"Data and labels should have same length for {dataset_name}"
        assert len(np.unique(labels)) >= 2, f"Should have at least 2 classes for clustering in {dataset_name}"
    
    @pytest.mark.parametrize("metric_name,metric_params", METRICS)
    def test_metric_consistency(self, dataset_data, metric_name, metric_params):
        """Test that metrics produce consistent results on multiple runs"""
        data, labels, dataset_name = dataset_data
        
        # Get the metric function
        metric_func = getattr(pyivm, metric_name)
        
        # Run metric twice
        score1 = metric_func(data, labels, **metric_params)
        score2 = metric_func(data, labels, **metric_params)
        
        assert score1 == score2, f"{metric_name} should be deterministic on {dataset_name}"
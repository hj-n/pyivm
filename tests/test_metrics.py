import pytest
import numpy as np
import requests
import os
import pyivm
import io

@pytest.fixture(scope="session")
def fashion_mnist_data():
    """Download and cache Fashion-MNIST data for testing"""
    
    # Create cache directory
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    data_file = os.path.join(cache_dir, "fashion_mnist_data.npy")
    label_file = os.path.join(cache_dir, "fashion_mnist_labels.npy")
    
    data_url = "https://github.com/hj-n/labeled-datasets/raw/master/npy/fashion_mnist/data.npy"
    label_url = "https://github.com/hj-n/labeled-datasets/raw/master/npy/fashion_mnist/label.npy"
    
    # Download and cache data if not exists
    if not os.path.exists(data_file):
        print("Downloading Fashion-MNIST data...")
        data_response = requests.get(data_url)
        data = np.load(io.BytesIO(data_response.content))
        np.save(data_file, data)
        print(f"Data cached to {data_file}")
    else:
        print(f"Loading cached data from {data_file}")
        data = np.load(data_file)
    
    # Download and cache labels if not exists
    if not os.path.exists(label_file):
        print("Downloading Fashion-MNIST labels...")
        label_response = requests.get(label_url)
        labels = np.load(io.BytesIO(label_response.content))
        np.save(label_file, labels)
        print(f"Labels cached to {label_file}")
    else:
        print(f"Loading cached labels from {label_file}")
        labels = np.load(label_file)
    
    return data, labels

class TestClusteringMetrics:
    """Test all clustering quality metrics"""
    
    def test_calinski_harabasz_original(self, fashion_mnist_data):
        """Test Calinski-Harabasz original form"""
        data, labels = fashion_mnist_data
        score = pyivm.calinski_harabasz(data, labels)
        assert isinstance(score, (int, float, np.number))
        assert score > 0
    
    def test_calinski_harabasz_adjusted(self, fashion_mnist_data):
        """Test Calinski-Harabasz adjusted form"""
        data, labels = fashion_mnist_data
        score = pyivm.calinski_harabasz(data, labels, adjusted=True)
        assert isinstance(score, (int, float, np.number))
        assert 0 <= score <= 1
    
    def test_davies_bouldin_original(self, fashion_mnist_data):
        """Test Davies-Bouldin original form"""
        data, labels = fashion_mnist_data
        score = pyivm.davies_bouldin(data, labels)
        assert isinstance(score, (int, float, np.number))
        assert score > 0
    
    def test_davies_bouldin_adjusted(self, fashion_mnist_data):
        """Test Davies-Bouldin adjusted form"""
        data, labels = fashion_mnist_data
        score = pyivm.davies_bouldin(data, labels, adjusted=True)
        assert isinstance(score, (int, float, np.number))
        assert 0 <= score <= 1
    
    def test_dunn_original(self, fashion_mnist_data):
        """Test Dunn original form"""
        data, labels = fashion_mnist_data
        score = pyivm.dunn(data, labels)
        assert isinstance(score, (int, float, np.number))
        assert score > 0
    
    def test_dunn_adjusted(self, fashion_mnist_data):
        """Test Dunn adjusted form"""
        data, labels = fashion_mnist_data
        score = pyivm.dunn(data, labels, adjusted=True)
        assert isinstance(score, (int, float, np.number))
        assert 0 <= score <= 1
    
    def test_i_index_original(self, fashion_mnist_data):
        """Test I-Index original form"""
        data, labels = fashion_mnist_data
        score = pyivm.i_index(data, labels)
        assert isinstance(score, (int, float, np.number))
        assert score > 0
    
    def test_i_index_adjusted(self, fashion_mnist_data):
        """Test I-Index adjusted form"""
        data, labels = fashion_mnist_data
        score = pyivm.i_index(data, labels, adjusted=True)
        assert isinstance(score, (int, float, np.number))
        assert 0 <= score <= 1
    
    def test_silhouette_original(self, fashion_mnist_data):
        """Test Silhouette original form"""
        data, labels = fashion_mnist_data
        score = pyivm.silhouette(data, labels)
        assert isinstance(score, (int, float, np.number))
        assert -1 <= score <= 1
    
    def test_silhouette_adjusted(self, fashion_mnist_data):
        """Test Silhouette adjusted form"""
        data, labels = fashion_mnist_data
        score = pyivm.silhouette(data, labels, adjusted=True)
        assert isinstance(score, (int, float, np.number))
    
    def test_xie_beni_original(self, fashion_mnist_data):
        """Test Xie-Beni original form"""
        data, labels = fashion_mnist_data
        score = pyivm.xie_beni(data, labels)
        assert isinstance(score, (int, float, np.number))
        assert score > 0
    
    def test_xie_beni_adjusted(self, fashion_mnist_data):
        """Test Xie-Beni adjusted form"""
        data, labels = fashion_mnist_data
        score = pyivm.xie_beni(data, labels, adjusted=True)
        assert isinstance(score, (int, float, np.number))
        assert 0 <= score <= 1

class TestMetricParameters:
    """Test metric parameters and edge cases"""
    
    def test_custom_k_parameters(self, fashion_mnist_data):
        """Test custom k parameters for adjusted forms"""
        data, labels = fashion_mnist_data
        
        # Test custom k values
        score1 = pyivm.calinski_harabasz(data, labels, adjusted=True, k=1.0)
        score2 = pyivm.calinski_harabasz(data, labels, adjusted=True, k=2.0)
        
        assert isinstance(score1, (int, float, np.number))
        assert isinstance(score2, (int, float, np.number))
    
    def test_data_shapes(self, fashion_mnist_data):
        """Test that data shapes are as expected"""
        data, labels = fashion_mnist_data
        
        assert data.ndim == 2
        assert labels.ndim == 1
        assert data.shape[0] == labels.shape[0]
        assert len(np.unique(labels)) == 10  # Fashion-MNIST has 10 classes
    
    def test_metric_consistency(self, fashion_mnist_data):
        """Test that metrics produce consistent results"""
        data, labels = fashion_mnist_data
        
        # Run same metric twice
        score1 = pyivm.calinski_harabasz(data, labels)
        score2 = pyivm.calinski_harabasz(data, labels)
        
        assert score1 == score2
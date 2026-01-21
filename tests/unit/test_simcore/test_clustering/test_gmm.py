import pytest
import numpy as np
import tempfile
import os
from netfaker.simcore.clustering import GMMClusterer


def test_gmm_clusterer_initialization():
    """测试 GMM 聚类器初始化"""
    # 测试默认参数
    clusterer = GMMClusterer()
    assert clusterer.n_components == 3
    assert clusterer.params["n_components"] == 3
    assert clusterer.params["covariance_type"] == "full"
    assert clusterer.params["random_state"] == 42
    
    # 测试自定义参数
    clusterer = GMMClusterer(n_components=4, covariance_type="tied", random_state=123)
    assert clusterer.n_components == 4
    assert clusterer.params["n_components"] == 4
    assert clusterer.params["covariance_type"] == "tied"
    assert clusterer.params["random_state"] == 123


def test_gmm_clusterer_fit_predict():
    """测试 GMM 聚类器拟合和预测"""
    # 创建测试数据
    np.random.seed(42)
    X = np.concatenate([
        np.random.normal(0, 1, (50, 16)),
        np.random.normal(5, 1, (50, 16)),
        np.random.normal(10, 1, (50, 16))
    ])
    
    clusterer = GMMClusterer(n_components=3, random_state=42)
    clusterer.fit(X)
    
    # 测试预测
    labels = clusterer.predict(X)
    assert labels.shape == (150,)
    assert len(np.unique(labels)) <= 3
    
    # 测试概率预测
    probs = clusterer.predict_proba(X)
    assert probs.shape == (150, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)  # 概率和为1


def test_gmm_clusterer_save_load():
    """测试 GMM 聚类器保存和加载"""
    # 创建测试数据
    np.random.seed(42)
    X = np.random.normal(0, 1, (100, 16))
    
    clusterer = GMMClusterer(n_components=3, random_state=42)
    clusterer.fit(X)
    
    # 保存和加载
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        temp_path = tmp.name
    
    try:
        clusterer.save(temp_path)
        loaded_clusterer = GMMClusterer.load(temp_path)
        
        # 验证加载后的模型
        assert loaded_clusterer.n_components == clusterer.n_components
        assert loaded_clusterer.params == clusterer.params
        
        # 验证预测结果一致
        original_labels = clusterer.predict(X)
        loaded_labels = loaded_clusterer.predict(X)
        assert np.array_equal(original_labels, loaded_labels)
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_gmm_clusterer_unfitted_predict():
    """测试未拟合的聚类器预测会抛出异常"""
    clusterer = GMMClusterer(n_components=3)
    
    X = np.random.normal(0, 1, (10, 16))
    
    with pytest.raises(RuntimeError, match="Model not fitted"):
        clusterer.predict(X)
    
    with pytest.raises(RuntimeError, match="Model not fitted"):
        clusterer.predict_proba(X)


def test_gmm_clusterer_bic_aic():
    """测试 GMM 聚类器的 BIC 和 AIC 计算"""
    # 创建测试数据
    np.random.seed(42)
    X = np.random.normal(0, 1, (100, 16))
    
    clusterer = GMMClusterer(n_components=3, random_state=42)
    clusterer.fit(X)
    
    # 验证 BIC 和 AIC 是数值
    bic = clusterer.bic
    aic = clusterer.aic
    assert isinstance(bic, float)
    assert isinstance(aic, float)
    assert not np.isnan(bic)
    assert not np.isnan(aic)


def test_gmm_clusterer_edge_cases():
    """测试 GMM 聚类器的边界情况"""
    # 测试最小样本数（GMM 需要至少 2 个样本）
    X_min = np.random.normal(0, 1, (2, 16))
    clusterer = GMMClusterer(n_components=2, random_state=42)
    clusterer.fit(X_min)
    
    # 测试预测单样本
    X_single = np.random.normal(0, 1, (1, 16))
    label = clusterer.predict(X_single)
    assert label.shape == (1,)
    
    prob = clusterer.predict_proba(X_single)
    assert prob.shape == (1, 2)
    assert np.allclose(prob.sum(axis=1), 1.0)
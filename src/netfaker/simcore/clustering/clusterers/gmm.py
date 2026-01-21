from .base import BaseClusterer
from sklearn.mixture import GaussianMixture
import joblib
import numpy as np
from typing import Dict, Any


class GMMClusterer(BaseClusterer):
    """
    基于高斯混合模型的聚类器。
    支持概率输出和模型保存/加载。
    """
    
    def __init__(self, n_components: int = 3, **kwargs):
        """
        初始化 GMM 聚类器。
        
        Args:
            n_components: 聚类数量
            **kwargs: 传递给 GaussianMixture 的其他参数
        """
        default_kwargs = {
            "covariance_type": "full",
            "max_iter": 200,
            "random_state": 42,
            "init_params": "kmeans"
        }
        default_kwargs.update(kwargs)
        
        self._n_components = n_components
        self._params = {"n_components": n_components, **default_kwargs}
        self._model = GaussianMixture(n_components=n_components, **default_kwargs)
        self._fitted = False
    
    def fit(self, X: np.ndarray) -> 'GMMClusterer':
        """
        在训练数据上拟合 GMM 模型。
        
        Args:
            X: 特征矩阵，形状为 (n_samples, n_features)
            
        Returns:
            self: 拟合后的模型实例
        """
        self._model.fit(X)
        self._fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测样本的聚类标签。
        
        Args:
            X: 特征矩阵，形状为 (n_samples, n_features)
            
        Returns:
            聚类标签数组，形状为 (n_samples,)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测样本属于每个聚类的概率。
        
        Args:
            X: 特征矩阵，形状为 (n_samples, n_features)
            
        Returns:
            概率矩阵，形状为 (n_samples, n_components)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict_proba(X)
    
    def save(self, path: str) -> None:
        """
        保存模型到文件。
        
        Args:
            path: 保存路径
        """
        joblib.dump({
            "model": self._model,
            "n_components": self._n_components,
            "params": self._params,
            "fitted": self._fitted
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'GMMClusterer':
        """
        从文件加载模型。
        
        Args:
            path: 加载路径
            
        Returns:
            加载的模型实例
        """
        data = joblib.load(path)
        # 从 params 中获取 n_components，避免重复传递
        clusterer = cls(**data["params"])
        clusterer._model = data["model"]
        clusterer._fitted = data["fitted"]
        clusterer._n_components = data["n_components"]
        return clusterer
    
    @property
    def n_components(self) -> int:
        """
        聚类数量。
        """
        return self._n_components
    
    @property
    def params(self) -> Dict[str, Any]:
        """
        模型参数。
        """
        return self._params
    
    @property
    def bic(self) -> float:
        """
        贝叶斯信息准则，用于模型选择。
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.bic(self._model.means_)
    
    @property
    def aic(self) -> float:
        """
        赤池信息准则，用于模型选择。
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.aic(self._model.means_)
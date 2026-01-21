"""
DBSCAN 聚类器实现。
"""

from typing import Any, Dict

import joblib
import numpy as np
from sklearn.cluster import DBSCAN

from .base import BaseClusterer


class DBSCANClusterer(BaseClusterer):
    """
    基于 DBSCAN 的聚类器。
    支持密度-based 聚类，对非球形聚类和噪声数据更鲁棒。
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5, **kwargs):
        """
        初始化 DBSCAN 聚类器。

        Args:
            eps: 邻域半径
            min_samples: 最小样本数
            **kwargs: 传递给 DBSCAN 的其他参数
        """
        default_kwargs = {
            "metric": "euclidean",
            "algorithm": "auto",
            "leaf_size": 30,
            "p": None
        }
        default_kwargs.update(kwargs)

        self._eps = eps
        self._min_samples = min_samples
        self._params = {"eps": eps, "min_samples": min_samples, **default_kwargs}
        self._model = DBSCAN(eps=eps, min_samples=min_samples, **default_kwargs)
        self._fitted = False

    def fit(self, X: np.ndarray) -> 'DBSCANClusterer':
        """
        在训练数据上拟合 DBSCAN 模型。

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
        注意：DBSCAN 不支持新样本的预测，这里返回基于距离的最近邻聚类。

        Args:
            X: 特征矩阵，形状为 (n_samples, n_features)

        Returns:
            聚类标签数组，形状为 (n_samples,)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # DBSCAN 不原生支持预测，我们使用最近邻方法
        # 获取核心样本
        core_samples_mask = np.zeros_like(self._model.labels_, dtype=bool)
        core_samples_mask[self._model.core_sample_indices_] = True
        core_labels = self._model.labels_[core_samples_mask]
        core_samples = self._model.components_

        # 对新样本使用最近邻分类
        from sklearn.neighbors import NearestNeighbors
        if len(core_samples) > 0:
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(core_samples)
            distances, indices = nn.kneighbors(X)
            return core_labels[indices.flatten()]
        else:
            # 如果没有核心样本，返回所有点为噪声
            return np.full(X.shape[0], -1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测样本属于每个聚类的概率。
        注意：DBSCAN 不支持概率输出，这里返回基于距离的伪概率。

        Args:
            X: 特征矩阵，形状为 (n_samples, n_features)

        Returns:
            概率矩阵，形状为 (n_samples, n_components)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        labels = self.predict(X)
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)

        # 创建伪概率矩阵
        proba = np.zeros((X.shape[0], n_classes))
        for i, label in enumerate(labels):
            if label in unique_labels:
                class_idx = np.where(unique_labels == label)[0][0]
                proba[i, class_idx] = 1.0

        return proba

    def save(self, path: str) -> None:
        """
        保存模型到文件。

        Args:
            path: 保存路径
        """
        joblib.dump({
            "model": self._model,
            "eps": self._eps,
            "min_samples": self._min_samples,
            "params": self._params,
            "fitted": self._fitted
        }, path)

    @classmethod
    def load(cls, path: str) -> 'DBSCANClusterer':
        """
        从文件加载模型。

        Args:
            path: 加载路径

        Returns:
            加载的模型实例
        """
        data = joblib.load(path)
        clusterer = cls(**data["params"])
        clusterer._model = data["model"]
        clusterer._fitted = data["fitted"]
        clusterer._eps = data["eps"]
        clusterer._min_samples = data["min_samples"]
        return clusterer

    @property
    def n_components(self) -> int:
        """
        聚类数量。
        注意：DBSCAN 会自动确定聚类数量，这里返回拟合后的聚类数量。
        """
        if not self._fitted:
            return 0
        return len(np.unique(self._model.labels_)) - (1 if -1 in self._model.labels_ else 0)

    @property
    def params(self) -> Dict[str, Any]:
        """
        模型参数。
        """
        return self._params

    @property
    def labels_(self) -> np.ndarray:
        """
        训练样本的聚类标签。
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.labels_

    @property
    def core_sample_indices_(self) -> np.ndarray:
        """
        核心样本的索引。
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.core_sample_indices_

    @property
    def components_(self) -> np.ndarray:
        """
        核心样本。
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.components_

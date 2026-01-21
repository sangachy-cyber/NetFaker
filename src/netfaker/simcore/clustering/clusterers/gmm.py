"""GMM聚类器模块，基于高斯混合模型的聚类实现。

该模块提供了以下功能：
- 高斯混合模型聚类
- 概率输出支持
- 模型保存和加载
- 模型选择指标（BIC、AIC）
- 与BaseClusterer接口兼容
"""

from typing import Any, Dict

import joblib
import numpy as np
from sklearn.mixture import GaussianMixture

from .base import BaseClusterer


class GMMClusterer(BaseClusterer):
    """基于高斯混合模型的聚类器。

    该类实现了基于高斯混合模型的聚类算法，支持概率输出、
    模型保存/加载，以及模型选择指标计算。

    典型使用场景：
    - 网络状态聚类分析
    - 概率分布建模
    - 异常检测
    - 软聚类任务

    示例：
    ```python
    # 初始化GMM聚类器，使用3个聚类
    clusterer = GMMClusterer(
        n_components=3,
        covariance_type="full",
        max_iter=500,
        random_state=42
    )

    # 拟合模型
    clusterer.fit(train_features)

    # 预测聚类标签
    labels = clusterer.predict(test_features)

    # 预测概率
    probabilities = clusterer.predict_proba(test_features)

    # 保存模型
    clusterer.save("models/gmm_model.joblib")

    # 加载模型
    loaded_clusterer = GMMClusterer.load("models/gmm_model.joblib")
    ```
    """

    def __init__(self, n_components: int = 3, **kwargs):
        """初始化GMM聚类器。

        Args:
            n_components: 聚类数量，默认为3
            **kwargs: 传递给GaussianMixture的其他参数，包括：
                - covariance_type: 协方差矩阵类型，默认为"full"
                - max_iter: 最大迭代次数，默认为500
                - random_state: 随机种子，默认为42
                - init_params: 初始化方法，默认为"kmeans"

        Attributes:
            _n_components: 聚类数量
            _params: 模型参数字典
            _model: 内部GaussianMixture实例
            _fitted: 模型是否已拟合的标志
        """
        default_kwargs = {
            "covariance_type": "full",
            "max_iter": 500,  # 增加迭代次数以提高收敛性
            "random_state": 42,
            "init_params": "kmeans"
        }
        default_kwargs.update(kwargs)

        self._n_components = n_components
        self._params = {"n_components": n_components, **default_kwargs}
        self._model = GaussianMixture(n_components=n_components, **default_kwargs)
        self._fitted = False

    def fit(self, X: np.ndarray) -> 'GMMClusterer':
        """在训练数据上拟合GMM模型。

        使用期望最大化(EM)算法在训练数据上拟合高斯混合模型。

        Args:
            X: 特征矩阵，形状为 (n_samples, n_features)

        Returns:
            GMMClusterer: 拟合后的模型实例，支持链式调用

        示例：
        ```python
        # 拟合模型
        clusterer = GMMClusterer(n_components=3)
        clusterer = clusterer.fit(train_features)

        # 链式调用
        clusterer = GMMClusterer(n_components=3).fit(train_features)
        ```
        """
        self._model.fit(X)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测样本的聚类标签。

        根据拟合的GMM模型，预测每个样本最可能属于的聚类。

        Args:
            X: 特征矩阵，形状为 (n_samples, n_features)

        Returns:
            np.ndarray: 聚类标签数组，形状为 (n_samples,)，
                标签值为0到n_components-1的整数

        Raises:
            RuntimeError: 如果模型尚未拟合

        示例：
        ```python
        # 预测聚类标签
        labels = clusterer.predict(test_features)
        print("预测标签:", labels)
        print("标签形状:", labels.shape)  # 输出 (n_samples,)
        ```
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测样本属于每个聚类的概率。

        根据拟合的GMM模型，计算每个样本属于每个聚类的后验概率。

        Args:
            X: 特征矩阵，形状为 (n_samples, n_features)

        Returns:
            np.ndarray: 概率矩阵，形状为 (n_samples, n_components)，
                每一行的和为1，表示样本属于对应聚类的概率

        Raises:
            RuntimeError: 如果模型尚未拟合

        示例：
        ```python
        # 预测概率
        probabilities = clusterer.predict_proba(test_features)
        print("概率矩阵形状:", probabilities.shape)  # 输出 (n_samples, n_components)
        print("第一个样本的概率:", probabilities[0])  # 输出 [p1, p2, p3]
        print("概率和:", np.sum(probabilities[0]))  # 输出 1.0
        ```
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict_proba(X)

    def save(self, path: str) -> None:
        """保存模型到文件。

        将GMM聚类器的完整状态保存到文件，包括：
        - 内部GaussianMixture模型
        - 聚类数量
        - 模型参数
        - 拟合状态

        Args:
            path: 保存路径，推荐使用.joblib扩展名

        示例：
        ```python
        # 保存模型
        clusterer.save("models/gmm_model.joblib")
        print("模型已保存到: models/gmm_model.joblib")
        ```
        """
        joblib.dump({
            "model": self._model,
            "n_components": self._n_components,
            "params": self._params,
            "fitted": self._fitted
        }, path)

    @classmethod
    def load(cls, path: str) -> 'GMMClusterer':
        """从文件加载模型。

        从之前保存的文件中加载GMM聚类器的完整状态。

        Args:
            path: 加载路径，必须是之前用save()方法保存的文件

        Returns:
            GMMClusterer: 加载的模型实例，保持原有的拟合状态

        示例：
        ```python
        # 加载模型
        loaded_clusterer = GMMClusterer.load("models/gmm_model.joblib")
        print("模型已加载，聚类数量:", loaded_clusterer.n_components)
        print("模型是否已拟合:", loaded_clusterer._fitted)

        # 使用加载的模型进行预测
        labels = loaded_clusterer.predict(test_features)
        ```
        """
        data = joblib.load(path)
        # 从params中获取n_components，避免重复传递
        clusterer = cls(**data["params"])
        clusterer._model = data["model"]
        clusterer._fitted = data["fitted"]
        clusterer._n_components = data["n_components"]
        return clusterer

    @property
    def n_components(self) -> int:
        """聚类数量。

        Returns:
            int: 模型的聚类数量
        """
        return self._n_components

    @property
    def params(self) -> Dict[str, Any]:
        """模型参数。

        Returns:
            Dict[str, Any]: 模型参数字典，包括n_components和其他GaussianMixture参数
        """
        return self._params

    @property
    def bic(self) -> float:
        """贝叶斯信息准则，用于模型选择。

        较低的BIC值表示模型性能较好，同时考虑了模型复杂度。

        Returns:
            float: 贝叶斯信息准则值

        Raises:
            RuntimeError: 如果模型尚未拟合

        示例：
        ```python
        # 计算BIC值
        if clusterer._fitted:
            bic_value = clusterer.bic
            print("BIC值:", bic_value)
        ```
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.bic(self._model.means_)

    @property
    def aic(self) -> float:
        """赤池信息准则，用于模型选择。

        较低的AIC值表示模型性能较好，同时考虑了模型复杂度。

        Returns:
            float: 赤池信息准则值

        Raises:
            RuntimeError: 如果模型尚未拟合

        示例：
        ```python
        # 计算AIC值
        if clusterer._fitted:
            aic_value = clusterer.aic
            print("AIC值:", aic_value)
        ```
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.aic(self._model.means_)

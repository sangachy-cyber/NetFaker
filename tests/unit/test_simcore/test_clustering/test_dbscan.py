"""
DBSCAN 聚类器测试。
"""

import os
import tempfile

import numpy as np
import pytest

from netfaker.simcore.clustering.clusterers.dbscan import DBSCANClusterer


class TestDBSCANClusterer:
    """DBSCAN 聚类器测试类。"""

    def test_initialization(self):
        """测试 DBSCAN 聚类器初始化。"""
        # 测试默认参数初始化
        clusterer = DBSCANClusterer()
        assert clusterer.params["eps"] == 0.5
        assert clusterer.params["min_samples"] == 5
        assert clusterer.params["metric"] == "euclidean"

        # 测试自定义参数初始化
        clusterer = DBSCANClusterer(
            eps=0.3,
            min_samples=3,
            metric="manhattan",
            algorithm="ball_tree"
        )
        assert clusterer.params["eps"] == 0.3
        assert clusterer.params["min_samples"] == 3
        assert clusterer.params["metric"] == "manhattan"
        assert clusterer.params["algorithm"] == "ball_tree"

    def test_fit(self):
        """测试 DBSCAN 聚类器拟合。"""
        clusterer = DBSCANClusterer(eps=1.0, min_samples=2)

        # 创建测试数据
        X = np.array([
            [1, 2], [2, 2], [2, 3],
            [8, 7], [8, 8], [25, 80]
        ])

        # 测试拟合
        result = clusterer.fit(X)
        assert result is clusterer  # 验证返回 self

        # 验证拟合状态
        assert clusterer._fitted is True

        # 验证标签属性
        labels = clusterer.labels_
        assert labels.shape == (6,)

        # 验证核心样本
        core_indices = clusterer.core_sample_indices_
        assert len(core_indices) >= 0

        # 验证组件
        components = clusterer.components_
        assert components.shape[1] == 2  # 特征维度

    def test_predict(self):
        """测试 DBSCAN 聚类器预测。"""
        clusterer = DBSCANClusterer(eps=0.5, min_samples=2)

        # 创建训练数据
        X_train = np.array([
            [1, 2], [2, 2], [2, 3],
            [8, 7], [8, 8]
        ])

        # 拟合模型
        clusterer.fit(X_train)

        # 创建测试数据
        X_test = np.array([[1.5, 2.5], [7.5, 7.5]])

        # 测试预测
        labels = clusterer.predict(X_test)
        assert labels.shape == (2,)

        # 验证预测结果
        assert isinstance(labels[0], (int, np.integer))
        assert isinstance(labels[1], (int, np.integer))

    def test_predict_unfitted(self):
        """测试未拟合模型的预测。"""
        clusterer = DBSCANClusterer()

        # 创建测试数据
        X_test = np.array([[1.5, 2.5]])

        # 测试未拟合时的异常
        with pytest.raises(RuntimeError, match="Model not fitted"):
            clusterer.predict(X_test)

    def test_predict_without_core_samples(self):
        """测试无核心样本的预测。"""
        clusterer = DBSCANClusterer(eps=0.1, min_samples=5)

        # 创建训练数据（所有点都不会成为核心样本）
        X_train = np.array([[1, 2], [3, 4], [5, 6]])

        # 拟合模型
        clusterer.fit(X_train)

        # 创建测试数据
        X_test = np.array([[2, 3], [4, 5]])

        # 测试预测（应该返回所有点为噪声）
        labels = clusterer.predict(X_test)
        assert labels.shape == (2,)
        assert all(label == -1 for label in labels)

    def test_predict_proba(self):
        """测试 DBSCAN 聚类器概率预测。"""
        clusterer = DBSCANClusterer(eps=0.5, min_samples=2)

        # 创建训练数据
        X_train = np.array([
            [1, 2], [2, 2], [2, 3],
            [8, 7], [8, 8]
        ])

        # 拟合模型
        clusterer.fit(X_train)

        # 创建测试数据
        X_test = np.array([[1.5, 2.5], [7.5, 7.5]])

        # 测试概率预测
        proba = clusterer.predict_proba(X_test)
        assert proba.shape[0] == 2
        assert proba.shape[1] > 0  # 至少有一个聚类

        # 验证概率和为 1
        for row in proba:
            assert np.isclose(np.sum(row), 1.0)

    def test_predict_proba_unfitted(self):
        """测试未拟合模型的概率预测。"""
        clusterer = DBSCANClusterer()

        # 创建测试数据
        X_test = np.array([[1.5, 2.5]])

        # 测试未拟合时的异常
        with pytest.raises(RuntimeError, match="Model not fitted"):
            clusterer.predict_proba(X_test)

    def test_save_and_load(self):
        """测试 DBSCAN 聚类器保存和加载。"""
        clusterer = DBSCANClusterer(eps=0.4, min_samples=3)

        # 创建训练数据
        X_train = np.array([
            [1, 2], [2, 2], [2, 3],
            [8, 7], [8, 8], [9, 7]
        ])

        # 拟合模型
        clusterer.fit(X_train)

        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            model_path = tmp.name

        try:
            # 保存模型
            clusterer.save(model_path)

            # 加载模型
            loaded_clusterer = DBSCANClusterer.load(model_path)

            # 验证模型参数
            assert loaded_clusterer.params["eps"] == 0.4
            assert loaded_clusterer.params["min_samples"] == 3
            assert loaded_clusterer._fitted is True

            # 验证预测结果
            X_test = np.array([[1.5, 2.5], [8.5, 7.5]])
            original_labels = clusterer.predict(X_test)
            loaded_labels = loaded_clusterer.predict(X_test)
            assert np.array_equal(original_labels, loaded_labels)

        finally:
            # 清理临时文件
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_n_components(self):
        """测试聚类数量属性。"""
        clusterer = DBSCANClusterer(eps=0.5, min_samples=2)

        # 测试未拟合时的聚类数量
        assert clusterer.n_components == 0

        # 创建训练数据
        X_train = np.array([
            [1, 2], [2, 2], [2, 3],
            [8, 7], [8, 8]
        ])

        # 拟合模型
        clusterer.fit(X_train)

        # 测试拟合后的聚类数量
        n_components = clusterer.n_components
        assert n_components >= 0

    def test_labels_property_unfitted(self):
        """测试未拟合模型的标签属性访问。"""
        clusterer = DBSCANClusterer()

        # 测试未拟合时的异常
        with pytest.raises(RuntimeError, match="Model not fitted"):
            _ = clusterer.labels_

    def test_core_sample_indices_property_unfitted(self):
        """测试未拟合模型的核心样本索引属性访问。"""
        clusterer = DBSCANClusterer()

        # 测试未拟合时的异常
        with pytest.raises(RuntimeError, match="Model not fitted"):
            _ = clusterer.core_sample_indices_

    def test_components_property_unfitted(self):
        """测试未拟合模型的组件属性访问。"""
        clusterer = DBSCANClusterer()

        # 测试未拟合时的异常
        with pytest.raises(RuntimeError, match="Model not fitted"):
            _ = clusterer.components_

    def test_edge_case_empty_data(self):
        """测试空数据的边缘情况。"""
        clusterer = DBSCANClusterer()

        # 创建空数据
        X_empty = np.array([]).reshape(0, 2)

        # 测试拟合空数据（应该抛出异常）
        with pytest.raises(ValueError, match="Found array with 0 sample"):
            clusterer.fit(X_empty)

    def test_edge_case_single_sample(self):
        """测试单个样本的边缘情况。"""
        clusterer = DBSCANClusterer()

        # 创建单个样本数据
        X_single = np.array([[1, 2]])

        # 测试拟合单个样本
        result = clusterer.fit(X_single)
        assert result is clusterer
        assert clusterer._fitted is True

        # 测试标签
        labels = clusterer.labels_
        assert labels.shape == (1,)
        assert labels[0] == -1  # 应该被标记为噪声

    def test_custom_parameters(self):
        """测试自定义参数。"""
        # 测试不同的 DBSCAN 参数
        clusterer = DBSCANClusterer(
            eps=1.0,
            min_samples=4,
            metric="cosine",
            algorithm="kd_tree",
            leaf_size=20
        )

        assert clusterer.params["eps"] == 1.0
        assert clusterer.params["min_samples"] == 4
        assert clusterer.params["metric"] == "cosine"
        assert clusterer.params["algorithm"] == "kd_tree"
        assert clusterer.params["leaf_size"] == 20

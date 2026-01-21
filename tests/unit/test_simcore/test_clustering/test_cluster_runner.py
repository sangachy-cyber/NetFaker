import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from netfaker.simcore.clustering import ClusterRunner


def create_test_parquet_files():
    """创建测试用的 parquet 文件"""
    # 创建测试数据
    np.random.seed(42)

    # 训练集数据
    train_data = []
    for _i in range(100):
        window = {
            "raw_delay_up": list(np.random.normal(10, 2, 100)),
            "raw_loss_up": list(np.random.normal(0, 0.01, 100)),
            "raw_delay_down": list(np.random.normal(15, 3, 100)),
            "raw_loss_down": list(np.random.normal(0, 0.01, 100)),
            "is_valid": True
        }
        train_data.append(window)

    # 测试集数据
    test_data = []
    for _i in range(50):
        window = {
            "raw_delay_up": list(np.random.normal(10, 2, 100)),
            "raw_loss_up": list(np.random.normal(0, 0.01, 100)),
            "raw_delay_down": list(np.random.normal(15, 3, 100)),
            "raw_loss_down": list(np.random.normal(0, 0.01, 100)),
            "is_valid": True
        }
        test_data.append(window)

    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        train_path = tmp.name

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        test_path = tmp.name

    # 保存为 parquet
    train_df = pd.DataFrame(train_data)
    train_df.to_parquet(train_path)

    test_df = pd.DataFrame(test_data)
    test_df.to_parquet(test_path)

    return train_path, test_path


def test_cluster_runner_basic():
    """测试聚类执行器的基本功能"""
    # 创建测试数据
    train_path, test_path = create_test_parquet_files()

    try:
        runner = ClusterRunner(algorithm="gmm", n_components=3)
        stats = runner.run(train_path, test_path)

        # 验证返回的统计信息
        assert stats["algorithm"] == "gmm"
        assert stats["n_components"] == 3
        assert "train" in stats
        assert "test" in stats
        assert stats["train"]["n_samples"] == 100
        assert stats["test"]["n_samples"] == 50

        # 验证状态分布
        assert isinstance(stats["train"]["state_distribution"], dict)
        assert isinstance(stats["test"]["state_distribution"], dict)

        # 验证概率值
        assert 0 <= stats["train"]["mean_proba"] <= 1
        assert 0 <= stats["test"]["mean_proba"] <= 1

        # 验证文件生成
        assert os.path.exists("data/clusters/train_with_state.parquet")
        assert os.path.exists("data/clusters/test_with_state.parquet")
        assert os.path.exists("data/clusters/gmm_model.joblib")
        assert os.path.exists("data/clusters/feature_scaler.joblib")

    finally:
        # 清理临时文件
        if os.path.exists(train_path):
            os.unlink(train_path)
        if os.path.exists(test_path):
            os.unlink(test_path)
        # 清理生成的文件
        for file in [
            "data/clusters/train_with_state.parquet",
            "data/clusters/test_with_state.parquet",
            "data/clusters/gmm_model.joblib",
            "data/clusters/feature_scaler.joblib"
        ]:
            if os.path.exists(file):
                os.unlink(file)


def test_cluster_runner_invalid_algorithm():
    """测试聚类执行器使用无效算法"""
    # 创建测试数据
    train_path, test_path = create_test_parquet_files()

    try:
        runner = ClusterRunner(algorithm="invalid", n_components=3)
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            runner.run(train_path, test_path)
    finally:
        # 清理临时文件
        if os.path.exists(train_path):
            os.unlink(train_path)
        if os.path.exists(test_path):
            os.unlink(test_path)


def test_cluster_runner_with_invalid_windows():
    """测试聚类执行器处理包含无效窗口的数据"""
    # 创建包含无效窗口的测试数据
    np.random.seed(42)

    # 训练集数据：50 个有效，50 个无效
    train_data = []
    for _i in range(50):
        window = {
            "raw_delay_up": list(np.random.normal(10, 2, 100)),
            "raw_loss_up": list(np.random.normal(0, 0.01, 100)),
            "raw_delay_down": list(np.random.normal(15, 3, 100)),
            "raw_loss_down": list(np.random.normal(0, 0.01, 100)),
            "is_valid": True
        }
        train_data.append(window)

    for _i in range(50):
        window = {
            "raw_delay_up": list(np.random.normal(10, 2, 100)),
            "raw_loss_up": list(np.random.normal(0, 0.01, 100)),
            "raw_delay_down": list(np.random.normal(15, 3, 100)),
            "raw_loss_down": list(np.random.normal(0, 0.01, 100)),
            "is_valid": False
        }
        train_data.append(window)

    # 测试集数据
    test_data = []
    for _i in range(30):
        window = {
            "raw_delay_up": list(np.random.normal(10, 2, 100)),
            "raw_loss_up": list(np.random.normal(0, 0.01, 100)),
            "raw_delay_down": list(np.random.normal(15, 3, 100)),
            "raw_loss_down": list(np.random.normal(0, 0.01, 100)),
            "is_valid": True
        }
        test_data.append(window)

    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        train_path = tmp.name

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        test_path = tmp.name

    # 保存为 parquet
    train_df = pd.DataFrame(train_data)
    train_df.to_parquet(train_path)

    test_df = pd.DataFrame(test_data)
    test_df.to_parquet(test_path)

    try:
        runner = ClusterRunner(algorithm="gmm", n_components=3)
        stats = runner.run(train_path, test_path)

        # 验证只处理了有效窗口
        assert stats["train"]["n_samples"] == 50
        assert stats["test"]["n_samples"] == 30

    finally:
        # 清理临时文件
        if os.path.exists(train_path):
            os.unlink(train_path)
        if os.path.exists(test_path):
            os.unlink(test_path)
        # 清理生成的文件
        for file in [
            "data/clusters/train_with_state.parquet",
            "data/clusters/test_with_state.parquet",
            "data/clusters/gmm_model.joblib",
            "data/clusters/feature_scaler.joblib"
        ]:
            if os.path.exists(file):
                os.unlink(file)

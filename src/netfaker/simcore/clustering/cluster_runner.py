import contextlib
import json
import os
import time
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import (
    StandardScaler,  # 使用 StandardScaler 替代 RobustScaler
)

from .clusterers import GMMClusterer
from .feature_extractor import extract_features_from_window
from .state_namer import StateNamer
from .visualizer import Visualizer  # 添加 Visualizer 导入


class ClusterRunner:
    """聚类执行器，处理整个聚类流程的完整实现。

    该类封装了从数据加载到结果分析的完整聚类流程，
    包括特征提取、标准化、GMM聚类、状态分配、结果保存
    和可视化生成等步骤。

    典型使用场景：
    - 网络状态聚类分析
    - 流量模式识别
    - 异常检测基础

    示例：
    ```python
    # 初始化聚类执行器，使用3个聚类
    runner = ClusterRunner(
        algorithm="gmm",
        n_components=3,
        confidence_threshold=0.85,
        visualize=True
    )

    # 执行聚类流程
    results = runner.run(
        "data/train.parquet",
        "data/test.parquet"
    )

    # 查看聚类结果
    print("训练集纯净状态比例:", results["train"]["n_pure"] / results["train"]["n_samples"])
    print("测试集纯净状态比例:", results["test"]["n_pure"] / results["test"]["n_samples"])
    ```
    """

    def __init__(self, algorithm: str = "gmm", n_components: int = 3,
                 confidence_threshold: float = 0.85, visualize: bool = True):
        """初始化聚类执行器。

        Args:
            algorithm: 聚类算法，当前仅支持 "gmm"
            n_components: 聚类数量，建议根据实际数据复杂度调整
            confidence_threshold: 纯净状态的置信度阈值，
                值越高，纯净状态的判定越严格
            visualize: 是否生成可视化结果，包括UMAP和t-SNE降维图

        Attributes:
            algorithm: 聚类算法名称
            n_components: 聚类数量
            confidence_threshold: 纯净状态置信度阈值
            visualize: 是否生成可视化
            scaler: 特征标准化器
            clusterer: 聚类模型实例
            state_namer: 状态命名器实例
            timestamp: 时间戳，用于生成唯一标识符
        """
        self.algorithm = algorithm
        self.n_components = n_components
        self.confidence_threshold = confidence_threshold
        self.visualize = visualize
        self.scaler = StandardScaler()  # 使用 StandardScaler 进行特征标准化
        self.clusterer = None
        self.state_namer = None
        self.timestamp = int(time.time())

    def run(self, train_path: str, test_path: str) -> Dict[str, Any]:
        """执行完整的聚类流程。

        该方法是聚类执行器的核心方法，按照以下步骤执行：
        1. 加载训练集和测试集数据
        2. 过滤有效窗口数据
        3. 提取特征向量
        4. 特征标准化处理
        5. 初始化并拟合GMM聚类模型
        6. 初始化状态命名器
        7. 为训练集分配状态
        8. 为测试集分配状态
        9. 保存聚类结果
        10. 生成统计信息
        11. 保存日志文件
        12. 生成可视化结果（如果启用）

        Args:
            train_path: 训练集路径，支持parquet格式
            test_path: 测试集路径，支持parquet格式

        Returns:
            Dict[str, Any]: 包含以下信息的字典：
                - algorithm: 使用的聚类算法
                - n_components: 聚类数量
                - confidence_threshold: 置信度阈值
                - timestamp: 执行时间戳
                - train: 训练集统计信息
                - test: 测试集统计信息
                - gmm: GMM模型特定统计信息（如BIC、AIC）
                - visualization: 可视化文件路径（如果生成）

        示例：
        ```python
        # 执行聚类并获取结果
        results = runner.run("data/train.parquet", "data/test.parquet")

        # 分析结果
        print("聚类算法:", results["algorithm"])
        print("聚类数量:", results["n_components"])
        print("训练集样本数:", results["train"]["n_samples"])
        print("测试集样本数:", results["test"]["n_samples"])
        print("训练集纯净状态数:", results["train"]["n_pure"])
        print("测试集纯净状态数:", results["test"]["n_pure"])
        ```
        """
        # 1. 加载数据
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)

        # 2. 过滤有效窗口
        train_valid = train_df[train_df["is_valid"]]
        test_valid = test_df[test_df["is_valid"]]

        # 3. 提取特征
        train_features = self._extract_features(train_valid)
        test_features = self._extract_features(test_valid)

        # 4. 特征标准化
        train_features_scaled = self.scaler.fit_transform(train_features)
        test_features_scaled = self.scaler.transform(test_features)

        # 5. 初始化并拟合聚类器
        if self.algorithm == "gmm":
            # 使用 full 协方差矩阵，适合多类聚类
            # 使用 kmeans 初始化，提高聚类中心质量
            self.clusterer = GMMClusterer(
                n_components=self.n_components,
                covariance_type="full",
                init_params="kmeans"
            )
            self.clusterer.fit(train_features_scaled)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        # 6. 初始化状态命名器
        self.state_namer = StateNamer(
            n_components=self.n_components,
            confidence_threshold=self.confidence_threshold
        )

        # 7. 为训练集分配状态
        train_probas = self.clusterer.predict_proba(train_features_scaled)
        train_state_info = self.state_namer.assign_states(train_probas)

        # 8. 为测试集分配状态
        test_probas = self.clusterer.predict_proba(test_features_scaled)
        test_state_info = self.state_namer.assign_states(test_probas)

        # 9. 保存结果
        self._save_results(
            train_valid, train_state_info,
            test_valid, test_state_info
        )

        # 10. 生成统计信息
        stats = self._generate_stats(
            train_state_info,
            test_state_info
        )

        # 11. 保存日志
        self._save_log(stats)

        # 12. 生成可视化
        if self.visualize:
            try:
                visualizer = Visualizer("data/clusters/state_metadata.json")
                viz_paths = visualizer.visualize_combined(
                    train_features_scaled, train_state_info,
                    test_features_scaled, test_state_info
                )
                stats["visualization"] = viz_paths
            except Exception as e:
                logger.warning(f"可视化失败: {e}")

        return stats

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """从网络状态数据框中提取特征向量。

        该方法遍历数据框中的每一行，从网络状态窗口中提取
        特征向量，用于后续的聚类分析。

        Args:
            df: 包含网络状态窗口的数据框，必须包含以下列：
                - raw_delay_up: 上行延迟原始数据
                - raw_loss_up: 上行丢包原始数据
                - raw_delay_down: 下行延迟原始数据
                - raw_loss_down: 下行丢包原始数据

        Returns:
            np.ndarray: 特征矩阵，形状为 (n_samples, n_features)，
                其中 n_samples 是数据框中的样本数，
                n_features 是提取的特征维度

        示例：
        ```python
        # 假设 df 是包含网络状态窗口的数据框
        features = runner._extract_features(df)
        print("特征矩阵形状:", features.shape)  # 输出 (n_samples, n_features)
        ```
        """
        features = []
        for _, row in df.iterrows():
            window = {
                "raw_delay_up": row["raw_delay_up"],
                "raw_loss_up": row["raw_loss_up"],
                "raw_delay_down": row["raw_delay_down"],
                "raw_loss_down": row["raw_loss_down"]
            }
            feature_vector = extract_features_from_window(window)
            features.append(feature_vector)
        return np.array(features)

    def _save_results(self, train_valid: pd.DataFrame, train_state_info: Dict[str, np.ndarray],
                     test_valid: pd.DataFrame, test_state_info: Dict[str, np.ndarray]):
        """保存聚类结果到文件系统。

        该方法将聚类结果保存为以下文件：
        1. 训练集带状态信息的parquet文件
        2. 测试集带状态信息的parquet文件
        3. GMM聚类模型文件
        4. 特征缩放器文件
        5. 状态元数据JSON文件

        Args:
            train_valid: 训练集有效窗口数据框
            train_state_info: 训练集状态信息字典，包含：
                - state_id: 状态ID数组
                - state_name: 状态名称数组
                - is_pure: 是否为纯净状态的布尔数组
                - state_proba: 状态概率数组
                - top2_state_ids: 前两个可能状态ID的二维数组
                - top2_state_probas: 前两个可能状态概率的二维数组
            test_valid: 测试集有效窗口数据框
            test_state_info: 测试集状态信息字典，结构同上

        保存的文件：
            - data/clusters/train_with_state.parquet: 训练集带状态信息
            - data/clusters/test_with_state.parquet: 测试集带状态信息
            - data/clusters/gmm_model.joblib: GMM聚类模型
            - data/clusters/feature_scaler.joblib: 特征缩放器
            - data/clusters/state_metadata.json: 状态元数据
        """
        # 确保输出目录存在
        os.makedirs("data/clusters", exist_ok=True)

        # 保存训练集结果
        train_with_state = train_valid.copy()
        train_with_state["state_id"] = train_state_info["state_id"]
        train_with_state["state_name"] = train_state_info["state_name"]
        train_with_state["is_pure"] = train_state_info["is_pure"]
        train_with_state["state_proba"] = train_state_info["state_proba"]
        train_with_state["top2_state_ids"] = list(train_state_info["top2_state_ids"])
        train_with_state["top2_state_probas"] = list(train_state_info["top2_state_probas"])
        train_with_state["base_state_id"] = train_state_info["top2_state_ids"][:, 0]  # 主基础状态
        train_with_state.to_parquet("data/clusters/train_with_state.parquet")

        # 保存测试集结果
        test_with_state = test_valid.copy()
        test_with_state["state_id"] = test_state_info["state_id"]
        test_with_state["state_name"] = test_state_info["state_name"]
        test_with_state["is_pure"] = test_state_info["is_pure"]
        test_with_state["state_proba"] = test_state_info["state_proba"]
        test_with_state["top2_state_ids"] = list(test_state_info["top2_state_ids"])
        test_with_state["top2_state_probas"] = list(test_state_info["top2_state_probas"])
        test_with_state["base_state_id"] = test_state_info["top2_state_ids"][:, 0]  # 主基础状态
        test_with_state.to_parquet("data/clusters/test_with_state.parquet")

        # 保存模型和缩放器
        if self.clusterer:
            self.clusterer.save("data/clusters/gmm_model.joblib")
        joblib.dump(self.scaler, "data/clusters/feature_scaler.joblib")

        # 保存状态元数据
        if self.state_namer:
            self.state_namer.save_metadata("data/clusters/state_metadata.json")

    def _generate_stats(self, train_state_info: Dict[str, np.ndarray],
                       test_state_info: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """生成聚类统计信息。

        该方法从训练集和测试集的状态信息中提取统计数据，
        包括样本数量、纯净状态数量、混合状态数量、
        状态分布、平均概率和概率标准差等。

        Args:
            train_state_info: 训练集状态信息字典
            test_state_info: 测试集状态信息字典

        Returns:
            Dict[str, Any]: 包含以下统计信息的字典：
                - algorithm: 使用的聚类算法
                - n_components: 聚类数量
                - confidence_threshold: 置信度阈值
                - timestamp: 执行时间戳
                - train: 训练集统计信息
                - test: 测试集统计信息
                - gmm: GMM模型特定统计信息（如BIC、AIC，可选）

        示例：
        ```python
        # 生成统计信息
        stats = runner._generate_stats(train_state_info, test_state_info)

        # 分析统计结果
        print("训练集纯净状态比例:", stats["train"]["n_pure"] / stats["train"]["n_samples"])
        print("测试集纯净状态比例:", stats["test"]["n_pure"] / stats["test"]["n_samples"])
        print("训练集状态分布:", stats["train"]["state_distribution"])
        print("测试集状态分布:", stats["test"]["state_distribution"])
        ```
        """
        train_state_ids = train_state_info["state_id"]
        train_state_probas = train_state_info["state_proba"]
        train_is_pure = train_state_info["is_pure"]

        test_state_ids = test_state_info["state_id"]
        test_state_probas = test_state_info["state_proba"]
        test_is_pure = test_state_info["is_pure"]

        stats = {
            "algorithm": self.algorithm,
            "n_components": self.n_components,
            "confidence_threshold": self.confidence_threshold,
            "timestamp": self.timestamp,
            "train": {
                "n_samples": len(train_state_ids),
                "n_pure": int(np.sum(train_is_pure)),
                "n_mixed": int(np.sum(~train_is_pure)),
                "state_distribution": {int(k): int(v) for k, v in zip(*np.unique(train_state_ids, return_counts=True), strict=False)},
                "mean_proba": float(np.mean(train_state_probas)),
                "std_proba": float(np.std(train_state_probas))
            },
            "test": {
                "n_samples": len(test_state_ids),
                "n_pure": int(np.sum(test_is_pure)),
                "n_mixed": int(np.sum(~test_is_pure)),
                "state_distribution": {int(k): int(v) for k, v in zip(*np.unique(test_state_ids, return_counts=True), strict=False)},
                "mean_proba": float(np.mean(test_state_probas)),
                "std_proba": float(np.std(test_state_probas))
            }
        }

        # 添加 GMM 特定统计信息
        if self.algorithm == "gmm" and self.clusterer:
            with contextlib.suppress(BaseException):
                stats["gmm"] = {
                    "bic": float(self.clusterer.bic),
                    "aic": float(self.clusterer.aic)
                }

        return stats

    def _save_log(self, stats: Dict[str, Any]):
        """保存聚类统计信息到日志文件。

        该方法将聚类统计信息保存为JSON格式的日志文件，
        便于后续分析和调试。

        Args:
            stats: 统计信息字典，包含算法参数、训练集和测试集统计数据等

        保存的文件：
            - logs/clustering_{timestamp}.json: 聚类日志文件，
                其中 {timestamp} 是执行时的时间戳

        示例：
        ```python
        # 保存日志
        runner._save_log(stats)

        # 日志文件路径类似于：logs/clustering_1620000000.json
        ```
        """
        os.makedirs("logs", exist_ok=True)
        log_path = f"logs/clustering_{self.timestamp}.json"

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

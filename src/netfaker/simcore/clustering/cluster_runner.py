from .feature_extractor import extract_features_from_window
from .clusterers import GMMClusterer
from sklearn.preprocessing import RobustScaler
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import json
import time
import os


class ClusterRunner:
    """
    聚类执行器，处理整个聚类流程：
    1. 特征提取
    2. 特征标准化
    3. GMM 聚类
    4. 状态分配
    5. 结果保存
    """
    
    def __init__(self, algorithm: str = "gmm", n_components: int = 3):
        """
        初始化聚类执行器。
        
        Args:
            algorithm: 聚类算法，默认为 "gmm"
            n_components: 聚类数量，默认为 3
        """
        self.algorithm = algorithm
        self.n_components = n_components
        self.scaler = RobustScaler()
        self.clusterer = None
        self.timestamp = int(time.time())
    
    def run(self, train_path: str, test_path: str) -> Dict[str, Any]:
        """
        执行聚类流程。
        
        Args:
            train_path: 训练集路径
            test_path: 测试集路径
            
        Returns:
            聚类结果统计信息
        """
        # 1. 加载数据
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        
        # 2. 过滤有效窗口
        train_valid = train_df[train_df["is_valid"] == True]
        test_valid = test_df[test_df["is_valid"] == True]
        
        # 3. 提取特征
        train_features = self._extract_features(train_valid)
        test_features = self._extract_features(test_valid)
        
        # 4. 特征标准化
        train_features_scaled = self.scaler.fit_transform(train_features)
        test_features_scaled = self.scaler.transform(test_features)
        
        # 5. 初始化并拟合聚类器
        if self.algorithm == "gmm":
            self.clusterer = GMMClusterer(n_components=self.n_components)
            self.clusterer.fit(train_features_scaled)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        # 6. 为训练集分配状态
        train_state_ids = self.clusterer.predict(train_features_scaled)
        train_state_probas = self.clusterer.predict_proba(train_features_scaled).max(axis=1)
        
        # 7. 为测试集分配状态
        test_state_ids = self.clusterer.predict(test_features_scaled)
        test_state_probas = self.clusterer.predict_proba(test_features_scaled).max(axis=1)
        
        # 8. 保存结果
        self._save_results(
            train_valid, train_state_ids, train_state_probas,
            test_valid, test_state_ids, test_state_probas
        )
        
        # 9. 生成统计信息
        stats = self._generate_stats(
            train_state_ids, train_state_probas,
            test_state_ids, test_state_probas
        )
        
        # 10. 保存日志
        self._save_log(stats)
        
        return stats
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        从数据框中提取特征。
        
        Args:
            df: 包含网络状态窗口的数据框
            
        Returns:
            特征矩阵，形状为 (n_samples, n_features)
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
    
    def _save_results(self, train_valid: pd.DataFrame, train_state_ids: np.ndarray, 
                     train_state_probas: np.ndarray, test_valid: pd.DataFrame, 
                     test_state_ids: np.ndarray, test_state_probas: np.ndarray):
        """
        保存聚类结果。
        
        Args:
            train_valid: 训练集有效窗口
            train_state_ids: 训练集状态 ID
            train_state_probas: 训练集状态概率
            test_valid: 测试集有效窗口
            test_state_ids: 测试集状态 ID
            test_state_probas: 测试集状态概率
        """
        # 确保输出目录存在
        os.makedirs("data/clusters", exist_ok=True)
        
        # 保存训练集结果
        train_with_state = train_valid.copy()
        train_with_state["state_id"] = train_state_ids
        train_with_state["state_proba"] = train_state_probas
        train_with_state.to_parquet("data/clusters/train_with_state.parquet")
        
        # 保存测试集结果
        test_with_state = test_valid.copy()
        test_with_state["state_id"] = test_state_ids
        test_with_state["state_proba"] = test_state_probas
        test_with_state.to_parquet("data/clusters/test_with_state.parquet")
        
        # 保存模型和缩放器
        if self.clusterer:
            self.clusterer.save("data/clusters/gmm_model.joblib")
        joblib.dump(self.scaler, "data/clusters/feature_scaler.joblib")
    
    def _generate_stats(self, train_state_ids: np.ndarray, train_state_probas: np.ndarray, 
                       test_state_ids: np.ndarray, test_state_probas: np.ndarray) -> Dict[str, Any]:
        """
        生成聚类统计信息。
        
        Args:
            train_state_ids: 训练集状态 ID
            train_state_probas: 训练集状态概率
            test_state_ids: 测试集状态 ID
            test_state_probas: 测试集状态概率
            
        Returns:
            统计信息字典
        """
        stats = {
            "algorithm": self.algorithm,
            "n_components": self.n_components,
            "timestamp": self.timestamp,
            "train": {
                "n_samples": len(train_state_ids),
                "state_distribution": {int(k): int(v) for k, v in zip(*np.unique(train_state_ids, return_counts=True))},
                "mean_proba": float(np.mean(train_state_probas)),
                "std_proba": float(np.std(train_state_probas))
            },
            "test": {
                "n_samples": len(test_state_ids),
                "state_distribution": {int(k): int(v) for k, v in zip(*np.unique(test_state_ids, return_counts=True))},
                "mean_proba": float(np.mean(test_state_probas)),
                "std_proba": float(np.std(test_state_probas))
            }
        }
        
        # 添加 GMM 特定统计信息
        if self.algorithm == "gmm" and self.clusterer:
            try:
                stats["gmm"] = {
                    "bic": float(self.clusterer.bic),
                    "aic": float(self.clusterer.aic)
                }
            except:
                pass
        
        return stats
    
    def _save_log(self, stats: Dict[str, Any]):
        """
        保存聚类日志。
        
        Args:
            stats: 统计信息字典
        """
        os.makedirs("logs", exist_ok=True)
        log_path = f"logs/clustering_{self.timestamp}.json"
        
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
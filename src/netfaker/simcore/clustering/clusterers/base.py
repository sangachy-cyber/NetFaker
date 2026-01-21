from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


class BaseClusterer(ABC):
    """
    聚类算法基类，定义统一接口。
    支持未来扩展 KMeans、DBSCAN 等算法。
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseClusterer':
        """
        在训练数据上拟合聚类模型。
        
        Args:
            X: 特征矩阵，形状为 (n_samples, n_features)
            
        Returns:
            self: 拟合后的模型实例
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测样本的聚类标签。
        
        Args:
            X: 特征矩阵，形状为 (n_samples, n_features)
            
        Returns:
            聚类标签数组，形状为 (n_samples,)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测样本属于每个聚类的概率。
        
        Args:
            X: 特征矩阵，形状为 (n_samples, n_features)
            
        Returns:
            概率矩阵，形状为 (n_samples, n_components)
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        保存模型到文件。
        
        Args:
            path: 保存路径
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'BaseClusterer':
        """
        从文件加载模型。
        
        Args:
            path: 加载路径
            
        Returns:
            加载的模型实例
        """
        pass
    
    @property
    @abstractmethod
    def n_components(self) -> int:
        """
        聚类数量。
        """
        pass
    
    @property
    @abstractmethod
    def params(self) -> Dict[str, Any]:
        """
        模型参数。
        """
        pass
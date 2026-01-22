"""状态预测器模块。

该模块提供了基于聚类模型和状态命名器的状态预测功能，
可以根据输入的特征向量预测对应的state_id。
"""

import joblib
import numpy as np
from loguru import logger

from netfaker.simcore.clustering.clusterers.gmm import GMMClusterer
from netfaker.simcore.clustering.feature_extractor import extract_features_from_window
from netfaker.simcore.clustering.state_namer import StateNamer


class StatePredictor:
    """状态预测器，用于根据特征向量预测state_id。

    该类整合了聚类模型、缩放器和状态命名器，实现了完整的状态预测流程：
    1. 加载保存的模型和状态命名器
    2. 对输入特征进行标准化
    3. 使用聚类模型预测概率
    4. 使用状态命名器分配状态
    5. 返回预测的state_id

    典型使用场景：
    - 基于生成的仿真数据预测状态
    - 实时状态监测
    - 仿真结果验证

    示例：
    ```python
    # 初始化状态预测器
    predictor = StatePredictor()

    # 准备特征向量
    features = np.array([...])  # 16维特征向量

    # 预测状态
    state_id = predictor.predict(features)
    print(f"预测的state_id: {state_id}")
    ```
    """

    def __init__(self,
                 model_path: str = "data/clusters/gmm_model.joblib",
                 scaler_path: str = "data/clusters/feature_scaler.joblib",
                 metadata_path: str = "data/clusters/state_metadata.json"):
        """初始化状态预测器。

        Args:
            model_path: 聚类模型路径
            scaler_path: 特征缩放器路径
            metadata_path: 状态元数据路径

        Attributes:
            model: 加载的GMM聚类模型
            scaler: 加载的特征缩放器
            state_namer: 状态命名器实例
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.metadata_path = metadata_path

        # 加载模型和缩放器
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        self.state_namer = self._load_state_namer()

    def _load_model(self) -> GMMClusterer:
        """加载聚类模型。

        Returns:
            加载的GMM聚类模型
        """
        try:
            model = GMMClusterer.load(self.model_path)
            logger.info(f"成功加载聚类模型: {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"加载聚类模型失败: {e}")
            raise

    def _load_scaler(self):
        """加载特征缩放器。

        Returns:
            加载的特征缩放器
        """
        try:
            scaler = joblib.load(self.scaler_path)
            logger.info(f"成功加载特征缩放器: {self.scaler_path}")
            return scaler
        except Exception as e:
            logger.error(f"加载特征缩放器失败: {e}")
            raise

    def _load_state_namer(self) -> StateNamer:
        """加载状态命名器。

        Returns:
            状态命名器实例
        """
        try:
            # 从元数据文件中读取配置
            import json
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # 初始化状态命名器
            state_namer = StateNamer(
                n_components=metadata["n_components"],
                confidence_threshold=metadata["confidence_threshold"]
            )
            logger.info(f"成功加载状态命名器: {self.metadata_path}")
            return state_namer
        except Exception as e:
            logger.error(f"加载状态命名器失败: {e}")
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测状态ID。

        Args:
            features: 输入特征向量，形状为(n_samples, n_features)或(1, n_features)

        Returns:
            预测的state_id数组，形状为(n_samples,)或(1,)
        """
        # 确保输入是二维数组
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # 特征标准化
        scaled_features = self.scaler.transform(features)

        # 预测概率
        probabilities = self.model.predict_proba(scaled_features)

        # 分配状态
        state_info = self.state_namer.assign_states(probabilities)

        return state_info["state_id"]

    def predict_from_window(self, window: dict) -> int:
        """从窗口数据中提取特征并预测状态ID。

        Args:
            window: 包含网络状态数据的字典，必须包含以下字段：
                - raw_delay_up: 上行延迟序列
                - raw_loss_up: 上行丢包率序列
                - raw_delay_down: 下行延迟序列
                - raw_loss_down: 下行丢包率序列

        Returns:
            预测的state_id
        """
        # 提取特征
        features = extract_features_from_window(window)

        # 预测状态
        state_id = self.predict(features)

        return state_id.item() if hasattr(state_id, 'item') else state_id

    def predict_from_data_points(self, data_points: list) -> int:
        """从数据点列表中提取特征并预测状态ID。

        Args:
            data_points: 数据点列表，每个数据点包含6个值：
                [ul_delay, ul_loss, ul_bw, dl_delay, dl_loss, dl_bw]

        Returns:
            预测的state_id
        """
        # 将数据点转换为窗口格式
        window = {
            "raw_delay_up": [point[0] for point in data_points],
            "raw_loss_up": [point[1] for point in data_points],
            "raw_bw_up": [point[2] for point in data_points],
            "raw_delay_down": [point[3] for point in data_points],
            "raw_loss_down": [point[4] for point in data_points],
            "raw_bw_down": [point[5] for point in data_points]
        }

        # 提取特征并预测
        return self.predict_from_window(window)

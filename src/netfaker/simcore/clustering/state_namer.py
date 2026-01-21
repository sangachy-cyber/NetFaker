"""
状态命名器模块。
负责基于 GMM 后验概率生成唯一的 state_id、可解释的 state_name，
以及相关的状态元数据。
"""

import json
import os
import time
from typing import Any, Dict, Tuple

import numpy as np


class StateNamer:
    """
    状态命名器，基于 GMM 后验概率生成唯一的状态标识和元数据。
    """

    def __init__(self, n_components: int = 3, confidence_threshold: float = 0.85):
        """
        初始化状态命名器。

        Args:
            n_components: GMM 聚类数量
            confidence_threshold: 纯净状态的置信度阈值
        """
        self.n_components = n_components
        self.confidence_threshold = confidence_threshold
        self.state_mapping = self._generate_state_mapping()
        self.color_mapping = self._generate_color_mapping()

    def _generate_state_mapping(self) -> Dict[Tuple[int, ...], Dict[str, Any]]:
        """
        生成状态映射字典。

        Returns:
            状态映射字典，键为基础状态元组，值为包含 state_id 和 state_name 的字典
        """
        mapping = {}
        state_id = 0

        # 纯净状态
        for i in range(self.n_components):
            mapping[(i,)] = {
                "state_id": state_id,
                "state_name": self._get_base_state_name(i),
                "type": "pure"
            }
            state_id += 1

        # 混合状态（按字典序）
        for i in range(self.n_components):
            for j in range(i + 1, self.n_components):
                mapping[(i, j)] = {
                    "state_id": state_id,
                    "state_name": f"{self._get_base_state_name(i)}/{self._get_base_state_name(j)}",
                    "type": "mixed"
                }
                state_id += 1

        return mapping

    def _get_base_state_name(self, component_id: int) -> str:
        """
        获取基础状态名称。

        Args:
            component_id: 基础状态 ID

        Returns:
            基础状态名称
        """
        state_names = ["stable", "jittery", "abnormal"]
        if component_id < len(state_names):
            return state_names[component_id]
        return f"state_{component_id}"

    def _generate_color_mapping(self) -> Dict[int, str]:
        """
        生成状态颜色映射。

        Returns:
            颜色映射字典，键为 state_id，值为颜色代码
        """
        colors = [
            "#4CAF50",  # 绿色 - stable
            "#FF9800",  # 橙色 - jittery
            "#F44336",  # 红色 - abnormal
            "#FB8C00",  # 深橙色 - jittery/abnormal
            "#9C27B0",  # 紫色 - stable/abnormal
            "#2196F3"   # 蓝色 - stable/jittery
        ]

        color_map = {}
        for _state_tuple, state_info in self.state_mapping.items():
            state_id = state_info["state_id"]
            if state_id < len(colors):
                color_map[state_id] = colors[state_id]
            else:
                # 为超出预定义颜色的状态生成随机颜色
                import random
                color_map[state_id] = f"#{random.randint(0, 0xFFFFFF):06x}"

        return color_map

    def assign_states(self, probabilities: np.ndarray) -> Dict[str, np.ndarray]:
        """
        为样本分配状态。

        Args:
            probabilities: 后验概率矩阵，形状为 (n_samples, n_components)

        Returns:
            包含状态信息的字典
        """
        n_samples = probabilities.shape[0]

        # 初始化结果数组
        state_ids = np.zeros(n_samples, dtype=int)
        state_names = np.empty(n_samples, dtype=object)
        is_pure = np.zeros(n_samples, dtype=bool)
        state_probas = np.zeros(n_samples, dtype=float)
        top2_state_ids = np.zeros((n_samples, 2), dtype=int)
        top2_state_probas = np.zeros((n_samples, 2), dtype=float)

        for i in range(n_samples):
            # 获取当前样本的概率分布
            probs = probabilities[i]

            # 获取概率最高的两个基础状态
            top_indices = np.argsort(probs)[::-1][:2]
            top_probs = probs[top_indices]

            # 确定状态类型
            max_prob = top_probs[0]
            if max_prob >= self.confidence_threshold:
                # 纯净状态
                state_tuple = (top_indices[0],)
            else:
                # 混合状态
                state_tuple = tuple(sorted(top_indices[:2]))

            # 获取状态信息
            state_info = self.state_mapping[state_tuple]
            state_id = state_info["state_id"]
            state_name = state_info["state_name"]

            # 填充结果
            state_ids[i] = state_id
            state_names[i] = state_name
            is_pure[i] = (max_prob >= self.confidence_threshold)
            state_probas[i] = max_prob
            top2_state_ids[i] = top_indices
            top2_state_probas[i] = top_probs

        return {
            "state_id": state_ids,
            "state_name": state_names,
            "is_pure": is_pure,
            "state_proba": state_probas,
            "top2_state_ids": top2_state_ids,
            "top2_state_probas": top2_state_probas
        }

    def generate_metadata(self) -> Dict[str, Any]:
        """
        生成状态元数据。

        Returns:
            状态元数据字典
        """
        states = []
        for state_tuple, state_info in self.state_mapping.items():
            state = {
                "state_id": state_info["state_id"],
                "state_name": state_info["state_name"],
                "type": state_info["type"],
                "base_states": list(state_tuple),
                "color": self.color_mapping[state_info["state_id"]],
                "description": f"{'纯净状态' if len(state_tuple) == 1 else '混合状态'}"
            }
            states.append(state)

        metadata = {
            "algorithm": "gmm",
            "n_components": self.n_components,
            "confidence_threshold": self.confidence_threshold,
            "states": states,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

        return metadata

    def save_metadata(self, path: str) -> None:
        """
        保存状态元数据到文件。

        Args:
            path: 保存路径
        """
        metadata = self.generate_metadata()

        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    @property
    def total_states(self) -> int:
        """
        总状态数。

        Returns:
            总状态数
        """
        return len(self.state_mapping)

    def get_state_info(self, state_id: int) -> Dict[str, Any]:
        """
        根据 state_id 获取状态信息。

        Args:
            state_id: 状态 ID

        Returns:
            状态信息字典
        """
        for _state_tuple, state_info in self.state_mapping.items():
            if state_info["state_id"] == state_id:
                return {
                    **state_info,
                    "color": self.color_mapping[state_id]
                }
        raise ValueError(f"Unknown state_id: {state_id}")

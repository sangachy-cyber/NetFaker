#!/usr/bin/env python3

"""基于统计特征的样本生成方案
包含：
1. 基于统计特征的样本生成逻辑
2. 原始窗口与生成窗口的对比分析
3. 对比结果报告生成
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats, spatial
from typing import Dict, List, Tuple


class FeatureBasedGenerator:
    """基于统计特征的样本生成器"""

    def __init__(self, seed: int = 42):
        """初始化生成器
        
        Args:
            seed: 随机种子，用于复现结果
        """
        np.random.seed(seed)

    def extract_window_features(self, window: List[Dict]) -> Dict[str, float]:
        """提取窗口的统计特征
        
        Args:
            window: 窗口数据，包含delay_up_origin, delay_down_origin, loss_up_origin, loss_down_origin等字段
            
        Returns:
            窗口的统计特征字典
        """
        # 提取原始延迟和丢包率数据
        delays_up = np.array([row['delay_up_origin'] for row in window])
        delays_down = np.array([row['delay_down_origin'] for row in window])
        losses_up = np.array([row['loss_up_origin'] for row in window])
        losses_down = np.array([row['loss_down_origin'] for row in window])

        # 计算统计特征
        features = {
            # 延迟特征
            'delay_up_mean': np.mean(delays_up),
            'delay_up_std': np.std(delays_up),
            'delay_up_p1': np.percentile(delays_up, 1),
            'delay_up_p50': np.percentile(delays_up, 50),
            'delay_up_p95': np.percentile(delays_up, 95),
            'delay_up_p99': np.percentile(delays_up, 99),
            
            'delay_down_mean': np.mean(delays_down),
            'delay_down_std': np.std(delays_down),
            'delay_down_p1': np.percentile(delays_down, 1),
            'delay_down_p50': np.percentile(delays_down, 50),
            'delay_down_p95': np.percentile(delays_down, 95),
            'delay_down_p99': np.percentile(delays_down, 99),
            
            # 丢包率特征
            'loss_up_mean': np.mean(losses_up),
            'loss_up_std': np.std(losses_up),
            'loss_up_nonzero_ratio': np.mean(losses_up > 0),
            
            'loss_down_mean': np.mean(losses_down),
            'loss_down_std': np.std(losses_down),
            'loss_down_nonzero_ratio': np.mean(losses_down > 0),
            
            # 行为模式特征
            'delay_up_trend': self._calculate_trend(delays_up),
            'delay_down_trend': self._calculate_trend(delays_down),
            'delay_up_peak_count': self._count_peaks(delays_up),
            'delay_down_peak_count': self._count_peaks(delays_down),
        }

        return features

    def _calculate_trend(self, data: np.ndarray) -> float:
        """计算数据的趋势
        
        Args:
            data: 一维数组数据
            
        Returns:
            趋势值，范围[-1, 1]，负值表示下降趋势，正值表示上升趋势
        """
        # 使用线性回归计算趋势
        x = np.arange(len(data))
        slope, _, _, _, _ = stats.linregress(x, data)
        return slope

    def _count_peaks(self, data: np.ndarray, height_threshold: float = 0.7) -> int:
        """计算数据的峰值数量
        
        Args:
            data: 一维数组数据
            height_threshold: 峰值高度阈值，相对于数据的标准差
            
        Returns:
            峰值数量
        """
        # 计算阈值
        threshold = np.mean(data) + height_threshold * np.std(data)
        peaks = 0
        for i in range(1, len(data)-1):
            if data[i] > threshold and data[i] > data[i-1] and data[i] > data[i+1]:
                peaks += 1
        return peaks

    def generate_window(self, original_features: Dict[str, float], window_length: int = 100) -> Dict[str, np.ndarray]:
        """基于原始窗口特征生成新的窗口数据
        
        Args:
            original_features: 原始窗口的统计特征
            window_length: 生成窗口的长度
            
        Returns:
            生成的窗口数据，包含delay_up, delay_down, loss_up, loss_down
        """
        # 生成基础延迟序列
        # 使用正态分布作为基础，然后调整分布形状
        delays_up_base = np.random.normal(
            loc=original_features['delay_up_mean'],
            scale=original_features['delay_up_std'],
            size=window_length
        )
        delays_down_base = np.random.normal(
            loc=original_features['delay_down_mean'],
            scale=original_features['delay_down_std'],
            size=window_length
        )

        # 调整延迟序列，使其符合分位数分布
        delays_up = self._adjust_to_quantiles(delays_up_base, original_features, prefix='delay_up')
        delays_down = self._adjust_to_quantiles(delays_down_base, original_features, prefix='delay_down')

        # 注入行为模式
        delays_up = self._inject_behavior_pattern(delays_up, original_features, prefix='delay_up')
        delays_down = self._inject_behavior_pattern(delays_down, original_features, prefix='delay_down')

        # 确保延迟值为非负数
        delays_up = np.maximum(delays_up, 0)
        delays_down = np.maximum(delays_down, 0)

        # 生成丢包率数据
        losses_up = self._generate_losses(original_features, prefix='loss_up', length=window_length)
        losses_down = self._generate_losses(original_features, prefix='loss_down', length=window_length)

        return {
            'delay_up': delays_up,
            'delay_down': delays_down,
            'loss_up': losses_up,
            'loss_down': losses_down
        }

    def _adjust_to_quantiles(self, data: np.ndarray, features: Dict[str, float], prefix: str) -> np.ndarray:
        """调整数据使其符合指定的分位数
        
        Args:
            data: 原始数据
            features: 特征字典，包含分位数信息
            prefix: 特征前缀
            
        Returns:
            调整后的数据
        """
        # 使用排序和映射的方式调整分位数
        sorted_data = np.sort(data)
        sorted_indices = np.argsort(data)

        # 期望的分位数
        expected_quantiles = {
            1: features[f'{prefix}_p1'],
            50: features[f'{prefix}_p50'],
            95: features[f'{prefix}_p95'],
            99: features[f'{prefix}_p99'],
        }

        # 计算当前分位数
        current_quantiles = {
            1: np.percentile(data, 1),
            50: np.percentile(data, 50),
            95: np.percentile(data, 95),
            99: np.percentile(data, 99),
        }

        # 使用线性映射调整数据
        for p, expected in expected_quantiles.items():
            current = current_quantiles[p]
            if current != expected:
                # 计算调整比例
                data[sorted_indices] = self._linear_map(sorted_data, current, expected)
                sorted_data = np.sort(data)

        return data

    def _linear_map(self, data: np.ndarray, old_value: float, new_value: float) -> np.ndarray:
        """线性映射数据，将old_value映射到new_value
        
        Args:
            data: 数据数组
            old_value: 原始值
            new_value: 新值
            
        Returns:
            映射后的数据
        """
        # 计算映射比例
        if old_value == 0:
            return data
        scale = new_value / old_value
        return data * scale

    def _inject_behavior_pattern(self, data: np.ndarray, features: Dict[str, float], prefix: str) -> np.ndarray:
        """注入行为模式，如趋势和峰值
        
        Args:
            data: 基础数据
            features: 特征字典
            prefix: 特征前缀
            
        Returns:
            注入行为模式后的数据
        """
        window_length = len(data)
        result = data.copy()

        # 注入趋势
        trend = features[f'{prefix}_trend']
        if abs(trend) > 0.01:  # 如果趋势显著
            trend_slope = trend * window_length
            trend_line = np.linspace(0, trend_slope, window_length)
            result += trend_line

        # 注入峰值
        peak_count = features[f'{prefix}_peak_count']
        if peak_count > 0:
            # 随机生成峰值位置
            peak_positions = np.random.choice(window_length, size=peak_count, replace=False)
            peak_heights = features[f'{prefix}_p95'] - features[f'{prefix}_mean']
            
            for pos in peak_positions:
                # 在峰值位置周围生成一个小的峰值
                peak_window = max(1, int(window_length * 0.05))  # 峰值宽度为窗口长度的5%
                start = max(0, pos - peak_window // 2)
                end = min(window_length, pos + peak_window // 2)
                
                # 生成高斯形状的峰值
                x = np.arange(start, end)
                gaussian = peak_heights * np.exp(-((x - pos)**2) / (2 * (peak_window/4)**2))
                result[start:end] += gaussian

        return result

    def _generate_losses(self, features: Dict[str, float], prefix: str, length: int) -> np.ndarray:
        """生成丢包率数据
        
        Args:
            features: 特征字典
            prefix: 特征前缀
            length: 生成数据的长度
            
        Returns:
            生成的丢包率数据
        """
        # 基于非零比例生成二值丢包率
        nonzero_ratio = features[f'{prefix}_nonzero_ratio']
        losses = np.random.choice([0, 1], size=length, p=[1-nonzero_ratio, nonzero_ratio])
        
        # 调整丢包率使其符合均值
        current_mean = np.mean(losses)
        if abs(current_mean - features[f'{prefix}_mean']) > 0.01:
            # 计算需要调整的数量
            target_mean = features[f'{prefix}_mean']
            diff = int((target_mean - current_mean) * length)
            
            if diff > 0:
                # 需要增加1的数量
                zero_indices = np.where(losses == 0)[0]
                if len(zero_indices) >= diff:
                    selected_indices = np.random.choice(zero_indices, size=diff, replace=False)
                    losses[selected_indices] = 1
            elif diff < 0:
                # 需要增加0的数量
                one_indices = np.where(losses == 1)[0]
                if len(one_indices) >= abs(diff):
                    selected_indices = np.random.choice(one_indices, size=abs(diff), replace=False)
                    losses[selected_indices] = 0

        return losses


class WindowComparer:
    """窗口数据对比分析器"""

    def compare_windows(self, original_window: List[Dict], generated_window: Dict[str, np.ndarray]) -> Dict:
        """对比原始窗口和生成窗口
        
        Args:
            original_window: 原始窗口数据
            generated_window: 生成的窗口数据
            
        Returns:
            对比结果字典
        """
        # 提取原始窗口的延迟和丢包率数据
        original_delays_up = np.array([row['delay_up_origin'] for row in original_window])
        original_delays_down = np.array([row['delay_down_origin'] for row in original_window])
        original_losses_up = np.array([row['loss_up_origin'] for row in original_window])
        original_losses_down = np.array([row['loss_down_origin'] for row in original_window])

        # 生成窗口数据
        gen_delays_up = generated_window['delay_up']
        gen_delays_down = generated_window['delay_down']
        gen_losses_up = generated_window['loss_up']
        gen_losses_down = generated_window['loss_down']

        # 统计特征对比
        stats_comparison = self._compare_statistics(
            original_delays_up, gen_delays_up, '上行延迟')
        stats_comparison.update(self._compare_statistics(
            original_delays_down, gen_delays_down, '下行延迟'))
        stats_comparison.update(self._compare_statistics(
            original_losses_up, gen_losses_up, '上行丢包率'))
        stats_comparison.update(self._compare_statistics(
            original_losses_down, gen_losses_down, '下行丢包率'))

        # 相似度分析
        similarity_results = {
            'delay_up_similarity': self._calculate_similarity(original_delays_up, gen_delays_up),
            'delay_down_similarity': self._calculate_similarity(original_delays_down, gen_delays_down),
            'loss_up_similarity': self._calculate_similarity(original_losses_up, gen_losses_up),
            'loss_down_similarity': self._calculate_similarity(original_losses_down, gen_losses_down),
        }

        # 行为模式对比
        pattern_comparison = {
            'delay_up_trend_diff': self._calculate_trend(original_delays_up) - self._calculate_trend(gen_delays_up),
            'delay_down_trend_diff': self._calculate_trend(original_delays_down) - self._calculate_trend(gen_delays_down),
            'delay_up_peak_diff': self._count_peaks(original_delays_up) - self._count_peaks(gen_delays_up),
            'delay_down_peak_diff': self._count_peaks(original_delays_down) - self._count_peaks(gen_delays_down),
        }

        return {
            'statistics': stats_comparison,
            'similarity': similarity_results,
            'pattern': pattern_comparison
        }

    def _compare_statistics(self, original: np.ndarray, generated: np.ndarray, name: str) -> Dict:
        """对比统计特征
        
        Args:
            original: 原始数据
            generated: 生成数据
            name: 数据名称
            
        Returns:
            统计特征对比结果
        """
        stats_dict = {
            f'{name}_original_mean': np.mean(original),
            f'{name}_generated_mean': np.mean(generated),
            f'{name}_mean_diff': np.mean(generated) - np.mean(original),
            
            f'{name}_original_std': np.std(original),
            f'{name}_generated_std': np.std(generated),
            f'{name}_std_diff': np.std(generated) - np.std(original),
            
            f'{name}_original_p95': np.percentile(original, 95),
            f'{name}_generated_p95': np.percentile(generated, 95),
            f'{name}_p95_diff': np.percentile(generated, 95) - np.percentile(original, 95),
        }
        return stats_dict

    def _calculate_similarity(self, original: np.ndarray, generated: np.ndarray) -> float:
        """计算两个序列的相似度
        
        Args:
            original: 原始数据
            generated: 生成数据
            
        Returns:
            相似度值，范围[0, 1]
        """
        # 确保两个序列长度相同
        min_len =
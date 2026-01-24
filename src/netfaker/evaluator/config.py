"""评估模块配置管理。

负责加载和管理评估模块的配置，包括各项指标的阈值和权重。
"""

import os
from typing import Dict, Optional

import yaml


class EvaluationConfig:
    """评估模块配置类。
    
    管理各项指标的阈值和配置，支持从YAML文件加载配置。
    """

    def __init__(self, config_path: Optional[str] = None):
        """初始化评估配置。
        
        Args:
            config_path: 配置文件路径，默认使用内置配置
        """
        # 默认配置
        self._default_config = {
            'metrics': {
                # 窗口级指标
                'p90_delay_err': {
                    'threshold': 150.0,
                    'weight': 1.0
                },
                'p99_delay_err': {
                    'threshold': 300.0,
                    'weight': 1.0
                },
                'ks_statistic': {
                    'threshold': 0.3,
                    'weight': 1.0
                },
                'acf_mae_lag1_10': {
                    'threshold': 0.15,
                    'weight': 1.0
                },
                'psd_log_mae': {
                    'threshold': 1.0,
                    'weight': 1.0
                },
                'burst_count_diff': {
                    'threshold': 1,
                    'weight': 1.0
                },
                'burst_len_mae': {
                    'threshold': 2.0,
                    'weight': 1.0
                },
                'cond_delay_bias': {
                    'threshold': 150.0,
                    'weight': 1.0
                },
                # 序列级指标
                'acf_mae_lag1_100': {
                    'threshold': 0.20,
                    'weight': 1.0
                },
                'state_transition_jsd': {
                    'threshold': 0.15,
                    'weight': 1.0
                },
                'rolling_p99_drift': {
                    'threshold': 80.0,
                    'weight': 1.0
                },
                'inter_burst_interval_mae': {
                    'threshold': 50.0,
                    'weight': 1.0
                }
            },
            'report': {
                'generate_csv': True,
                'generate_markdown': True,
                'generate_visualizations': True,
                'viz_sample_rate': 0.1  # 生成10%的窗口可视化
            }
        }

        # 加载配置
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置文件。
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            Dict: 合并后的配置
        """
        if config_path is None:
            return self._default_config

        if not os.path.exists(config_path):
            print(f"警告：配置文件不存在，使用默认配置: {config_path}")
            return self._default_config

        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)

            # 合并配置（用户配置覆盖默认配置）
            merged_config = self._deep_merge(self._default_config, user_config)
            return merged_config
        except Exception as e:
            print(f"警告：配置文件加载失败，使用默认配置: {e}")
            return self._default_config

    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """深度合并两个字典。
        
        Args:
            dict1: 基础字典
            dict2: 覆盖字典
            
        Returns:
            Dict: 合并后的字典
        """
        result = dict1.copy()

        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # 递归合并子字典
                result[key] = self._deep_merge(result[key], value)
            else:
                # 覆盖或添加键值对
                result[key] = value

        return result

    def get_threshold(self, metric_name: str) -> float:
        """获取指定指标的阈值。
        
        Args:
            metric_name: 指标名称
            
        Returns:
            float: 指标阈值
        """
        return self.config['metrics'].get(metric_name, {}).get('threshold', float('inf'))

    def get_weight(self, metric_name: str) -> float:
        """获取指定指标的权重。
        
        Args:
            metric_name: 指标名称
            
        Returns:
            float: 指标权重
        """
        return self.config['metrics'].get(metric_name, {}).get('weight', 1.0)

    def should_generate_csv(self) -> bool:
        """是否生成CSV报告。
        
        Returns:
            bool: 是否生成CSV报告
        """
        return self.config['report'].get('generate_csv', True)

    def should_generate_markdown(self) -> bool:
        """是否生成Markdown报告。
        
        Returns:
            bool: 是否生成Markdown报告
        """
        return self.config['report'].get('generate_markdown', True)

    def should_generate_visualizations(self) -> bool:
        """是否生成可视化图表。
        
        Returns:
            bool: 是否生成可视化图表
        """
        return self.config['report'].get('generate_visualizations', True)

    def get_viz_sample_rate(self) -> float:
        """获取可视化抽样率。
        
        Returns:
            float: 可视化抽样率（0.0-1.0）
        """
        return self.config['report'].get('viz_sample_rate', 0.1)

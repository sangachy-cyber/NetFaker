"""HoloWAN 数据集保存模块。

负责：
1. 将窗口字典转换为 DataFrame
2. 确保所有必要列存在
3. 将 DataFrame 保存为 Parquet 文件
"""

import os
import pandas as pd
from typing import List, Dict, Any


class DatasetSaver:
    """将窗口数据保存为 Parquet 文件的类。"""
    
    def __init__(self, datasets_dir: str = 'data/datasets/'):
        """初始化数据集保存器。
        
        Args:
            datasets_dir: 保存数据集的目录
        """
        self.datasets_dir = datasets_dir
        
        # 创建目录（如果不存在）
        os.makedirs(datasets_dir, exist_ok=True)
    
    def save_dataset(self, windows: List[Dict[str, Any]], filename: str) -> int:
        """将窗口保存为 Parquet 数据集。
        
        Args:
            windows: 窗口字典列表
            filename: 输出文件名
        
        Returns:
            保存的样本数量
        
        Examples:
            >>> saver = DatasetSaver()
            >>> windows = [
            ...     {
            ...         'window_id': 'test_idx0',
            ...         'source_file': 'test.parquet',
            ...         'start_index': 0,
            ...         'is_valid': True,
            ...         'raw_delay_up': [100.0] * 100,
            ...         'raw_loss_up': [0.1] * 100,
            ...         'raw_bw_up': [10.0] * 100,
            ...         'raw_delay_down': [95.0] * 100,
            ...         'raw_loss_down': [0.0] * 100,
            ...         'raw_bw_down': [12.0] * 100,
            ...         'norm_delay_up': [0.0] * 100,
            ...         'norm_loss_up': [0.001] * 100,
            ...         'norm_bw_up': [10.0] * 100,
            ...         'norm_delay_down': [0.0] * 100,
            ...         'norm_loss_down': [0.0] * 100,
            ...         'norm_bw_down': [12.0] * 100
            ...     }
            ... ]
            >>> saved_count = saver.save_dataset(windows, 'test.parquet')
            >>> saved_count
            1
        """
        if not windows:
            # 创建具有正确架构的空 DataFrame
            df = self._create_empty_dataframe()
        else:
            # 将窗口转换为 DataFrame
            df = self._windows_to_dataframe(windows)
        
        # 确保输出路径
        output_path = os.path.join(self.datasets_dir, filename)
        
        # 保存为 Parquet
        df.to_parquet(output_path, index=False)
        
        return len(df)
    
    def _windows_to_dataframe(self, windows: List[Dict[str, Any]]) -> pd.DataFrame:
        """将窗口列表转换为 DataFrame。
        
        Args:
            windows: 窗口字典列表
        
        Returns:
            包含所有必要列的 DataFrame
        """
        # 转换为 DataFrame
        df = pd.DataFrame(windows)
        
        # 确保所有必要列存在
        required_columns = [
            'window_id',
            'source_file',
            'start_index',
            'state_id',
            'is_valid',
            'raw_delay_up',
            'raw_loss_up',
            'raw_bw_up',
            'raw_delay_down',
            'raw_loss_down',
            'raw_bw_down',
            'norm_delay_up',
            'norm_loss_up',
            'norm_bw_up',
            'norm_delay_down',
            'norm_loss_down',
            'norm_bw_down'
        ]
        
        # 添加缺失的列，设置默认值
        for col in required_columns:
            if col not in df.columns:
                if col == 'state_id':
                    df[col] = 0
                elif col == 'is_valid':
                    df[col] = True
                else:
                    df[col] = None
        
        # 重新排序列以匹配规范
        df = df[required_columns]
        
        # 设置正确的数据类型
        df['start_index'] = df['start_index'].astype('int64')
        df['state_id'] = df['state_id'].astype('int32')
        df['is_valid'] = df['is_valid'].astype('bool')
        
        return df

    def _create_empty_dataframe(self) -> pd.DataFrame:
        """创建具有正确架构的空 DataFrame。
        
        Returns:
            包含所有必要列的空 DataFrame
        """
        columns = [
            'window_id',
            'source_file',
            'start_index',
            'state_id',
            'is_valid',
            'raw_delay_up',
            'raw_loss_up',
            'raw_bw_up',
            'raw_delay_down',
            'raw_loss_down',
            'raw_bw_down',
            'norm_delay_up',
            'norm_loss_up',
            'norm_bw_up',
            'norm_delay_down',
            'norm_loss_down',
            'norm_bw_down'
        ]
        
        dtypes = {
            'window_id': 'object',
            'source_file': 'object',
            'start_index': 'int64',
            'state_id': 'int32',
            'is_valid': 'bool',
            'raw_delay_up': 'object',  # 浮点数列表
            'raw_loss_up': 'object',    # 浮点数列表
            'raw_bw_up': 'object',      # 浮点数列表
            'raw_delay_down': 'object', # 浮点数列表
            'raw_loss_down': 'object',  # 浮点数列表
            'raw_bw_down': 'object',    # 浮点数列表
            'norm_delay_up': 'object',  # 浮点数列表
            'norm_loss_up': 'object',   # 浮点数列表
            'norm_bw_up': 'object',     # 浮点数列表
            'norm_delay_down': 'object',# 浮点数列表
            'norm_loss_down': 'object', # 浮点数列表
            'norm_bw_down': 'object'    # 浮点数列表
        }
        
        return pd.DataFrame(columns=columns).astype(dtypes)

"""HoloWAN 数据保存器模块。

将处理后的数据保存为 Parquet 文件，并序列化全局延迟 scaler。
"""

import os
from typing import Dict

import joblib
import pandas as pd


class HoloWANWriter:
    """HoloWAN 数据写入器类。

    负责将处理后的数据保存为 Parquet 文件，并序列化全局延迟 scaler。
    """

    def __init__(self, processed_data_dir: str = "data/processed/"):
        """初始化写入器。

        Args:
            processed_data_dir: 处理后数据的保存目录
        """
        self.processed_data_dir = processed_data_dir
        # 确保目录存在
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def save_all_data(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """保存所有处理后的数据。

        将所有处理后的数据保存为 Parquet 文件，文件名为原始文件名（不含扩展名）。

        Args:
            data_dict: 文件名到处理后 DataFrame 的映射

        Examples:
            >>> import pandas as pd
            >>> data = {
            ...     'raw_delay_up': [100.0],
            ...     'raw_loss_up': [0.1],
            ...     'raw_bw_up': [10.0],
            ...     'raw_delay_down': [95.0],
            ...     'raw_loss_down': [0.0],
            ...     'raw_bw_down': [12.0],
            ...     'norm_delay_up': [0.0],
            ...     'norm_loss_up': [0.001],
            ...     'norm_bw_up': [10.0],
            ...     'norm_delay_down': [0.0],
            ...     'norm_loss_down': [0.0],
            ...     'norm_bw_down': [12.0]
            ... }
            >>> df = pd.DataFrame(data)
            >>> writer = HoloWANWriter(processed_data_dir="data/processed/")
            >>> writer.save_all_data({'campus': df})
            ✅ 保存成功: data/processed/campus.parquet
        """
        for filename, df in data_dict.items():
            # 构建输出文件路径
            output_path = os.path.join(self.processed_data_dir, f"{filename}.parquet")
            # 保存为 Parquet 文件
            self.save_data(df, output_path)

    def save_data(self, df: pd.DataFrame, output_path: str) -> None:
        """保存单个处理后的数据。

        将单个处理后的数据保存为 Parquet 文件，确保输出目录存在。

        Args:
            df: 处理后的 DataFrame
            output_path: 输出文件路径

        Examples:
            >>> import pandas as pd
            >>> data = {
            ...     'raw_delay_up': [100.0],
            ...     'norm_delay_up': [0.0]
            ... }
            >>> df = pd.DataFrame(data)
            >>> writer = HoloWANWriter()
            >>> writer.save_data(df, "data/processed/test.parquet")
            ✅ 保存成功: data/processed/test.parquet
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        # 保存为 Parquet 文件
        df.to_parquet(output_path, index=False)
        print(f"✅ 保存成功: {output_path}")

    def save_scaler(self, scaler_info: Dict, output_path: str = "data/processed/global_delay_scaler.joblib") -> None:
        """保存全局延迟 scaler。

        将 scaler 信息序列化为 joblib 文件，以便后续使用。

        Args:
            scaler_info: scaler 信息字典
            output_path: 输出文件路径

        Examples:
            >>> scaler_info = {
            ...     'mode': 'shared',
            ...     'scaler': None  # 实际使用中会有真实的 scaler 对象
            ... }
            >>> writer = HoloWANWriter()
            >>> writer.save_scaler(scaler_info)
            ✅ 保存 scaler: data/processed/global_delay_scaler.joblib
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        # 序列化 scaler
        joblib.dump(scaler_info, output_path)
        print(f"✅ 保存 scaler: {output_path}")

"""HoloWAN 数据加载器模块。

从 data/raw/ 目录加载 .txt 文件，使用 HoloWANFile 解析，
转换为 DataFrame 并按规范重命名列。
"""

import os
from typing import Dict, Optional

import pandas as pd

from netfaker.simcore.utils.holowan import HoloWANFile


class HoloWANLoader:
    """HoloWAN 数据加载器类。
    
    负责从 data/raw/ 目录加载 .txt 文件，解析并转换为 DataFrame，
    同时按照规范重命名列。
    """

    def __init__(self, raw_data_dir: str = "data/raw/"):
        """初始化加载器。
        
        Args:
            raw_data_dir: 原始数据目录路径
        """
        self.raw_data_dir = raw_data_dir

    def load_all_files(self) -> Dict[str, pd.DataFrame]:
        """加载目录中所有 .txt 文件。
        
        Returns:
            Dict[str, pd.DataFrame]: 文件名到 DataFrame 的映射
            
        Examples:
            >>> loader = HoloWANLoader(raw_data_dir="data/raw/")
            >>> data_dict = loader.load_all_files()
            >>> print(f"成功加载 {len(data_dict)} 个文件")
            成功加载 2 个文件
            >>> print(list(data_dict.keys()))
            ['campus', 'wan_trace']
        """
        data_dict = {}

        # 确保目录存在
        if not os.path.exists(self.raw_data_dir):
            print(f"警告: 目录 {self.raw_data_dir} 不存在")
            return data_dict

        # 遍历目录中的 .txt 文件
        for filename in os.listdir(self.raw_data_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.raw_data_dir, filename)
                try:
                    # 加载文件
                    df = self.load_file(filepath)
                    if df is not None and not df.empty:
                        # 使用文件名（不含扩展名）作为键
                        key = os.path.splitext(filename)[0]
                        data_dict[key] = df
                        print(f"✅ 成功加载: {filename}")
                    else:
                        print(f"⚠️  空数据: {filename}")
                except Exception as e:
                    print(f"❌ 加载失败 {filename}: {e}")

        return data_dict

    def load_file(self, filepath: str) -> Optional[pd.DataFrame]:
        """加载单个 .txt 文件。
        
        Args:
            filepath: 文件路径
        
        Returns:
            Optional[pd.DataFrame]: 加载的数据，失败返回 None
            
        Examples:
            >>> loader = HoloWANLoader()
            >>> df = loader.load_file("data/raw/campus.txt")
            >>> print(df.shape)
            (100, 6)
            >>> print(list(df.columns))
            ['raw_delay_up', 'raw_loss_up', 'raw_bw_up', 'raw_delay_down', 'raw_loss_down', 'raw_bw_down']
        """
        # 使用 HoloWANFile 解析文件
        holowan_file = HoloWANFile.from_file(filepath)

        # 转换数据点为列表
        data_points = []
        for data_point in holowan_file.data:
            data_points.append({
                "delay1": data_point.delay1,
                "loss1": data_point.loss1,
                "bw1": data_point.bw1,
                "delay2": data_point.delay2,
                "loss2": data_point.loss2,
                "bw2": data_point.bw2
            })

        if not data_points:
            return None

        # 创建 DataFrame
        df = pd.DataFrame(data_points)

        # 重命名列
        df = self._rename_columns(df)

        # 添加原始值前缀
        df = self._add_raw_prefix(df)

        return df

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """重命名列，使用 _up/_down 后缀。
        
        将原始列名从数字标识转换为方向标识，提高可读性。
        
        Args:
            df: 原始 DataFrame
        
        Returns:
            pd.DataFrame: 重命名后的 DataFrame
            
        Examples:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     "delay1": [100.0],
            ...     "loss1": [0.1],
            ...     "bw1": [10.0],
            ...     "delay2": [95.0],
            ...     "loss2": [0.0],
            ...     "bw2": [12.0]
            ... })
            >>> loader = HoloWANLoader()
            >>> renamed_df = loader._rename_columns(df)
            >>> print(list(renamed_df.columns))
            ['delay_up', 'loss_up', 'bw_up', 'delay_down', 'loss_down', 'bw_down']
        """
        rename_map = {
            "delay1": "delay_up",
            "loss1": "loss_up",
            "bw1": "bw_up",
            "delay2": "delay_down",
            "loss2": "loss_down",
            "bw2": "bw_down"
        }

        return df.rename(columns=rename_map)

    def _add_raw_prefix(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加 raw_ 前缀到所有列。
        
        为原始值添加前缀，以便与后续处理后的归一化值区分。
        
        Args:
            df: 重命名后的 DataFrame
        
        Returns:
            pd.DataFrame: 添加前缀后的 DataFrame
            
        Examples:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     "delay_up": [100.0],
            ...     "loss_up": [0.1],
            ...     "bw_up": [10.0],
            ...     "delay_down": [95.0],
            ...     "loss_down": [0.0],
            ...     "bw_down": [12.0]
            ... })
            >>> loader = HoloWANLoader()
            >>> prefixed_df = loader._add_raw_prefix(df)
            >>> print(list(prefixed_df.columns))
            ['raw_delay_up', 'raw_loss_up', 'raw_bw_up', 'raw_delay_down', 'raw_loss_down', 'raw_bw_down']
        """
        rename_map = {col: f"raw_{col}" for col in df.columns}
        return df.rename(columns=rename_map)

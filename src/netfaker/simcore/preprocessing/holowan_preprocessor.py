"""HoloWAN 数据预处理器模块。

实现延迟归一化、丢包率缩放和带宽处理，生成标准化数据。
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class HoloWANPreprocessor:
    """HoloWAN 数据预处理器类。
    
    负责延迟归一化、丢包率缩放和带宽处理，生成标准化数据。
    """

    def __init__(self, shared_delay_scaler: bool = True):
        """初始化预处理器。
        
        Args:
            shared_delay_scaler: 是否使用共享的延迟 scaler
        """
        self.shared_delay_scaler = shared_delay_scaler
        self.global_scaler: Optional[StandardScaler] = None
        self.scaler_up: Optional[StandardScaler] = None
        self.scaler_down: Optional[StandardScaler] = None

    def fit_transform(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """拟合并转换所有数据。
        
        首先拟合 scaler（根据选择的模式），然后对所有数据进行转换。
        
        Args:
            data_dict: 文件名到 DataFrame 的映射
        
        Returns:
            Dict[str, pd.DataFrame]: 处理后的数据
            
        Examples:
            >>> import pandas as pd
            >>> data = {
            ...     'raw_delay_up': [100.0, 105.0, 98.0],
            ...     'raw_loss_up': [0.1, 0.2, 0.0],
            ...     'raw_bw_up': [10.0, 9.5, 10.5],
            ...     'raw_delay_down': [95.0, 98.0, 92.0],
            ...     'raw_loss_down': [0.0, 0.1, 0.0],
            ...     'raw_bw_down': [12.0, 11.5, 12.5]
            ... }
            >>> df = pd.DataFrame(data)
            >>> preprocessor = HoloWANPreprocessor(shared_delay_scaler=True)
            >>> processed_data = preprocessor.fit_transform({'test': df})
            >>> processed_df = processed_data['test']
            >>> print(list(processed_df.columns))
            ['raw_delay_up', 'raw_loss_up', 'raw_bw_up', 'raw_delay_down', 'raw_loss_down', 'raw_bw_down', 'norm_delay_up', 'norm_loss_up', 'norm_bw_up', 'norm_delay_down', 'norm_loss_down', 'norm_bw_down']
        """
        # 第一步：收集所有延迟数据用于拟合 scaler
        if self.shared_delay_scaler:
            self._fit_shared_scaler(data_dict)
        else:
            self._fit_separate_scalers(data_dict)

        # 第二步：转换所有数据
        processed_data = {}
        for filename, df in data_dict.items():
            processed_df = self.transform(df)
            processed_data[filename] = processed_df

        return processed_data

    def _fit_shared_scaler(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """拟合共享的全局 scaler。
        
        将所有 trace 的 delay_up 和 delay_down 合并为一个整体，拟合单一全局 scaler。
        
        Args:
            data_dict: 文件名到 DataFrame 的映射
        """
        # 收集所有延迟数据
        all_delay_data = []
        for df in data_dict.values():
            # 提取延迟数据
            delay_up = df["raw_delay_up"].values
            delay_down = df["raw_delay_down"].values

            # 应用 log(1 + x) 变换
            log_delay_up = np.log1p(delay_up)
            log_delay_down = np.log1p(delay_down)

            # 合并数据
            all_delay_data.extend(log_delay_up)
            all_delay_data.extend(log_delay_down)

        # 拟合 scaler
        self.global_scaler = StandardScaler()
        self.global_scaler.fit(np.array(all_delay_data).reshape(-1, 1))

    def _fit_separate_scalers(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """拟合独立的上行和下行 scaler。
        
        分别聚合所有 trace 的 delay_up 和 delay_down，拟合两个独立 scaler。
        
        Args:
            data_dict: 文件名到 DataFrame 的映射
        """
        # 收集上行延迟数据
        all_delay_up = []
        for df in data_dict.values():
            delay_up = df["raw_delay_up"].values
            log_delay_up = np.log1p(delay_up)
            all_delay_up.extend(log_delay_up)

        # 收集下行延迟数据
        all_delay_down = []
        for df in data_dict.values():
            delay_down = df["raw_delay_down"].values
            log_delay_down = np.log1p(delay_down)
            all_delay_down.extend(log_delay_down)

        # 拟合 scaler
        self.scaler_up = StandardScaler()
        self.scaler_up.fit(np.array(all_delay_up).reshape(-1, 1))

        self.scaler_down = StandardScaler()
        self.scaler_down.fit(np.array(all_delay_down).reshape(-1, 1))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换单个 DataFrame。
        
        对数据应用预处理步骤：延迟归一化、丢包率缩放和带宽保持。
        
        Args:
            df: 原始 DataFrame
        
        Returns:
            pd.DataFrame: 处理后的 DataFrame
            
        Examples:
            >>> import pandas as pd
            >>> data = {
            ...     'raw_delay_up': [100.0, 105.0],
            ...     'raw_loss_up': [0.1, 0.2],
            ...     'raw_bw_up': [10.0, 9.5],
            ...     'raw_delay_down': [95.0, 98.0],
            ...     'raw_loss_down': [0.0, 0.1],
            ...     'raw_bw_down': [12.0, 11.5]
            ... }
            >>> df = pd.DataFrame(data)
            >>> preprocessor = HoloWANPreprocessor(shared_delay_scaler=True)
            >>> # 先拟合
            >>> preprocessor.fit_transform({'test': df})
            >>> # 再转换
            >>> processed_df = preprocessor.transform(df)
            >>> print(f"原始延迟: {df['raw_delay_up'].tolist()}")
            原始延迟: [100.0, 105.0]
            >>> print(f"归一化延迟: {processed_df['norm_delay_up'].tolist()}")
            归一化延迟: [-1.0, 1.0]  # 示例值，实际值会根据数据分布变化
        """
        # 复制 DataFrame 避免修改原始数据
        processed_df = df.copy()

        # 处理延迟
        processed_df = self._process_delay(processed_df)

        # 处理丢包率
        processed_df = self._process_loss(processed_df)

        # 处理带宽（保持不变）
        processed_df = self._process_bandwidth(processed_df)

        return processed_df

    def _process_delay(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理延迟数据。
        
        对延迟数据应用 log(1 + x) 变换和 StandardScaler 标准化。
        
        Args:
            df: 输入 DataFrame
        
        Returns:
            pd.DataFrame: 处理后的 DataFrame
        """
        # 提取原始延迟数据
        delay_up = df["raw_delay_up"].values
        delay_down = df["raw_delay_down"].values

        # 应用 log(1 + x) 变换，缓解长尾分布
        log_delay_up = np.log1p(delay_up)
        log_delay_down = np.log1p(delay_down)

        # 标准化
        if self.shared_delay_scaler and self.global_scaler:
            norm_delay_up = self.global_scaler.transform(log_delay_up.reshape(-1, 1)).flatten()
            norm_delay_down = self.global_scaler.transform(log_delay_down.reshape(-1, 1)).flatten()
        elif not self.shared_delay_scaler and self.scaler_up and self.scaler_down:
            norm_delay_up = self.scaler_up.transform(log_delay_up.reshape(-1, 1)).flatten()
            norm_delay_down = self.scaler_down.transform(log_delay_down.reshape(-1, 1)).flatten()
        else:
            raise ValueError("Scaler not fitted yet")

        # 添加到 DataFrame
        df["norm_delay_up"] = norm_delay_up
        df["norm_delay_down"] = norm_delay_down

        return df

    def _process_loss(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理丢包率数据。
        
        对丢包率进行线性缩放，转换为 [0, 1] 范围。
        
        Args:
            df: 输入 DataFrame
        
        Returns:
            pd.DataFrame: 处理后的 DataFrame
        """
        # 提取原始丢包率数据
        loss_up = df["raw_loss_up"].values
        loss_down = df["raw_loss_down"].values

        # 线性缩放至 [0, 1] 范围
        norm_loss_up = loss_up / 100.0
        norm_loss_down = loss_down / 100.0

        # 添加到 DataFrame
        df["norm_loss_up"] = norm_loss_up
        df["norm_loss_down"] = norm_loss_down

        return df

    def _process_bandwidth(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理带宽数据（保持不变）。
        
        带宽数据保持原始值不变，不做任何变换。
        
        Args:
            df: 输入 DataFrame
        
        Returns:
            pd.DataFrame: 处理后的 DataFrame
        """
        # 直接复制原始值
        df["norm_bw_up"] = df["raw_bw_up"]
        df["norm_bw_down"] = df["raw_bw_down"]

        return df

    def get_scaler_info(self) -> Dict:
        """获取 scaler 信息。
        
        Returns:
            Dict: scaler 信息，包含模式和 scaler 实例
            
        Examples:
            >>> preprocessor = HoloWANPreprocessor(shared_delay_scaler=True)
            >>> # 先拟合
            >>> data = {'test': pd.DataFrame({'raw_delay_up': [100.0], 'raw_delay_down': [95.0]})}
            >>> preprocessor.fit_transform(data)
            >>> scaler_info = preprocessor.get_scaler_info()
            >>> print(scaler_info['mode'])
            shared
            >>> print('scaler' in scaler_info)
            True
        """
        if self.shared_delay_scaler:
            return {
                "mode": "shared",
                "scaler": self.global_scaler
            }
        else:
            return {
                "mode": "separate",
                "scaler_up": self.scaler_up,
                "scaler_down": self.scaler_down
            }

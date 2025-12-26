from typing import Dict, List, TypedDict

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer


class WindowMetaRaw(TypedDict):
    """原始窗口元数据
    
    Attributes:
        window: 窗口数据，包含列: timestamp, delay_up, delay_down, loss_up, loss_down
        is_first: 是否为首个窗口
        start_time: 窗口开始时间（秒级时间戳）
        trace_id: 轨迹标识符
    """
    window: pd.DataFrame
    is_first: bool
    start_time: float
    trace_id: str


class WindowMetaRenamed(TypedDict):
    """重命名后的窗口元数据（中间类型）
    
    Attributes:
        window: 窗口数据，包含列: timestamp, del_up, del_dn, loss_up, loss_dn
        is_first: 是否为首个窗口
        start_time: 窗口开始时间（秒级时间戳）
        trace_id: 轨迹标识符
    """
    window: pd.DataFrame
    is_first: bool
    start_time: float
    trace_id: str


class WindowMetaNormed(TypedDict):
    """归一化后的窗口元数据
    
    Attributes:
        window: 窗口数据，包含列: timestamp, del_up, del_dn, loss_up, loss_dn
        is_first: 是否为首个窗口
        start_time: 窗口开始时间（秒级时间戳）
        trace_id: 轨迹标识符
        cond: 23维条件向量
        keep: 是否保留该窗口（= not is_first）
    """
    window: pd.DataFrame
    is_first: bool
    start_time: float
    trace_id: str
    cond: np.ndarray
    keep: bool


class PreprocessingAssets(TypedDict):
    """预处理资源
    
    Attributes:
        qt_up: 上行延迟归一化模型
        qt_down: 下行延迟归一化模型
        init_local_cond: 初始局部条件向量
        cond_mean: 条件向量均值
        cond_std: 条件向量标准差
        mean_loss_cat2_up: 上行部分丢包均值（已废弃）
        mean_loss_cat2_dn: 下行部分丢包均值（已废弃）
        state_id_stats: 网络状态ID统计信息
    """
    qt_up: QuantileTransformer
    qt_down: QuantileTransformer
    init_local_cond: np.ndarray
    cond_mean: np.ndarray
    cond_std: np.ndarray
    mean_loss_cat2_up: float
    mean_loss_cat2_dn: float
    state_id_stats: Dict[str, Dict[int, Dict[str, float]]]


class PreprocessingResult(TypedDict):
    """预处理结果
    
    Attributes:
        datasets: 按数据集类型划分的归一化窗口元数据
        assets: 预处理资源
        stats: 预处理统计信息
    """
    datasets: Dict[str, List[WindowMetaNormed]]
    assets: PreprocessingAssets
    stats: Dict[str, any]


class PreprocessingConfig(TypedDict):
    """预处理配置
    
    Attributes:
        input_dir: 输入文件目录
        output_dir: 输出目录
        raw_interval_sec: 原始数据采样间隔（秒）
        window_size: 窗口大小
        step_size: 步长
        max_delay_ms: 最大延迟（毫秒）
        max_gap_sec: 最大时间间隙（秒）
        min_segment_rows: 最小段行数
        max_invalid_ratio: 最大无效比例
        split: 数据集划分比例
        quantile_transformer: 分位数转换器配置
        filename_pattern: 文件名模式
        default_network_state_id: 默认网络状态ID
        report_template: 报告模板路径
        output_dtype: 输出数据类型
    """
    input_dir: str
    output_dir: str
    raw_interval_sec: float
    window_size: int
    step_size: int
    max_delay_ms: int
    max_gap_sec: float
    min_segment_rows: int
    max_invalid_ratio: float
    split: Dict[str, float]
    quantile_transformer: Dict[str, any]
    filename_pattern: str
    default_network_state_id: int
    report_template: str
    output_dtype: str

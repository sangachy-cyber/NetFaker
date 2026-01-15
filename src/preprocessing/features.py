import numpy as np
import pandas as pd


def compute_window_features(window_df: pd.DataFrame) -> np.ndarray:
    """计算窗口的全局特征
    
    Args:
        window_df: 窗口DataFrame
        
    Returns:
        15维全局特征数组，第0维为网络状态ID，1-7为上行分位点，8-14为下行分位点
    """
    features = np.zeros(15)

    # 网络状态ID（默认0，后续会被行为发现模块覆盖）
    features[0] = 0.0  # network_state_id
    
    # 全局目标特征：只保留关键分位点
    # 使用7个关键分位点：1%,10%,25%,50%,75%,90%,99%
    quantiles = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    
    # 上行分位数特征
    up_quantiles = np.quantile(window_df["delay_up_rs"], quantiles)
    features[1:8] = up_quantiles
    
    # 下行分位数特征
    down_quantiles = np.quantile(window_df["delay_down_rs"], quantiles)
    features[8:15] = down_quantiles
    
    return features


def normalize_condition_vector(cond_vector: np.ndarray, delay_up_qt_mean: float = 0.0, delay_up_qt_std: float = 1.0, delay_down_qt_mean: float = 0.0, delay_down_qt_std: float = 1.0) -> np.ndarray:
    """直接返回条件向量，不进行Z-score标准化

    Args:
        cond_vector: 15维条件向量（只包含全局特征）
        delay_up_qt_mean: 已废弃参数，保持向后兼容
        delay_up_qt_std: 已废弃参数，保持向后兼容
        delay_down_qt_mean: 已废弃参数，保持向后兼容
        delay_down_qt_std: 已废弃参数，保持向后兼容
        
    Returns:
        15维条件向量，第0维为网络状态ID，1-7为上行分位点，8-14为下行分位点
    """
    # 直接返回条件向量，不做任何z-score标准化
    return cond_vector.copy()

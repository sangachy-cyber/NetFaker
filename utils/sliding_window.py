import numpy as np


def is_ul_dl_identical(window):
    ul_delay = window[:, 0]
    dl_delay = window[:, 2]
    return np.allclose(ul_delay, dl_delay, rtol=1e-5, atol=1e-8)


def has_high_static_ratio(window, static_threshold=0.1, delay_indices=[0, 2]):
    """
    检查窗口内是否有10%以上的连续相同时延
    Args:
        window: 窗口数据 (T, 4)
        static_threshold: 静态比例阈值，默认0.1（10%）
        delay_indices: 时延所在的列索引，默认[0, 2]（UL和DL时延）
    Returns:
        bool: 如果窗口内连续相同时延比例超过阈值，返回True，否则返回False
    """
    T = window.shape[0]
    for col in delay_indices:
        # 计算连续相同值的最大长度
        max_consecutive = 1
        current_consecutive = 1
        
        for i in range(1, T):
            if np.allclose(window[i, col], window[i-1, col], rtol=1e-5, atol=1e-8):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        
        # 检查是否超过阈值
        if max_consecutive / T > static_threshold:
            return True
    
    return False


def is_window_static(window):
    """
    检查窗口内的所有时间步是否都相同
    Args:
        window: 窗口数据 (T, 4)
    Returns:
        bool: 如果窗口内所有时间步都相同，返回True，否则返回False
    """
    # 检查窗口内的每个特征维度是否都没有变化
    for i in range(window.shape[1]):
        # 如果当前特征维度的所有值都相同，则继续检查下一个维度
        if np.allclose(window[:, i], window[0, i], rtol=1e-5, atol=1e-8):
            continue
        else:
            # 如果当前特征维度有变化，则窗口不是静态的
            return False
    # 如果所有特征维度都没有变化，则窗口是静态的
    return True


def generate_windows(
    long_sequence, window_size=100, stride=50,
    filter_identical=True, filter_static=True, filter_high_static_ratio=True,
    static_threshold=0.1, sample_interval_ms=100, max_identical_seconds=10
):
    """
    通用窗口生成函数，支持过滤连续相同的上下行窗口和静态窗口
    Args:
        long_sequence: 原始序列数据 (N, 4) 或类似结构
        window_size: 窗口大小
        stride: 滑动步长
        filter_identical: 是否过滤连续相同窗口
        filter_static: 是否过滤窗口内所有时间步都相同的静态窗口
        filter_high_static_ratio: 是否过滤窗口内连续相同时延比例超过阈值的窗口
        static_threshold: 静态比例阈值，默认0.1（10%）
        sample_interval_ms: 采样间隔（毫秒）
        max_identical_seconds: 最大允许的连续相同时长（秒）
    Returns:
        windows: list of 窗口数据
        window_starts: list of 窗口起始位置
    """
    T = len(long_sequence)
    if T < window_size:
        return [], []

    windows = []
    window_starts = []

    # 滑动切窗，生成所有窗口
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        win = long_sequence[start : start + window_size]
        
        # 过滤1: 静态窗口（所有时间步都相同）
        if filter_static and is_window_static(win):
            continue
        
        # 过滤2: 高静态比例窗口（10%以上连续相同）
        if filter_high_static_ratio and has_high_static_ratio(win, static_threshold):
            continue
        
        windows.append(win)
        window_starts.append(start)
    
    if (not filter_identical and not filter_static and not filter_high_static_ratio) or not windows:
        return windows, window_starts
    
    # 过滤连续相同的上下行窗口
    max_identical_ms = max_identical_seconds * 1000
    
    # 1. 标记每个窗口是否上下行相同
    is_identical_list = [is_ul_dl_identical(win) for win in windows]
    
    # 2. 检测连续的上下行相同窗口序列
    valid_indices = []
    i = 0
    n = len(windows)
    
    while i < n:
        if not is_identical_list[i]:
            # 不是相同窗口，直接保留
            valid_indices.append(i)
            i += 1
        else:
            # 找到连续相同窗口的起始位置
            start_idx = i
            # 统计连续相同的窗口数
            while i < n and is_identical_list[i]:
                i += 1
            end_idx = i
            
            # 计算连续相同窗口的总时长
            # 注意：窗口之间可能有重叠，需要计算实际覆盖的时间范围
            if start_idx < end_idx:
                # 计算第一个窗口的结束时间和最后一个窗口的开始时间
                first_start_time = window_starts[start_idx] * sample_interval_ms
                last_end_time = (window_starts[end_idx - 1] + window_size) * sample_interval_ms
                total_duration_ms = last_end_time - first_start_time
                
                if total_duration_ms <= max_identical_ms:
                    # 总时长不超过阈值，保留这些窗口
                    valid_indices.extend(range(start_idx, end_idx))
    
    # 3. 根据有效索引过滤窗口
    filtered_windows = [windows[i] for i in valid_indices]
    filtered_starts = [window_starts[i] for i in valid_indices]

    return filtered_windows, filtered_starts


def safe_sliding_window(
    long_sequence, window_size=100, stride=50, scaler=None, min_valid_windows=1,
    sample_interval_ms=100, max_identical_seconds=10
):
    """
    在单个长序列上安全滑动切窗，过滤连续10秒以上上下行相同的场景和静态窗口
    Args:
        long_sequence: 原始序列数据
        window_size: 窗口大小
        stride: 滑动步长
        scaler: 缩放器
        min_valid_windows: 最小有效窗口数
        sample_interval_ms: 采样间隔（毫秒）
        max_identical_seconds: 最大允许的连续相同时长（秒）
    Returns:
        valid_windows: list of (100,4)
        labels: list of str
        conditions: list of (3,)
    """
    from utils.condition_extractor import (
        assign_hybrid_label,
        extract_continuous_condition,
    )

    # 使用通用窗口生成函数
    all_windows, all_starts = generate_windows(
        long_sequence, window_size, stride,
        filter_identical=True, filter_static=True, filter_high_static_ratio=True,
        static_threshold=0.1, sample_interval_ms=sample_interval_ms,
        max_identical_seconds=max_identical_seconds
    )
    
    if not all_windows:
        return [], [], []
    
    # 分配标签和提取条件
    windows = []
    labels = []
    conditions = []
    
    for win in all_windows:
        # 分配标签
        label = assign_hybrid_label(win, scaler)
        if label is None:
            continue
        
        # 提取连续条件
        if label == "S0":
            cond = np.array([0.0, 0.0, 0.0])
        elif label == "S1_light":
            cond = extract_continuous_condition(win, mode="light")
        else:  # S1_anomaly
            cond = extract_continuous_condition(win, mode="anomaly")
        
        windows.append(win)
        labels.append(label)
        conditions.append(cond)

    # 至少保留 min_valid_windows 个窗口（防极端过滤）
    if len(windows) < min_valid_windows:
        return [], [], []

    return windows, labels, conditions

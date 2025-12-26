from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .features import compute_window_features, compute_local_features
from sklearn.preprocessing import QuantileTransformer



def parse_txt(txt_path: Path, interval_sec: float) -> pd.DataFrame:
    """解析原始 txt 文件为标准化 DataFrame
    
    Args:
        txt_path: txt文件路径
        interval_sec: 间隔秒数
        
    Returns:
        标准化 DataFrame，包含列: [timestamp, delay_up_origin, delay_down_origin, loss_up_origin, loss_down_origin, delay_up, delay_down, loss_up, loss_down]
        
    Raises:
        ValueError: 当文件中找不到Start Time时抛出
    """
    with open(txt_path) as f:
        lines = f.readlines()

    # 查找开始位置和Start Time
    start_idx, start_time_line = _find_start_info(lines)
    
    # 解析Start Time
    start_timestamp = _parse_start_time(start_time_line, txt_path)

    # 解析数据行
    data_lines = lines[start_idx:]
    timestamps, delay_up, loss_up, delay_down, loss_down = _parse_data_lines(
        data_lines, start_timestamp, interval_sec
    )

    # 构造 DataFrame，同时包含原始值和处理后的值
    df = pd.DataFrame({
        "timestamp": timestamps,
        "delay_up_origin": delay_up,
        "delay_down_origin": delay_down,
        "loss_up_origin": loss_up,
        "loss_down_origin": loss_down,
        # 初始值与原始值相同，后续会被归一化或二值化
        "delay_up": delay_up,
        "delay_down": delay_down,
        "loss_up": loss_up,
        "loss_down": loss_down,
    })

    return df


def _find_start_info(lines: List[str]) -> Tuple[int, str]:
    """查找开始位置和Start Time
    
    Args:
        lines: 文件行列表
        
    Returns:
        (start_idx, start_time_line) 二元组
    """
    start_idx = 0
    start_time_line = None

    for i, line in enumerate(lines):
        if line.startswith("Start Time:"):
            start_time_line = line
        elif line.strip() == "------------------------------------------------":
            start_idx = i + 1
            break
    
    return start_idx, start_time_line


def _parse_start_time(start_time_line: str, txt_path: Path) -> float:
    """解析Start Time
    
    Args:
        start_time_line: Start Time行
        txt_path: 文件路径
        
    Returns:
        开始时间戳
        
    Raises:
        ValueError: 当文件中找不到Start Time时抛出
    """
    if start_time_line:
        start_time_str = start_time_line.split(":", 1)[1].strip()
        # 处理可能的时间格式问题
        try:
            start_timestamp = datetime.fromisoformat(start_time_str).timestamp()
        except ValueError:
            # 尝试其他格式
            start_timestamp = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S").timestamp()
    else:
        raise ValueError(f"无法在文件 {txt_path} 中找到 Start Time")
    
    return start_timestamp


def _parse_data_lines(
    data_lines: List[str], 
    start_timestamp: float, 
    interval_sec: float
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    """解析数据行
    
    Args:
        data_lines: 数据行列表
        start_timestamp: 开始时间戳
        interval_sec: 间隔秒数
        
    Returns:
        (timestamps, delay_up, loss_up, delay_down, loss_down) 五元组
    """
    timestamps = []
    delay_up = []
    loss_up = []
    delay_down = []
    loss_down = []

    for i, line in enumerate(data_lines):
        if not line.strip():
            continue

        parts = line.strip().split(",")
        if len(parts) < 6:
            continue

        # 解析单行数据
        parsed_data = _parse_single_line(parts, start_timestamp, i, interval_sec)
        if parsed_data is not None:
            ts, du, lu, dd, ld = parsed_data
            timestamps.append(ts)
            delay_up.append(du)
            loss_up.append(lu)
            delay_down.append(dd)
            loss_down.append(ld)

    return timestamps, delay_up, loss_up, delay_down, loss_down


def _parse_single_line(
    parts: List[str], 
    start_timestamp: float, 
    line_idx: int, 
    interval_sec: float
) -> Tuple[float, float, float, float, float] or None:
    """解析单行数据
    
    Args:
        parts: 行数据分割后的列表
        start_timestamp: 开始时间戳
        line_idx: 行索引
        interval_sec: 间隔秒数
        
    Returns:
        解析后的数据元组，解析失败返回None
        
    Raises:
        ValueError: 当原始数据中存在负时延时抛出
    """
    # 解析各字段
    try:
        delay1 = float(parts[0])
        loss1_percent = float(parts[1])
        bandwidth1 = float(parts[2])
        delay2 = float(parts[3])
        loss2_percent = float(parts[4])
        bandwidth2 = float(parts[5])
    except ValueError:
        return None

    # 计算 timestamp
    timestamp = start_timestamp + line_idx * interval_sec

    # 计算 delay 和 loss (delay保持毫秒单位)
    # 检查时延是否为负，如果为负则抛出异常
    if delay1 < 0 or delay2 < 0:
        raise ValueError(f"原始数据中存在负时延：delay1={delay1}, delay2={delay2}")
    
    delay_up_val = delay1  # 直接使用毫秒单位，不转换为秒
    delay_down_val = delay2  # 直接使用毫秒单位，不转换为秒

    # loss 计算：如果带宽为0，则loss为1.0，否则为百分比/100
    loss_up_val = 1.0 if bandwidth1 == 0 else loss1_percent / 100.0
    loss_down_val = 1.0 if bandwidth2 == 0 else loss2_percent / 100.0

    return timestamp, delay_up_val, loss_up_val, delay_down_val, loss_down_val


def clean_and_truncate(df: pd.DataFrame, max_delay_ms: int) -> pd.DataFrame:
    """清洗并截断数据帧
    
    Args:
        df: 输入的DataFrame
        max_delay_ms: 最大延迟毫秒数
        
    Returns:
        清洗后的DataFrame
    """
    # 按 timestamp 排序、去重
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    # 将 delay < 0 或 loss ∉ [0,1] 设为 NaN
    df.loc[df["delay_up"] < 0, "delay_up"] = np.nan
    df.loc[df["delay_down"] < 0, "delay_down"] = np.nan
    df.loc[(df["loss_up"] < 0) | (df["loss_up"] > 1), "loss_up"] = np.nan
    df.loc[(df["loss_down"] < 0) | (df["loss_down"] > 1), "loss_down"] = np.nan

    # 从前往后扫描，首次出现 delay_up_origin ≥ max_delay_ms 或 delay_down_origin ≥ max_delay_ms → 丢弃该行及之后所有行
    truncate_idx = len(df)
    for i in range(len(df)):
        if df.iloc[i]["delay_up_origin"] >= max_delay_ms or df.iloc[i]["delay_down_origin"] >= max_delay_ms:
            truncate_idx = i
            break

    df = df.iloc[:truncate_idx]

    # 删除含 NaN 的行
    df = df.dropna().reset_index(drop=True)

    return df


def split_and_resample(
    df: pd.DataFrame,
    max_gap_sec: float,
    min_rows: int,
    freq_hz: int = 10,
    max_invalid_ratio: float = 0.01,
) -> List[pd.DataFrame]:
    """分割并重新采样数据
    
    Args:
        df: 输入的DataFrame
        max_gap_sec: 最大间隙秒数
        min_rows: 最小行数
        freq_hz: 频率Hz
        max_invalid_ratio: 最大无效比例
        
    Returns:
        分割后的DataFrame列表
    """
    if len(df) == 0:
        return []

    # 时间戳对齐（提升精度）
    df = df.copy()
    df["timestamp"] = np.round(df["timestamp"] / 0.1) * 0.1

    # 去重（防止 round 后时间戳重复）
    df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    if len(df) == 0:
        return []

    # 计算 gap = timestamp.diff()，若 gap > max_gap_sec → 切段
    df["gap"] = df["timestamp"].diff()
    segment_starts = [0]
    for i in range(1, len(df)):
        if df.iloc[i]["gap"] > max_gap_sec:
            segment_starts.append(i)

    segment_starts.append(len(df))

    # 提取所有有效段
    segments = []
    for i in range(len(segment_starts) - 1):
        start_idx = segment_starts[i]
        end_idx = segment_starts[i + 1]
        segment = df.iloc[start_idx:end_idx].copy()

        # 仅保留长度 ≥ min_rows 的段
        if len(segment) >= min_rows:
            # 删除临时gap列
            segment = segment.drop(columns=["gap"])
            segments.append(segment)

    if not segments:
        return []

    # 重采样至指定频率
    resampled_segments = []
    for segment in segments:
        resampled = resample_segment(segment, freq_hz, max_invalid_ratio)
        if resampled is not None:
            resampled_segments.append(resampled)

    return resampled_segments


def resample_segment(segment: pd.DataFrame, freq_hz: int, max_invalid_ratio: float) -> pd.DataFrame:
    """对单个数据段进行重采样
    
    Args:
        segment: 单个数据段
        freq_hz: 频率Hz
        max_invalid_ratio: 最大无效比例
        
    Returns:
        重采样后的数据段，若无效比例超过阈值则返回None
    """
    # 设置时间戳为索引并转换为datetime
    segment = segment.set_index("timestamp")
    segment.index = pd.to_datetime(segment.index, unit="s")

    # 重采样至指定频率 (10Hz = 0.1秒间隔)
    target_freq = f"{int(1000/freq_hz)}ms"  # 100ms for 10Hz
    resampled = segment.resample(target_freq).asfreq()

    # 对原始延迟使用线性插值
    resampled["delay_up_origin"] = resampled["delay_up_origin"].interpolate(method="linear")
    resampled["delay_down_origin"] = resampled["delay_down_origin"].interpolate(method="linear")
    # 对归一化延迟使用线性插值
    resampled["delay_up"] = resampled["delay_up"].interpolate(method="linear")
    resampled["delay_down"] = resampled["delay_down"].interpolate(method="linear")

    # 对原始丢包率使用前向填充
    resampled["loss_up_origin"] = resampled["loss_up_origin"].ffill().bfill()
    resampled["loss_down_origin"] = resampled["loss_down_origin"].ffill().bfill()
    # 对归一化丢包率使用前向填充
    resampled["loss_up"] = resampled["loss_up"].ffill().bfill()
    resampled["loss_down"] = resampled["loss_down"].ffill().bfill()
    
    # 检查非法值比例：loss ∉ [0,1]
    invalid_up = ((resampled["loss_up"] < 0) | (resampled["loss_up"] > 1)).sum()
    invalid_down = ((resampled["loss_down"] < 0) | (resampled["loss_down"] > 1)).sum()
    total_values = len(resampled) * 2  # up and down
    invalid_ratio = (invalid_up + invalid_down) / total_values if total_values > 0 else 0

    # 若重采样后非法比例 > max_invalid_ratio → 整段丢弃
    if invalid_ratio > max_invalid_ratio:
        return None
    
    # 否则：clip loss 到 [0.0, 1.0]
    resampled["loss_up"] = np.clip(resampled["loss_up"], 0.0, 1.0)
    resampled["loss_down"] = np.clip(resampled["loss_down"], 0.0, 1.0)

    # 重置索引，使 timestamp 回到列中
    resampled = resampled.reset_index()
    # 转换回时间戳
    resampled["timestamp"] = resampled["timestamp"].astype("int64") // 10**9
    
    return resampled


def extract_windows(segments: List[pd.DataFrame], window: int, step: int) -> List[pd.DataFrame]:
    """从段中提取窗口
    
    Args:
        segments: DataFrame段列表
        window: 窗口大小
        step: 步长
        
    Returns:
        窗口DataFrame列表
    """
    windows = []

    for segment in segments:
        # 使用滑动窗口提取数据
        for i in range(0, len(segment) - window + 1, step):
            window_df = segment.iloc[i:i + window].copy().reset_index(drop=True)
            windows.append(window_df)

    return windows


def mark_first_window(windows: List[pd.DataFrame], trace_id: str) -> List[Dict]:
    """标记首个窗口
    
    Args:
        windows: 窗口DataFrame列表
        trace_id: trace标识符
        
    Returns:
        WindowMetaRaw列表
    """
    window_metas = []

    for i, window_df in enumerate(windows):
        is_first = (i == 0)
        start_time = window_df["timestamp"].iloc[0] if len(window_df) > 0 else 0.0

        meta = {
            "window": window_df,
            "is_first": is_first,
            "start_time": start_time,
            "trace_id": trace_id,
        }

        window_metas.append(meta)

    return window_metas


def group_by_trace_id(all_metas: List[Dict]) -> Dict[str, List[Dict]]:
    """按 trace_id 分组
    
    Args:
        all_metas: 所有WindowMetaRaw列表
        
    Returns:
        按trace_id分组的字典
    """
    traces_dict = {}

    for meta in all_metas:
        trace_id = meta["trace_id"]
        if trace_id not in traces_dict:
            traces_dict[trace_id] = []
        traces_dict[trace_id].append(meta)

    return traces_dict


def assign_split_by_trace(
    traces: Dict[str, List],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[List, List, List]:
    """按 trace 分配训练/验证/测试集
    
    Args:
        traces: 按trace_id分组的字典
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        random_state: 随机种子
        
    Returns:
        (train_metas, val_metas, test_metas) 三元组
    """
    # 按 trace_id 随机打乱后划分
    trace_ids = list(traces.keys())
    rng = np.random.default_rng(random_state)
    rng.shuffle(trace_ids)

    n = len(trace_ids)

    # 确保至少有一个trace被分配到训练集
    if n == 1:
        # 如果只有一个trace，将其分配给训练集
        n_train = 1
        n_val = 0
    else:
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        # 确保至少有一个trace被分配到训练集
        if n_train == 0 and n > 0:
            n_train = 1

        # 调整验证集和测试集的数量
        remaining = n - n_train
        n_val = min(n_val, remaining)

    train_ids = set(trace_ids[:n_train])
    val_ids = set(trace_ids[n_train:n_train+n_val]) if n_val > 0 else set()
    test_ids = set(trace_ids[n_train+n_val:]) if n_train+n_val < n else set()

    # 分配
    train_metas = [meta for tid in train_ids for meta in traces[tid]]
    val_metas = [meta for tid in val_ids for meta in traces[tid]]
    test_metas = [meta for tid in test_ids for meta in traces[tid]]

    return train_metas, val_metas, test_metas


def fit_and_normalize(
    train, val, test, random_state: int = 42
) -> Tuple[Dict[str, List[Dict]], Dict]:
    """拟合并标准化数据
    
    使用训练集数据拟合QuantileTransformer，然后对所有数据集应用标准化变换。
    
    Args:
        train: 训练集
        val: 验证集
        test: 测试集
        random_state: 随机种子
        
    Returns:
        (renamed_datasets, assets) 二元组
        
    Raises:
        ValueError: 当训练集为空时抛出
    """
    # 收集所有训练数据用于拟合
    train_windows = [meta["window"] for meta in train]
    if not train_windows:
        raise ValueError("训练集为空")

    # 提取特征用于拟合 QuantileTransformer
    train_features_list = []
    for window_df in train_windows:
        # 提取原始延迟特征用于拟合
        delays_up = window_df["delay_up_origin"].values
        delays_down = window_df["delay_down_origin"].values
        train_features_list.append(np.column_stack([delays_up, delays_down]))

    train_features = np.vstack(train_features_list)

    # 拟合 QuantileTransformer
    qt_up = QuantileTransformer(
        output_distribution="normal",
        random_state=random_state,
        n_quantiles=min(1000, len(train_features)),
        subsample=min(100000, len(train_features)),
    )
    qt_up.fit(train_features[:, 0].reshape(-1, 1))

    qt_down = QuantileTransformer(
        output_distribution="normal",
        random_state=random_state,
        n_quantiles=min(1000, len(train_features)),
        subsample=min(100000, len(train_features)),
    )
    qt_down.fit(train_features[:, 1].reshape(-1, 1))

    # 处理数据集
    def process_dataset(dataset):
        renamed_metas = []
        for meta in dataset:
            window_df = meta["window"].copy()

            # 使用原始延迟列进行归一化，结果保存到 delay_up/delay_down 列
            window_df["delay_up"] = qt_up.transform(window_df["delay_up_origin"].values.reshape(-1, 1)).flatten()
            window_df["delay_down"] = qt_down.transform(window_df["delay_down_origin"].values.reshape(-1, 1)).flatten()
            # 重命名loss列
            window_df = window_df.rename(columns={
                "loss_down": "loss_dn",
            })

            # 构造新的元数据
            renamed_meta = {
                "window": window_df,
                "is_first": meta["is_first"],
                "start_time": meta["start_time"],
                "trace_id": meta["trace_id"],
            }

            renamed_metas.append(renamed_meta)

        return renamed_metas

    # 处理所有数据集
    train_renamed = process_dataset(train)
    val_renamed = process_dataset(val)
    test_renamed = process_dataset(test)

    # 构造返回结果
    renamed_datasets = {
        "train": train_renamed,
        "val": val_renamed,
        "test": test_renamed,
    }

    assets = {
        "qt_up": qt_up,
        "qt_down": qt_down,
    }

    return renamed_datasets, assets


def prepare_init_local_cond(train_dataset: List[Dict]) -> np.ndarray:
    """准备初始局部条件向量
    
    Args:
        train_dataset: 训练数据集
        
    Returns:
        10维初始局部条件向量
    """
    # 收集训练集中所有可能作为前驱的窗口（即每个trace中除最后一个窗口外的所有窗口）
    predecessor_windows = []
    train_traces = {}

    # 按trace_id分组
    for meta in train_dataset:
        trace_id = meta["trace_id"]
        if trace_id not in train_traces:
            train_traces[trace_id] = []
        train_traces[trace_id].append(meta)

    # 对每个trace，取除最后一个窗口外的所有窗口
    for trace_id, metas in train_traces.items():
        # 按 start_time 排序
        metas_sorted = sorted(metas, key=lambda x: x["start_time"])
        # 取除最后一个外的所有窗口
        predecessor_metas = metas_sorted[:-1] if len(metas_sorted) > 1 else metas_sorted
        for meta in predecessor_metas:
            predecessor_windows.append(meta["window"])

    # 计算局部特征
    local_features_list = []
    for window_df in predecessor_windows:
        local_features = compute_local_features(window_df)
        local_features_list.append(local_features)

    # 计算均值作为 init_local_cond
    if local_features_list:
        init_local_cond = np.mean(local_features_list, axis=0)
    else:
        init_local_cond = np.zeros(10)

    return init_local_cond


def prepare_norm_params(train_dataset: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """准备标准化参数
    
    Args:
        train_dataset: 训练数据集
        
    Returns:
        (cond_mean, cond_std) 二元组
    """
    # 计算训练集的全局特征用于标准化
    global_features_list = []
    local_features_list_for_norm = []
    for meta in train_dataset:
        if not meta["is_first"]:  # 跳过首窗口
            global_features = compute_window_features(meta["window"])
            # 使用默认网络状态ID 0
            network_state_id = 0
            global_features[11] = float(network_state_id)
            # 只取前12维（跳过网络状态ID维度11）
            selected_global_features = np.concatenate([global_features[:11], [global_features[12]]])
            global_features_list.append(selected_global_features)

            # 收集局部特征用于标准化
            local_features = compute_local_features(meta["window"])
            local_features_list_for_norm.append(local_features)

    # 计算均值和标准差用于标准化（22维特征：12维全局 + 10维局部）
    if global_features_list and local_features_list_for_norm:
        global_features_array = np.array(global_features_list)
        local_features_array = np.array(local_features_list_for_norm)
        # 合并全局和局部特征用于计算标准化参数
        all_features_for_norm = np.concatenate([global_features_array, local_features_array], axis=1)
        cond_mean = np.mean(all_features_for_norm, axis=0)
        cond_std = np.std(all_features_for_norm, axis=0)
        # 避免除零错误
        cond_std = np.where(cond_std == 0, 1.0, cond_std)
    else:
        cond_mean = np.zeros(22)
        cond_std = np.ones(22)

    return cond_mean, cond_std

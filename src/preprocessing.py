from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer


class WindowMetaRaw(TypedDict):
    window: pd.DataFrame          # columns: timestamp, delay_up, delay_down, loss_up, loss_down
    is_first: bool
    start_time: float             # seconds since epoch
    trace_id: str


class WindowMetaRenamed(TypedDict):  # ← 新增中间类型（v1.3）
    window: pd.DataFrame          # columns: timestamp, del_up, del_dn, loss_up, loss_dn
    is_first: bool
    start_time: float
    trace_id: str


class WindowMetaNormed(TypedDict):
    window: pd.DataFrame          # columns: timestamp, del_up, del_dn, loss_up, loss_dn
    is_first: bool
    start_time: float
    trace_id: str
    cond: np.ndarray              # shape=(23,) → [22 floats + 1 int (stored as float)]
    keep: bool                    # = not is_first


def stage1_parse_txt(txt_path: Path, interval_sec: float) -> pd.DataFrame:
    """解析原始 txt 文件为标准化 DataFrame
    
    该函数读取HoloWAN探针生成的原始文本文件，解析其中的网络性能数据，
    并将其转换为标准化的DataFrame格式。
    
    Example:
        >>> df = stage1_parse_txt(Path("data/sample.txt"), 0.1)
        >>> print(df.head())
           timestamp  delay_up  loss_up  delay_down  loss_down
        0  1764778324     0.012      0.0       0.015        0.0
        1  1764778325     0.014      0.0       0.016        0.0
    
    Args:
        txt_path: txt文件路径
        interval_sec: 间隔秒数
        
    Returns:
        标准化 DataFrame，包含列: ["timestamp", "delay_up", "loss_up", "delay_down", "loss_down"]
        
    Raises:
        ValueError: 当文件中找不到Start Time时抛出

    """
    with open(txt_path) as f:
        lines = f.readlines()

    # 查找开始位置和Start Time
    start_idx = 0
    start_time_line = None

    for i, line in enumerate(lines):
        if line.startswith("Start Time:"):
            start_time_line = line
        elif line.strip() == "------------------------------------------------":
            start_idx = i + 1
            break

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

    # 解析数据行
    data_lines = lines[start_idx:]
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

        # 解析各字段
        try:
            delay1 = float(parts[0])
            loss1_percent = float(parts[1])
            bandwidth1 = float(parts[2])
            delay2 = float(parts[3])
            loss2_percent = float(parts[4])
            bandwidth2 = float(parts[5])
        except ValueError:
            continue

        # 计算 timestamp
        timestamp = start_timestamp + i * interval_sec
        timestamps.append(timestamp)

        # 计算 delay 和 loss (转换delay为秒)
        delay_up_val = delay1 / 1000.0
        delay_down_val = delay2 / 1000.0

        # loss 计算：如果带宽为0，则loss为1.0，否则为百分比/100
        loss_up_val = 1.0 if bandwidth1 == 0 else loss1_percent / 100.0
        loss_down_val = 1.0 if bandwidth2 == 0 else loss2_percent / 100.0

        delay_up.append(delay_up_val)
        loss_up.append(loss_up_val)
        delay_down.append(delay_down_val)
        loss_down.append(loss_down_val)

    # 构造 DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "delay_up": delay_up,
        "loss_up": loss_up,
        "delay_down": delay_down,
        "loss_down": loss_down,
    })

    return df

def stage2_clean_and_truncate(df: pd.DataFrame, max_delay_ms: int) -> pd.DataFrame:
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

    # 从前往后扫描，首次出现 delay_up ≥ max_delay_ms 或 delay_down ≥ max_delay_ms → 丢弃该行及之后所有行
    max_delay_sec = max_delay_ms / 1000.0
    truncate_idx = len(df)
    for i in range(len(df)):
        if df.iloc[i]["delay_up"] >= max_delay_sec or df.iloc[i]["delay_down"] >= max_delay_sec:
            truncate_idx = i
            break

    df = df.iloc[:truncate_idx]

    # 删除含 NaN 的行
    df = df.dropna().reset_index(drop=True)

    return df


def stage3_split_and_resample(
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

    segments = []
    for i in range(len(segment_starts) - 1):
        start_idx = segment_starts[i]
        end_idx = segment_starts[i + 1]
        segment = df.iloc[start_idx:end_idx].copy()

        # 仅保留长度 ≥ min_rows 的段
        if len(segment) >= min_rows:
            segments.append(segment)

    if not segments:
        return []

    # 重采样至指定频率
    resampled_segments = []
    for segment in segments:
        # 设置时间戳为索引并转换为datetime
        segment = segment.set_index("timestamp")
        segment.index = pd.to_datetime(segment.index, unit="s")

        # 重采样至指定频率 (10Hz = 0.1秒间隔)
        target_freq = f"{int(1000/freq_hz)}ms"  # 100ms for 10Hz
        resampled = segment.resample(target_freq).asfreq()

        # 对 delay 使用线性插值
        resampled["delay_up"] = resampled["delay_up"].interpolate(method="linear")
        resampled["delay_down"] = resampled["delay_down"].interpolate(method="linear")

        # 对 loss 使用前向填充
        resampled["loss_up"] = resampled["loss_up"].ffill().bfill()
        resampled["loss_down"] = resampled["loss_down"].ffill().bfill()
        # 检查非法值比例：loss ∉ [0,1]
        invalid_up = ((resampled["loss_up"] < 0) | (resampled["loss_up"] > 1)).sum()
        invalid_down = ((resampled["loss_down"] < 0) | (resampled["loss_down"] > 1)).sum()
        total_values = len(resampled) * 2  # up and down
        invalid_ratio = (invalid_up + invalid_down) / total_values if total_values > 0 else 0

        # 若重采样后非法比例 > max_invalid_ratio → 整段丢弃
        if invalid_ratio <= max_invalid_ratio:
            # 否则：clip loss 到 [0.0, 1.0]
            resampled["loss_up"] = np.clip(resampled["loss_up"], 0.0, 1.0)
            resampled["loss_down"] = np.clip(resampled["loss_down"], 0.0, 1.0)

            # 重置索引，使 timestamp 回到列中
            resampled = resampled.reset_index()
            # 转换回时间戳
            resampled["timestamp"] = resampled["timestamp"].astype("int64") // 10**9
            resampled_segments.append(resampled)

    return resampled_segments

def stage4_extract_windows(segments: List[pd.DataFrame], window: int, step: int) -> List[pd.DataFrame]:
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


def stage5_mark_first_window(windows: List[pd.DataFrame], trace_id: str) -> List[WindowMetaRaw]:
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

        meta: WindowMetaRaw = {
            "window": window_df,
            "is_first": is_first,
            "start_time": start_time,
            "trace_id": trace_id,
        }

        window_metas.append(meta)

    return window_metas


def stage6_group_by_trace_id(all_metas: List[WindowMetaRaw]) -> Dict[str, List[WindowMetaRaw]]:
    """按 trace_id 分组
    
    Args:
        all_metas: 所有WindowMetaRaw列表
        
    Returns:
        按trace_id分组的字典

    """
    traces_dict: Dict[str, List[WindowMetaRaw]] = {}

    for meta in all_metas:
        trace_id = meta["trace_id"]
        if trace_id not in traces_dict:
            traces_dict[trace_id] = []
        traces_dict[trace_id].append(meta)

    return traces_dict


def stage7_assign_split_by_trace(
    traces: Dict[str, List],
    train_ratio=0.8,
    val_ratio=0.1,
    random_state=42,
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

def stage8_fit_and_normalize(
    train,
    val,
    test,
    random_state=42,
) -> Tuple[Dict[str, List[WindowMetaRenamed]], Dict]:
    """拟合并标准化数据
    
    使用训练集数据拟合QuantileTransformer，然后对所有数据集应用标准化变换。
    同时将列名从delay_*重命名为del_*。
    
    Example:
        >>> renamed_datasets, assets = stage8_fit_and_normalize(train_data, val_data, test_data)
        >>> print(list(renamed_datasets.keys()))
        ['train', 'val', 'test']
        >>> print(list(assets.keys()))
        ['qt_up', 'qt_down']
    
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
        # 提取 delay 特征
        delays_up = window_df["delay_up"].values
        delays_down = window_df["delay_down"].values
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
    def process_dataset(dataset, dataset_name):
        renamed_metas = []
        for meta in dataset:
            window_df = meta["window"].copy()

            # 重命名列
            window_df = window_df.rename(columns={
                "delay_up": "del_up",
                "delay_down": "del_dn",
                "loss_up": "loss_up",
                "loss_down": "loss_dn",
            })
            # 应用变换
            window_df["del_up"] = qt_up.transform(window_df["del_up"].values.reshape(-1, 1)).flatten()
            window_df["del_dn"] = qt_down.transform(window_df["del_dn"].values.reshape(-1, 1)).flatten()

            # 构造新的元数据
            renamed_meta: WindowMetaRenamed = {
                "window": window_df,
                "is_first": meta["is_first"],
                "start_time": meta["start_time"],
                "trace_id": meta["trace_id"],
            }

            renamed_metas.append(renamed_meta)

        return renamed_metas

    # 处理所有数据集
    train_renamed = process_dataset(train, "train")
    val_renamed = process_dataset(val, "val")
    test_renamed = process_dataset(test, "test")

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


def _compute_window_features(window_df):
    """计算窗口的全局特征
    
    Args:
        window_df: 窗口DataFrame
        
    Returns:
        13维全局特征数组

    """
    features = np.zeros(13)

    # 全局目标特征（13维中的前13个）
    features[0] = np.mean(window_df["del_up"])   # mean_del_up
    features[1] = np.std(window_df["del_up"])    # std_del_up
    features[2] = np.percentile(window_df["del_up"], 95)  # p95_del_up
    features[3] = np.mean(window_df["del_dn"])   # mean_del_dn
    features[4] = np.std(window_df["del_dn"])    # std_del_dn
    features[5] = np.percentile(window_df["del_dn"], 95)  # p95_del_dn
    # 简化丢包分类，只保留有意义的特征
    # 由于loss_up和loss_dn只有0和1两个值，frac_cat1等于均值
    features[6] = np.mean(window_df["loss_up"])    # frac_cat1_up (等于均值)
    features[7] = 0.0  # frac_cat2_up (废弃)
    features[8] = 0.0  # 保留位置但废弃
    features[9] = np.mean(window_df["loss_dn"])    # frac_cat1_dn (等于均值)
    features[10] = 0.0  # frac_cat2_dn (废弃)
    # features[11] 网络状态ID将在后面设置
    features[12] = 0.0  # reserved

    return features
def _compute_local_features(window_df, last_n=5):
    """计算窗口的局部特征（最后n行）
    
    Args:
        window_df: 窗口DataFrame
        last_n: 使用最后几行计算特征
        
    Returns:
        10维局部特征数组

    """
    if len(window_df) < last_n:
        # 如果窗口行数不足，使用全部数据并发出警告
        last_rows = window_df
    else:
        last_rows = window_df.tail(last_n)

    features = np.zeros(10)

    # 局部连续性特征（10维）
    features[0] = np.mean(last_rows["del_up"])   # prev_mean_del_up
    features[1] = np.std(last_rows["del_up"])    # prev_std_del_up
    features[2] = np.mean(last_rows["loss_up"])    # prev_frac_cat1_up (等于均值)
    features[3] = 0.0  # prev_frac_cat2_up (废弃)
    features[4] = 0.0  # 保留位置但废弃
    features[5] = np.mean(last_rows["del_dn"])   # prev_mean_del_dn
    features[6] = np.std(last_rows["del_dn"])    # prev_std_del_dn
    features[7] = np.mean(last_rows["loss_dn"])    # prev_frac_cat1_dn (等于均值)
    features[8] = 0.0  # prev_frac_cat2_dn (废弃)
    features[9] = 0.0  # prev_reserved

    return features


def stage9_recompute_condition_vectors(    datasets: Dict[str, List[WindowMetaRenamed]],
    assets: Dict,
    network_state_map: Dict[str, int],
) -> Tuple[Dict[str, List[WindowMetaNormed]], Dict]:
    """重新计算条件向量
    
    基于归一化数据计算23维条件向量，并进行Z-score标准化。
    条件向量包括13维全局特征和10维局部特征。
    
    Example:
        >>> final_datasets, extra_assets = stage9_recompute_condition_vectors(
        ...     renamed_datasets, assets, network_state_map)
        >>> train_data = final_datasets['train']
        >>> print(train_data[0]['cond'].shape)
        (23,)
    
    Args:
        datasets: 重命名后的数据集
        assets: 资源字典
        network_state_map: 网络状态映射
        
    Returns:
        (final_datasets, extra_assets) 二元组

    """
    # 首先在训练集上计算 init_local_cond.npy
    train_dataset = datasets["train"]

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
        local_features = _compute_local_features(window_df)
        local_features_list.append(local_features)

    # 计算均值作为 init_local_cond
    if local_features_list:
        init_local_cond = np.mean(local_features_list, axis=0)
    else:
        init_local_cond = np.zeros(10)

    # 计算训练集的全局特征用于标准化
    global_features_list = []
    local_features_list_for_norm = []
    for meta in train_dataset:
        if not meta["is_first"]:  # 跳过首窗口
            global_features = _compute_window_features(meta["window"])
            # 设置网络状态ID（维度11）
            network_state_id = network_state_map.get(meta["trace_id"], 0)
            global_features[11] = float(network_state_id)
            # 只取前12维（跳过网络状态ID维度11）
            selected_global_features = np.concatenate([global_features[:11], [global_features[12]]])
            global_features_list.append(selected_global_features)

            # 收集局部特征用于标准化
            local_features = _compute_local_features(meta["window"])
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

    # 计算训练集中部分丢包的均值（已废弃，但仍保留以保持接口兼容性）
    mean_loss_cat2_up = 0.5
    mean_loss_cat2_dn = 0.5

    extra_assets = {
        "init_local_cond": init_local_cond,
        "cond_mean": cond_mean,
        "cond_std": cond_std,
        "mean_loss_cat2_up": mean_loss_cat2_up,
        "mean_loss_cat2_dn": mean_loss_cat2_dn,
    }

    # 处理所有数据集
    final_datasets = {}
    for dataset_name, dataset in datasets.items():
        # 按 trace_id 分组并排序
        traces = {}
        for meta in dataset:
            trace_id = meta["trace_id"]
            if trace_id not in traces:
                traces[trace_id] = []
            traces[trace_id].append(meta)

        # 对每个trace内的窗口按 start_time 排序
        for trace_id, metas in traces.items():
            traces[trace_id] = sorted(metas, key=lambda x: x["start_time"])

        # 处理每个窗口
        normed_metas = []
        for trace_id, metas in traces.items():
            for i, meta in enumerate(metas):
                window_df = meta["window"].copy()  # 创建副本以避免修改原始数据

                # 简化丢包处理，移除部分丢包类别（cat=2）
                # 上行丢包只保留0和1两个类别
                window_df.loc[window_df["loss_up"] > 0, "loss_up"] = 1.0

                # 下行丢包只保留0和1两个类别
                window_df.loc[window_df["loss_dn"] > 0, "loss_dn"] = 1.0

                # 计算全局特征
                global_features = _compute_window_features(window_df)

                # 设置网络状态ID（维度11）
                network_state_id = network_state_map.get(trace_id, 0)
                global_features[11] = float(network_state_id)

                # 计算局部特征
                if i == 0:  # 该 trace 在此数据集中的第一个窗口，视为逻辑首窗
                    local_features = init_local_cond
                else:
                    # 使用前一个窗口的最后5行
                    prev_window_df = metas[i-1]["window"]
                    local_features = _compute_local_features(prev_window_df)

                # 合并特征形成23维条件向量
                cond_vector = np.concatenate([global_features, local_features])

                # Level 2 Z-score 标准化（仅对22个float维度）
                # 分离出需要标准化的维度（除了维度11网络状态ID）
                # 全局特征取前11维+保留位12，局部特征是10维，总共22维
                features_to_normalize = np.concatenate([cond_vector[:11], [cond_vector[12]], cond_vector[13:]])
                normalized_features = (features_to_normalize - cond_mean) / cond_std
                # 重新组合，保持网络状态ID不变
                normalized_cond_vector = np.concatenate([
                    normalized_features[:11],
                    [cond_vector[11]],  # 网络状态ID
                    [normalized_features[11]],  # 标准化后的保留位
                    normalized_features[12:],
                ])

                # keep = not is_first
                keep = not meta["is_first"]

                # 构造新的元数据
                normed_meta: WindowMetaNormed = {
                    "window": window_df,
                    "is_first": meta["is_first"],
                    "start_time": meta["start_time"],
                    "trace_id": meta["trace_id"],
                    "cond": normalized_cond_vector,
                    "keep": keep,
                }

                normed_metas.append(normed_meta)

        final_datasets[dataset_name] = normed_metas
    print("已移除部分丢包类别（cat=2），只保留无丢包（cat=0）和全丢包（cat=1）两类")
    return final_datasets, extra_assets


def stage10_save_artifacts(datasets, assets, out_dir: Path, dtype=np.float32):
    """保存处理后的数据和资源
    
    将处理后的数据集和相关资源保存到指定目录中。
    数据集保存为JSONL格式，资源保存为NumPy和Pickle格式。
    
    Example:
        >>> stage10_save_artifacts(final_datasets, assets, Path("output"))
        >>> import os
        >>> os.listdir("output/assets")
        ['qt_up.pkl', 'qt_down.pkl', 'init_local_cond.npy', ...]
    
    Args:
        datasets: 数据集字典
        assets: 资源字典
        out_dir: 输出目录
        dtype: 数据类型

    """
    # 创建必要的目录
    assets_dir = out_dir / "assets"
    meta_dir = out_dir / "meta"
    datasets_dir = out_dir / "datasets"

    assets_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # 保存资源文件
    import pickle
    with open(assets_dir / "qt_up.pkl", "wb") as f:
        pickle.dump(assets["qt_up"], f)

    with open(assets_dir / "qt_down.pkl", "wb") as f:
        pickle.dump(assets["qt_down"], f)

    # 保存条件向量标准化参数
    np.save(meta_dir / "cond_mean.npy", assets["cond_mean"].astype(dtype))
    np.save(meta_dir / "cond_std.npy", assets["cond_std"].astype(dtype))

    # 保存初始局部条件向量
    np.save(assets_dir / "init_local_cond.npy", assets["init_local_cond"].astype(dtype))

    # 保存部分丢包均值
    np.save(assets_dir / "mean_loss_cat2_up.npy", np.array(assets["mean_loss_cat2_up"]).astype(dtype))
    np.save(assets_dir / "mean_loss_cat2_dn.npy", np.array(assets["mean_loss_cat2_dn"]).astype(dtype))

    # 保存元信息
    with open(meta_dir / "pipeline_version.txt", "w") as f:
        f.write("v1.3")

    # 保存数据集（仅保存 keep=True 的样本）
    for dataset_name, dataset in datasets.items():
        filepath = datasets_dir / f"{dataset_name}.jsonl"
        with open(filepath, "w") as f:
            for meta in dataset:
                if meta["keep"]:
                    # 构造要保存的数据
                    record = {
                        "trace_id": meta["trace_id"],
                        "start_time": float(meta["start_time"]),
                        "window": meta["window"].to_dict("records"),
                        "cond": meta["cond"].tolist(),
                    }
                    import json
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")


def stage11_generate_report(stats: dict, config: dict, template_path: Path, out_path: Path):
    """生成数据报告
    
    Args:
        stats: 统计信息
        config: 配置信息
        template_path: 模板路径
        out_path: 输出路径

    """
    try:
        from jinja2 import Template
        with open(template_path) as f:
            template_str = f.read()
        template = Template(template_str)
        report_content = template.render(stats=stats, config=config)
    except Exception:
        # 如果模板加载失败，使用简单格式
        report_content = f"""# 数据处理报告

## 概述

- 处理时间: {stats.get('timestamp', 'N/A')}
- 总窗口数: {stats.get('total_windows', 0)}
- 训练集窗口数: {stats.get('train_count', 0)}
- 验证集窗口数: {stats.get('val_count', 0)}
- 测试集窗口数: {stats.get('test_count', 0)}

## 文件统计

- 处理文件总数: {stats.get('processed_files', 0)}
- 失败文件数: {stats.get('failed_files', 0)}
- 空文件数: {stats.get('empty_files', 0)}

## 配置信息

- 管道版本: {config.get('pipeline_version', 'N/A')}
- 窗口大小: {config.get('window_size', 'N/A')}
- 步长: {config.get('step_size', 'N/A')}
- 最大延迟(ms): {config.get('max_delay_ms', 'N/A')}
- 最大间隙(秒): {config.get('max_gap_sec', 'N/A')}

## 数据划分

- 训练集比例: {config.get('split', {}).get('train', 'N/A')}
- 验证集比例: {config.get('split', {}).get('val', 'N/A')}
- 测试集比例: {config.get('split', {}).get('test', 'N/A')}
"""

    with open(out_path, "w") as f:
        f.write(report_content)

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict, Optional

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

        # 计算 delay 和 loss (delay保持毫秒单位)
        # 检查时延是否为负，如果为负则抛出异常
        if delay1 < 0 or delay2 < 0:
            raise ValueError(f"原始数据中存在负时延：delay1={delay1}, delay2={delay2}")
        
        delay_up_val = delay1  # 直接使用毫秒单位，不转换为秒
        delay_down_val = delay2  # 直接使用毫秒单位，不转换为秒

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
    # 注意：现在delay值已经是毫秒单位，直接与max_delay_ms比较
    truncate_idx = len(df)
    for i in range(len(df)):
        if df.iloc[i]["delay_up"] >= max_delay_ms or df.iloc[i]["delay_down"] >= max_delay_ms:
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

            # 保存原始延迟列，只对新列进行归一化
            window_df["del_up"] = qt_up.transform(window_df["delay_up"].values.reshape(-1, 1)).flatten()
            window_df["del_dn"] = qt_down.transform(window_df["delay_down"].values.reshape(-1, 1)).flatten()
            # 重命名loss列
            window_df = window_df.rename(columns={
                "loss_up": "loss_up",
                "loss_down": "loss_dn",
            })
            
            # 校验：原始时延应该都是非负的，归一化后在[-5,5]之间
            if dataset_name == "train" and len(renamed_metas) < 5:  # 只在训练集前几个窗口打印
                print(f"\n=== 数据集 {dataset_name} 延迟分布校验 ===")
                print(f"原始上行时延统计：")
                print(f"  均值: {np.mean(window_df['delay_up']):.2f}")
                print(f"  中位数: {np.median(window_df['delay_up']):.2f}")
                print(f"  最小值: {np.min(window_df['delay_up']):.2f}")
                print(f"  最大值: {np.max(window_df['delay_up']):.2f}")
                print(f"  >10ms比例: {(np.mean(window_df['delay_up'] > 10) * 100):.1f}%")
                print(f"  非负比例: {(np.mean(window_df['delay_up'] >= 0) * 100):.1f}%")
                print(f"归一化后上行时延统计：")
                print(f"  均值: {np.mean(window_df['del_up']):.2f}")
                print(f"  中位数: {np.median(window_df['del_up']):.2f}")
                print(f"  最小值: {np.min(window_df['del_up']):.2f}")
                print(f"  最大值: {np.max(window_df['del_up']):.2f}")
                print(f"  在[-5,5]区间比例: {(np.mean((window_df['del_up'] >= -5) & (window_df['del_up'] <= 5)) * 100):.1f}%")

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
    features[2] = np.percentile(window_df["del_up"], 1)  # p1_del_up
    features[3] = np.percentile(window_df["del_up"], 99)  # p99_del_up
    features[4] = np.mean(window_df["del_dn"])   # mean_del_dn
    features[5] = np.std(window_df["del_dn"])    # std_del_dn
    features[6] = np.percentile(window_df["del_dn"], 1)  # p1_del_dn
    features[7] = np.percentile(window_df["del_dn"], 99)  # p99_del_dn
    
    # 新增：range特征
    features[8] = features[3] - features[2]  # range_up = p99_up - p1_up
    features[9] = features[7] - features[6]  # range_dn = p99_dn - p1_dn
    # 简化丢包分类，只保留有意义的特征
    # 由于loss_up和loss_dn只有0和1两个值，frac_cat1等于均值
    features[10] = np.mean(window_df["loss_up"])    # frac_cat1_up (等于均值)
    features[11] = np.mean(window_df["loss_dn"])    # frac_cat1_dn (等于均值) - 后续会被网络状态ID覆盖
    features[12] = 0.0  # reserved2

    return features


def _compute_6d_features(window_df):
    """计算窗口的6维特征（用于聚类）
    
    Args:
        window_df: 窗口DataFrame
        
    Returns:
        6维特征数组 [p99_up, std_up, p99_dn, std_dn, loss_up, loss_dn]

    """
    p99_up = np.percentile(window_df["del_up"], 99)
    std_up = np.std(window_df["del_up"])
    p99_dn = np.percentile(window_df["del_dn"], 99)
    std_dn = np.std(window_df["del_dn"])
    loss_up = np.mean(window_df["loss_up"])
    loss_dn = np.mean(window_df["loss_dn"])
    
    return np.array([p99_up, std_up, p99_dn, std_dn, loss_up, loss_dn])


def _max_consecutive(arr):
    """计算数组中连续True的最大长度
    
    Args:
        arr: 布尔数组
        
    Returns:
        int: 最大连续True的长度
    """
    if not arr.any():
        return 0
    
    # 计算连续True的长度
    consecutive_counts = []
    count = 0
    for val in arr:
        if val:
            count += 1
        else:
            if count > 0:
                consecutive_counts.append(count)
                count = 0
    if count > 0:
        consecutive_counts.append(count)
    
    return max(consecutive_counts) if consecutive_counts else 0


def _compute_features(window_df):
    """计算窗口的特征（用于聚类）
    
    Args:
        window_df: 窗口DataFrame
        
    Returns:
        20维特征数组 [raw_mean_delay_up, p95_up, p1_up, std_up, trend_slope_up, 
                      autocorr_lag5_up, loss_up, max_consec_loss_up, 
                      max_burst_up, n_switches_up,
                      raw_mean_delay_down, p95_down, p1_down, std_down, trend_slope_down,
                      autocorr_lag5_down, loss_down, max_consec_loss_down,
                      max_burst_down, n_switches_down]

    """
    # 使用上下行数据
    # 使用原始的delay_up和delay_down列，而不是分位数归一化后的列
    del_up = window_df["delay_up"].values
    loss_up = window_df["loss_up"].values
    del_down = window_df["delay_down"].values
    loss_down = window_df["loss_dn"].values
    
    # 检查时延是否为负，如果为负则抛出异常
    if np.any(del_up < 0):
        raise ValueError(f"上行延迟数据中存在负值：{del_up[del_up < 0]}")
    if np.any(del_down < 0):
        raise ValueError(f"下行延迟数据中存在负值：{del_down[del_down < 0]}")
    
    # 检查原始延迟是否真的 <= 2000
    if np.max(del_up) > 2000:
        print(f"警告：发现上行延迟 >2000ms！max={np.max(del_up)}")
    if np.max(del_down) > 2000:
        print(f"警告：发现下行延迟 >2000ms！max={np.max(del_down)}")
    
    # 强制将延迟限制在 [0, 2000] ms
    del_up = np.clip(del_up, 0, 2000)
    del_down = np.clip(del_down, 0, 2000)
    
    # 获取时间戳序列
    timestamps = window_df["timestamp"].values
    
    # 验证时间戳是否单调递增
    if not np.all(np.diff(timestamps) >= 0):
        print("警告：时间戳非单调！")
        timestamps = np.sort(timestamps)  # 或跳过该窗口
    
    # 辅助函数：计算延迟趋势斜率
    from scipy.stats import linregress
    def compute_trend_slope(delays, timestamps):
        """计算延迟趋势斜率
        
        Args:
            delays: 延迟序列（ms）
            timestamps: 时间戳序列（秒）
            
        Returns:
            趋势斜率（ms/s）
        """
        if len(delays) < 2:
            return 0.0
        # 使用真实时间差作为x轴，而不是采样点索引
        x = timestamps
        # 归一化x轴到[0, window_duration]，单位：秒
        x_normalized = x - x[0]
        slope, _, _, _, _ = linregress(x_normalized, delays)
        return slope
    
    # 辅助函数：计算延迟序列的自相关系数
    def compute_autocorr(delays, lag=5):
        """计算延迟序列的自相关系数"""
        if len(delays) <= lag:
            return 0.0
        return np.corrcoef(delays[:-lag], delays[lag:])[0, 1] if np.std(delays) > 0 else 0.0
    
    # 辅助函数：计算最大连续高延迟段
    def compute_max_burst(delays):
        """计算最大连续高延迟段（秒）"""
        if len(delays) == 0:
            return 0.0
        # 高延迟阈值：90%分位数
        high_delay_threshold = np.percentile(delays, 90)
        high_delay_mask = delays > high_delay_threshold
        return _max_consecutive(high_delay_mask) / 10.0  # 每10个点=1秒
    
    # 辅助函数：计算状态切换次数
    def compute_n_switches(delays, window_size=20, threshold_pct=95):
        """计算状态切换次数"""
        if len(delays) < window_size * 2:
            return 0
        # 计算滑动窗口方差
        import pandas as pd
        window_std = pd.Series(delays).rolling(window_size).std().fillna(0).values
        # 计算方差差异
        diff_std = np.abs(np.diff(window_std))
        # 设置阈值
        threshold = np.percentile(diff_std, threshold_pct)
        # 统计突变次数
        return (diff_std > threshold).sum()
    
    # === 上行特征计算 ===
    # 1. raw_mean_delay_up: 原始上行延迟序列的均值
    raw_mean_delay_up = np.mean(del_up)
    
    # 2. p95_up: 上行延迟的95%分位数
    p95_up = np.percentile(del_up, 95)
    
    # 3. p1_up: 上行延迟的1%分位数
    p1_up = np.percentile(del_up, 1)
    
    # 4. std_up: 上行延迟的标准差
    std_up = np.std(del_up)
    
    # 5. trend_slope_up: 上行延迟趋势斜率（ms/s）
    trend_slope_up = compute_trend_slope(del_up, timestamps)
    
    # 6. autocorr_lag5_up: 上行延迟的自相关系数（lag=5）
    autocorr_lag5_up = compute_autocorr(del_up)
    
    # 7. loss_up: 上行平均丢包率
    loss_up_mean = np.mean(loss_up)
    
    # 8. max_consec_loss_up: 上行最长连续丢包窗口数 → 转秒
    max_consec_loss_up = _max_consecutive(loss_up > 0) / 10.0  # 每10个点=1秒
    
    # 9. max_burst_up: 上行最长连续高延迟段（秒）
    max_burst_up = compute_max_burst(del_up)
    
    # 10. n_switches_up: 上行滑动窗口方差突变次数
    n_switches_up = compute_n_switches(del_up)
    
    # === 下行特征计算 ===
    # 11. raw_mean_delay_down: 原始下行延迟序列的均值
    raw_mean_delay_down = np.mean(del_down)
    
    # 12. p95_down: 下行延迟的95%分位数
    p95_down = np.percentile(del_down, 95)
    
    # 13. p1_down: 下行延迟的1%分位数
    p1_down = np.percentile(del_down, 1)
    
    # 14. std_down: 下行延迟的标准差
    std_down = np.std(del_down)
    
    # 15. trend_slope_down: 下行延迟趋势斜率（ms/s）
    trend_slope_down = compute_trend_slope(del_down, timestamps)
    
    # 16. autocorr_lag5_down: 下行延迟的自相关系数（lag=5）
    autocorr_lag5_down = compute_autocorr(del_down)
    
    # 17. loss_down: 下行平均丢包率
    loss_down_mean = np.mean(loss_down)
    
    # 18. max_consec_loss_down: 下行最长连续丢包窗口数 → 转秒
    max_consec_loss_down = _max_consecutive(loss_down > 0) / 10.0  # 每10个点=1秒
    
    # 19. max_burst_down: 下行最长连续高延迟段（秒）
    max_burst_down = compute_max_burst(del_down)
    
    # 20. n_switches_down: 下行滑动窗口方差突变次数
    n_switches_down = compute_n_switches(del_down)
    
    # 构建20维特征向量
    features = np.array([
        # 上行特征
        raw_mean_delay_up,
        p95_up,
        p1_up,
        std_up,
        trend_slope_up,
        autocorr_lag5_up,
        loss_up_mean,
        max_consec_loss_up,
        max_burst_up,
        n_switches_up,
        # 下行特征
        raw_mean_delay_down,
        p95_down,
        p1_down,
        std_down,
        trend_slope_down,
        autocorr_lag5_down,
        loss_down_mean,
        max_consec_loss_down,
        max_burst_down,
        n_switches_down
    ])
    
    # === 安全 Clip（防御性）===
    # 上行延迟类特征：raw_mean, p95, p1, std
    features[0:4] = np.clip(features[0:4], 0, 2000)
    # 上行趋势斜率：限制在±50 ms/s，符合真实网络
    features[4] = np.clip(features[4], -50, 50)
    # 上行自相关系数：限制在[-1, 1]，理论范围
    features[5] = np.clip(features[5], -1, 1)
    # 上行丢包率：限制在[0, 1]，理论范围
    features[6] = np.clip(features[6], 0, 1)
    
    # 下行延迟类特征：raw_mean, p95, p1, std
    features[10:14] = np.clip(features[10:14], 0, 2000)
    # 下行趋势斜率：限制在±50 ms/s，符合真实网络
    features[14] = np.clip(features[14], -50, 50)
    # 下行自相关系数：限制在[-1, 1]，理论范围
    features[15] = np.clip(features[15], -1, 1)
    # 下行丢包率：限制在[0, 1]，理论范围
    features[16] = np.clip(features[16], 0, 1)
    
    # === 非线性压缩（避免 NaN）===
    # 上行延迟统计量：非负，安全使用log1p
    features[0:4] = np.log1p(features[0:4])
    # 上行其他非负特征：loss, max_consec_loss, max_burst, n_switches
    features[6:10] = np.log1p(features[6:10])
    
    # 下行延迟统计量：非负，安全使用log1p
    features[10:14] = np.log1p(features[10:14])
    # 下行其他非负特征：loss, max_consec_loss, max_burst, n_switches
    features[16:20] = np.log1p(features[16:20])
    
    # 趋势斜率：带符号的log压缩，保留方向
    # 上行趋势斜率
    slope_up = features[4]
    features[4] = np.sign(slope_up) * np.log1p(np.abs(slope_up) + 1e-8)  # 加epsilon防止log(0)
    # 下行趋势斜率
    slope_down = features[14]
    features[14] = np.sign(slope_down) * np.log1p(np.abs(slope_down) + 1e-8)  # 加epsilon防止log(0)
    
    # 确保所有值都是有限的，替换NaN和无穷大
    features = np.where(np.isfinite(features), features, 0.0)
    
    return features


def _select_optimal_k(features, max_k=10):
    """使用肘部法则和轮廓系数自动选择最优的K值
    
    Args:
        features: 特征数组，形状为(n_samples, n_features)
        max_k: 最大尝试的K值
        
    Returns:
        tuple: (optimal_k, k_range, inertias, silhouettes, davies_bouldins)
            optimal_k: 最优的K值
            k_range: K值范围
            inertias: 不同K值下的惯性列表
            silhouettes: 不同K值下的轮廓系数列表
            davies_bouldins: 不同K值下的Davies-Bouldin指数列表

    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    
    k_range = range(2, max_k+1)
    inertias = []
    silhouettes = []
    davies_bouldins = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        
        # 计算惯性（inertia）
        inertias.append(kmeans.inertia_)
        
        # 计算轮廓系数（silhouette score）
        silhouette = silhouette_score(features, cluster_labels)
        silhouettes.append(silhouette)
        
        # 计算Davies-Bouldin指数
        davies_bouldin = davies_bouldin_score(features, cluster_labels)
        davies_bouldins.append(davies_bouldin)
        
        print(f"K={k}: 惯性={kmeans.inertia_:.2f}, 轮廓系数={silhouette:.4f}, Davies-Bouldin指数={davies_bouldin:.4f}")
    
    # 寻找肘部点
    wcss_diff = np.diff(inertias)
    wcss_diff_ratio = wcss_diff[1:] / wcss_diff[:-1]
    elbow_idx = np.argmin(wcss_diff_ratio) + 2  # +2 因为从k=2开始
    
    # 同时考虑轮廓系数最大值
    silhouette_idx = np.argmax(silhouettes) + 2
    
    # 考虑Davies-Bouldin指数最小值
    davies_bouldin_idx = np.argmin(davies_bouldins) + 2
    
    # 综合考虑，选择更合理的K值
    # 优先选择轮廓系数最大值对应的K值
    optimal_k = silhouette_idx
    
    print(f"\n自动选择K值结果：")
    print(f"- 肘部法则推荐：{elbow_idx}")
    print(f"- 轮廓系数推荐：{silhouette_idx}")
    print(f"- Davies-Bouldin推荐：{davies_bouldin_idx}")
    print(f"- 最终选择：{optimal_k}")
    
    return optimal_k, k_range, inertias, silhouettes, davies_bouldins


def _compute_local_features(window_df, last_n=5):
    """计算窗口的局部特征（最后n行）
    
    Args:
        window_df: 窗口DataFrame
        last_n: 使用最后几行计算特征
        
    Returns:
        10维局部特征数组

    """
    # 忽略局部特征，直接返回零向量
    return np.zeros(10)


def stage9_recompute_condition_vectors(    datasets: Dict[str, List[WindowMetaRenamed]],
    assets: Dict,
    network_state_map: Dict[str, int],
    fixed_k: Optional[int] = None,
) -> Tuple[Dict[str, List[WindowMetaNormed]], Dict]:
    """重新计算条件向量
    
    基于归一化数据计算23维条件向量，并进行Z-score标准化。
    条件向量包括13维全局特征和10维局部特征。
    使用10维特征（p99_up, std_up, p99_dn, std_dn, loss_up, loss_dn, p1_up, p1_dn, range_up, range_dn）进行自动聚类，
    生成新的network_state_id。
    
    Example:
        >>> final_datasets, extra_assets = stage9_recompute_condition_vectors(
        ...     renamed_datasets, assets, network_state_map)
        >>> train_data = final_datasets['train']
        >>> print(train_data[0]['cond'].shape)
        (23,)
    
    Args:
        datasets: 重命名后的数据集
        assets: 资源字典
        network_state_map: 网络状态映射（将被自动聚类结果替换）
        
    Returns:
        (final_datasets, extra_assets) 二元组

    """
    # 首先收集所有窗口的10维特征用于自动聚类
    all_features = []
    all_metas = []
    
    # 收集所有数据集的窗口
    for dataset_name, dataset in datasets.items():
        for meta in dataset:
            all_metas.append(meta)
            # 计算10维特征
            features = _compute_features(meta["window"])
            all_features.append(features)
    
    all_features = np.array(all_features)
    
    print(f"\n=== 开始聚类分析 ===")
    print(f"收集到的特征数量: {all_features.shape[0]}")
    print(f"特征维度: {all_features.shape[1]}")
    
    # 标准化特征用于聚类和可视化（使用RobustScaler，抗异常值）
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    all_features_scaled = scaler.fit_transform(all_features)
    
    # 校验：确保特征是基于原始延迟数据，显示整体统计信息
    print(f"\n=== 特征校验 ===")
    print(f"特征统计（归一化前）：")
    print(f"  整体均值: {np.mean(all_features):.2f}")
    print(f"  整体中位数: {np.median(all_features):.2f}")
    print(f"  整体最小值: {np.min(all_features):.2f}")
    print(f"  整体最大值: {np.max(all_features):.2f}")
    print(f"  整体标准差: {np.std(all_features):.2f}")
    
    print(f"\n特征统计（RobustScaler归一化后）：")
    print(f"  整体均值: {np.mean(all_features_scaled):.2f}")
    print(f"  整体中位数: {np.median(all_features_scaled):.2f}")
    print(f"  整体最小值: {np.min(all_features_scaled):.2f}")
    print(f"  整体最大值: {np.max(all_features_scaled):.2f}")
    print(f"  整体标准差: {np.std(all_features_scaled):.2f}")
    
    # 检查特征值范围是否合理
    print(f"\n归一化后特征值范围检查：")
    print(f"  99% 分位数上限: {np.percentile(all_features_scaled, 99):.2f}")
    print(f"  1% 分位数下限: {np.percentile(all_features_scaled, 1):.2f}")
    print(f"  超过 [-10, 10] 范围的特征值比例: {(np.mean((all_features_scaled < -10) | (all_features_scaled > 10)) * 100):.1f}%")
    
    # 使用HDBSCAN进行聚类
    import hdbscan
    from sklearn.cluster import KMeans
    print("\n=== 开始HDBSCAN聚类分析 ===")
    
    # HDBSCAN参数设置
    min_cluster_size = 5  # 簇的最小样本数
    min_samples = 3       # 每个点成为核心点所需的最小邻域样本数
    
    # 执行HDBSCAN聚类
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    cluster_labels = hdb.fit_predict(all_features_scaled)
    
    # 统计聚类结果
    unique_labels = np.unique(cluster_labels)
    # 过滤掉噪声点（标签为-1）
    valid_labels = unique_labels[unique_labels != -1]
    optimal_k = len(valid_labels)
    
    # 统计各聚类的数量分布（包括噪声点）
    if -1 in cluster_labels:
        noise_count = np.sum(cluster_labels == -1)
        print(f"HDBSCAN初始聚类: 有效簇数={optimal_k}, 噪声点={noise_count}")
    else:
        print(f"HDBSCAN初始聚类: 有效簇数={optimal_k}, 无噪声点")
    
    # 如果HDBSCAN未能找到有效聚类，使用K-means作为fallback
    if optimal_k == 0:
        print("HDBSCAN未能找到有效聚类，切换到K-means聚类...")
        
        # 使用K-means，默认K=6
        fallback_k = 6
        kmeans = KMeans(n_clusters=fallback_k, random_state=42)
        cluster_labels = kmeans.fit_predict(all_features_scaled)
        optimal_k = fallback_k
        
        cluster_counts = np.bincount(cluster_labels)
        print(f"K-means聚类分布: {cluster_counts}")
        print(f"最终聚类数量: {optimal_k}")
        
        # 为了可视化兼容，生成假的inertias等数据
        k_range = range(2, optimal_k+2)
        inertias = [float('inf')] * len(k_range)  # 占位用
        silhouettes = [0.0] * len(k_range)  # 占位用
        davies_bouldins = [float('inf')] * len(k_range)  # 占位用
    else:
        print(f"最终聚类数量: {optimal_k}")
        
        # 为了可视化兼容，生成假的inertias等数据
        k_range = range(2, optimal_k+2 if optimal_k > 0 else 3)
        inertias = [float('inf')] * len(k_range)  # 占位用
        silhouettes = [0.0] * len(k_range)  # 占位用
        davies_bouldins = [float('inf')] * len(k_range)  # 占位用
        
        # 调整标签，确保没有负数（将噪声点分配到现有簇或新簇）
        if -1 in cluster_labels:
            print(f"将 {np.sum(cluster_labels == -1)} 个噪声点分配到最近的簇...")
            # 计算每个点到其最近簇中心的距离
            from sklearn.metrics import pairwise_distances
            
            # 为每个有效簇计算中心
            cluster_centers = []
            for label in valid_labels:
                cluster_points = all_features_scaled[cluster_labels == label]
                cluster_center = np.mean(cluster_points, axis=0)
                cluster_centers.append(cluster_center)
            cluster_centers = np.array(cluster_centers)
            
            # 对每个噪声点，找到最近的簇中心
            noise_mask = cluster_labels == -1
            noise_points = all_features_scaled[noise_mask]
            
            distances = pairwise_distances(noise_points, cluster_centers)
            closest_clusters = np.argmin(distances, axis=1)
            
            # 更新噪声点的标签为最近簇的标签
            cluster_labels[noise_mask] = valid_labels[closest_clusters]
            
            # 重新统计聚类分布
            cluster_counts = np.bincount(cluster_labels)
            print(f"调整后聚类分布: {cluster_counts}")
    
    # 保存每个窗口的聚类结果到元数据中
    # 注意：这里不再使用network_state_map按trace_id分配聚类ID，而是为每个窗口分配独立的聚类结果
    # 构建窗口到聚类结果的映射
    window_cluster_map = {}
    for i, meta in enumerate(all_metas):
        window_key = f"{meta['trace_id']}_{meta['start_time']}"
        window_cluster_map[window_key] = int(cluster_labels[i])
    
    # 生成可视化图表
    from src.visualization.clustering_visualization import ClusteringVisualizer
    visualizer = ClusteringVisualizer(Path("output/clustering_visualization"))
    
    # 可视化使用标准化后的特征（与聚类使用的特征一致）
    features_scaled = all_features_scaled
    
    # 生成聚类可视化图表
    visualizer.generate_clustering_visualizations(
        features_scaled, cluster_labels, k_range, inertias, silhouettes, davies_bouldins, optimal_k
    )
    
    # 准备窗口数据用于整合可视化
    window_data = []
    for meta in all_metas:
        # 转换为与典型案例可视化兼容的格式
        window_data.append({
            "window": meta["window"].to_dict("records"),
            "trace_id": meta["trace_id"],
            "start_time": meta["start_time"]
        })
    
    # 生成整合可视化，包含降维图和典型样本图
    visualizer.generate_integrated_visualization(
        features_scaled, cluster_labels, optimal_k, window_data, 
        assets["qt_up"], assets["qt_down"]
    )
    
    # 打印自动聚类结果
    print(f"\n=== 自动聚类结果 ===")
    print(f"特征数量: {all_features.shape[0]}")
    print(f"特征维度: {all_features.shape[1]}")
    print(f"自动选择的K值: {optimal_k}")
    print(f"聚类标签分布: {np.bincount(cluster_labels)}")
    print(f"使用自动聚类生成的窗口聚类映射，包含{len(window_cluster_map)}个窗口")
    
    # 验证聚类结果
    print(f"\n=== 聚类结果验证 ===")
    from sklearn.metrics import silhouette_score
    silhouette = silhouette_score(all_features_scaled, cluster_labels)
    print(f"轮廓系数: {silhouette:.4f}")
    
    # 保存聚类结果，用于典型案例展示
    cluster_dir = Path("output/10d_clustering_analysis")
    cluster_dir.mkdir(parents=True, exist_ok=True)
    np.save(cluster_dir / "cluster_labels.npy", cluster_labels)
    
    # 生成split_labels，与cluster_labels一一对应
    split_labels = []
    for dataset_name, dataset in datasets.items():
        split_labels.extend([dataset_name] * len(dataset))
    split_labels = np.array(split_labels)
    np.save(cluster_dir / "split_labels.npy", split_labels)
    
    # 注意：不再替换network_state_map，而是使用window_cluster_map来获取每个窗口的聚类结果
    
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
            window_key = f"{meta['trace_id']}_{meta['start_time']}"
            network_state_id = window_cluster_map.get(window_key, 0)
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

    # 初始化extra_assets字典
    extra_assets = {
        "init_local_cond": init_local_cond,
        "cond_mean": cond_mean,
        "cond_std": cond_std,
        "mean_loss_cat2_up": mean_loss_cat2_up,
        "mean_loss_cat2_dn": mean_loss_cat2_dn,
        # 预先添加空的统计信息，后续会被更新
        "state_id_stats": {},
        "cluster_features": "10D",
        "optimal_k": 0
    }

    # 处理所有数据集
    final_datasets = {}
    state_id_stats = {}
    
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
        dataset_state_ids = []
        
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
                window_key = f"{meta['trace_id']}_{meta['start_time']}"
                network_state_id = window_cluster_map.get(window_key, 0)
                global_features[11] = float(network_state_id)
                
                # 统计state_id分布（只统计keep=True的样本）
                if not meta["is_first"]:  # keep = not is_first
                    dataset_state_ids.append(int(network_state_id))

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
        
        # 统计当前数据集的state_id分布
        if dataset_state_ids:
            state_counts = np.bincount(dataset_state_ids)
            state_id_stats[dataset_name] = {}
            total_samples = len(dataset_state_ids)
            
            for state_id in range(len(state_counts)):
                count = state_counts[state_id]
                if count > 0:
                    percentage = (count / total_samples) * 100
                    state_id_stats[dataset_name][state_id] = {
                        "count": count,
                        "percentage": percentage
                    }
        else:
            state_id_stats[dataset_name] = {}
    
    # 输出统计信息
    print("\n=== 网络状态ID分布统计 ===")
    print(f"使用10D特征聚类，K={optimal_k}")
    print("\n各数据集状态ID分布:")
    for dataset_name, stats in state_id_stats.items():
        print(f"\n{dataset_name.upper()}集:")
        total_samples = sum(stat["count"] for stat in stats.values())
        print(f"  总样本数: {total_samples}")
        print(f"  状态ID分布:")
        for state_id in sorted(stats.keys()):
            stat = stats[state_id]
            print(f"    状态ID {state_id}: {stat['count']}个样本 ({stat['percentage']:.2f}%)")
    
    print("\n已移除部分丢包类别（cat=2），只保留无丢包（cat=0）和全丢包（cat=1）两类")
    
    # 将统计信息添加到extra_assets中
    extra_assets["state_id_stats"] = state_id_stats
    extra_assets["cluster_features"] = "10D"
    extra_assets["optimal_k"] = optimal_k
    
    # 打印extra_assets字典的内容
    print("\n=== 函数返回前extra_assets内容 ===")
    print(f"extra_assets键: {list(extra_assets.keys())}")
    
    return final_datasets, extra_assets


def stage10_save_artifacts(datasets, assets, out_dir: Path, dtype=np.float32):
    """保存处理后的数据和资源
    
    将处理后的数据集和相关资源保存到指定目录中。
    数据集保存为JSONL格式，资源保存为NumPy和Pickle格式。
    同时生成典型样本可视化图表。
    
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

    # 保存聚类结果和统计信息
    import json
    
    # 准备聚类信息，确保所有numpy类型都转换为Python原生类型
    cluster_features = assets.get("cluster_features", "6D")
    optimal_k = int(assets.get("optimal_k", 6)) if hasattr(assets.get("optimal_k", 6), "item") else assets.get("optimal_k", 6)
    
    # 转换state_id_stats中的numpy类型为Python原生类型
    state_id_stats = assets.get("state_id_stats", {})
    converted_stats = {}
    for dataset_name, stats in state_id_stats.items():
        converted_stats[dataset_name] = {}
        for state_id, stat in stats.items():
            converted_stats[dataset_name][int(state_id)] = {
                "count": int(stat["count"]),
                "percentage": float(stat["percentage"])
            }
    
    clustering_info = {
        "cluster_features": cluster_features,
        "optimal_k": optimal_k,
        "state_id_stats": converted_stats
    }
    
    with open(meta_dir / "clustering_info.json", "w", encoding="utf-8") as f:
        json.dump(clustering_info, f, ensure_ascii=False, indent=2)

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
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    # 生成典型样本可视化
    print("\n=== 生成典型样本可视化 ===")
    import matplotlib.pyplot as plt
    from pathlib import Path
    import pandas as pd
    
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 加载聚类结果
    cluster_dir = Path("output/10d_clustering_analysis")
    cluster_labels = np.load(cluster_dir / "cluster_labels.npy")
    
    # 收集所有窗口数据
    window_data = []
    for dataset_name, dataset in datasets.items():
        for meta in dataset:
            if meta["keep"]:
                window_data.append({
                    "window": meta["window"].to_dict("records"),
                    "trace_id": meta["trace_id"],
                    "start_time": meta["start_time"]
                })
    
    # 选择典型案例
    def select_typical_cases(window_data, cluster_labels, num_cases=3):
        """从每个聚类中选择一个典型案例
        
        优化选择策略：优先选择包含下行丢包的窗口，以展示完整的网络状态
        """
        typical_cases = {}
        unique_clusters = np.unique(cluster_labels)
        
        # 确保window_data和cluster_labels长度匹配
        min_length = min(len(window_data), len(cluster_labels))
        if min_length == 0:
            return typical_cases
        
        # 截断cluster_labels到window_data的长度
        cluster_labels = cluster_labels[:min_length]
        unique_clusters = np.unique(cluster_labels)
        
        for cluster_id in unique_clusters[:num_cases]:
            # 找出当前聚类的所有窗口
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
            
            # 优先选择包含下行丢包的窗口
            selected_idx = None
            for idx in cluster_indices:
                try:
                    # 检查窗口数据中是否包含下行丢包
                    window_df = pd.DataFrame(window_data[idx]["window"])
                    if len(window_df) > 0:
                        loss_dn = window_df["loss_dn"].values
                        # 检查是否有下行丢包
                        if np.any(loss_dn > 0):
                            selected_idx = idx
                            break
                except IndexError:
                    continue
            
            # 如果没有找到包含下行丢包的窗口，选择第一个有效窗口
            if selected_idx is None:
                # 找到第一个有效的索引
                for idx in cluster_indices:
                    if idx < len(window_data):
                        selected_idx = idx
                        break
                else:
                    continue
            
            typical_cases[cluster_id] = {
                "window": window_data[selected_idx],
                "cluster_id": cluster_id
            }
        
        return typical_cases
    
    # 反归一化函数
    def inverse_transform_delay(delays, qt_model):
        """对时延数据进行反归一化"""
        delays_2d = delays.reshape(-1, 1)
        inverse_delays = qt_model.inverse_transform(delays_2d)
        return inverse_delays.flatten()
    
    # 绘制典型案例
    def plot_typical_case(ax, case_data, qt_up, qt_down, title=""):
        """绘制典型案例的时延和卡顿情况"""
        # 提取窗口数据
        window_df = pd.DataFrame(case_data["window"]["window"])
        # 只取前100个点
        window_df = window_df.head(100)
        
        # 反归一化时延数据
        del_up = window_df["del_up"].values
        del_dn = window_df["del_dn"].values
        loss_up = window_df["loss_up"].values
        loss_dn = window_df["loss_dn"].values
        
        inverse_del_up = inverse_transform_delay(del_up, qt_up)
        inverse_del_dn = inverse_transform_delay(del_dn, qt_down)
        
        # 绘制上行和下行时延
        ax.plot(inverse_del_up, label="上行时延 (ms)", color="blue")
        ax.plot(inverse_del_dn, label="下行时延 (ms)", color="red")
        
        # 添加卡顿标记（丢包时）
        loss_up_indices = np.where(loss_up > 0)[0]
        loss_dn_indices = np.where(loss_dn > 0)[0]
        
        if len(loss_up_indices) > 0:
            ax.scatter(loss_up_indices, inverse_del_up[loss_up_indices], 
                      color="blue", marker="x", s=50, label="上行丢包")
        if len(loss_dn_indices) > 0:
            ax.scatter(loss_dn_indices, inverse_del_dn[loss_dn_indices], 
                      color="red", marker="x", s=50, label="下行丢包")
        
        ax.set_xlabel("时间点 (100ms间隔)")
        ax.set_ylabel("时延 (ms)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 执行典型案例选择和绘制
    typical_cases = select_typical_cases(window_data, cluster_labels, num_cases=3)
    
    if typical_cases:
        # 绘制典型案例
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        for i, (cluster_id, case_data) in enumerate(typical_cases.items()):
            if i >= len(axes):
                break
            
            title = f"类别 {cluster_id} - 典型案例"
            plot_typical_case(axes[i], case_data, assets["qt_up"], assets["qt_down"], title)
        
        # 保存图表
        output_dir = Path("output/typical_cases")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "typical_cases.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"典型样本可视化已保存到: {output_path}")
        plt.close()
    else:
        print("未找到典型案例，跳过可视化生成")
    
    print("\n=== 典型样本可视化生成完成 ===")


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

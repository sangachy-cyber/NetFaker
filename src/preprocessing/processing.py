from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .features import compute_window_features
from sklearn.preprocessing import RobustScaler



def parse_txt(txt_path: Path, interval_sec: float, file_config: Dict[str, str] = None) -> pd.DataFrame:
    """解析原始 txt 文件为标准化 DataFrame
    
    Args:
        txt_path: txt文件路径
        interval_sec: 间隔秒数
        file_config: 文件对应的配置信息
        
    Returns:
        标准化 DataFrame，包含列: [timestamp, delay_up, delay_down, loss_up, loss_down, 
                                    bandwidth_up, bandwidth_down, total_pkts_up, total_pkts_down,
                                    loss_pkts_up, loss_pkts_down, send_pkts_up, send_pkts_down]
        
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
    timestamps, delay_up, loss_up, delay_down, loss_down, bandwidth_up, bandwidth_down = _parse_data_lines(
        data_lines, start_timestamp, interval_sec
    )

    # 构造 DataFrame，包含原始值
    df = pd.DataFrame({
        "timestamp": timestamps,
        "delay_up": delay_up,  # 原始上行延迟
        "delay_down": delay_down,  # 原始下行延迟
        "loss_up": loss_up,  # 原始上行丢包率
        "loss_down": loss_down,  # 原始下行丢包率
        "bandwidth_up": bandwidth_up,  # 原始上行带宽
        "bandwidth_down": bandwidth_down,  # 原始下行带宽
    })

    # 如果有配置文件，计算丢包数和发包数
    if file_config:
        # 解析配置信息
        try:
            # 提取配置信息
            up_rate_kbps = float(file_config.get("上行速率", "0").replace(" Kbps", ""))
            down_rate_kbps = float(file_config.get("下行速率", "0").replace(" Kbps", ""))
            pkt_size = float(file_config.get("包大小(Byte)", "1500"))
            sample_interval_str = file_config.get("采样间隔", "100")
            sample_interval_ms = float(sample_interval_str.replace(" ms", ""))
            
            # 转换单位
            up_rate_bps = up_rate_kbps * 1000  # Kbps -> bps
            down_rate_bps = down_rate_kbps * 1000  # Kbps -> bps
            sample_interval_s = sample_interval_ms / 1000  # ms -> s
            pkt_size_bits = pkt_size * 8  # Byte -> bits
            
            # 计算每个周期的总包数
            df["total_pkts_up"] = (up_rate_bps * sample_interval_s) / pkt_size_bits
            df["total_pkts_down"] = (down_rate_bps * sample_interval_s) / pkt_size_bits
            
            # 确保总包数为正数
            df["total_pkts_up"] = df["total_pkts_up"].clip(lower=1.0)
            df["total_pkts_down"] = df["total_pkts_down"].clip(lower=1.0)
            
            # 计算丢包数和发包数
            def find_best_packet_count(total_pkts, loss_rate):
                """找到最接近原始丢包率的整数组合（发包数，丢包数）
                
                Args:
                    total_pkts: 估算的总包数（浮点型）
                    loss_rate: 原始丢包率
                    
                Returns:
                    (send_pkts, loss_pkts): 发包数和丢包数的整数组合
                """
                # 基础发包数，使用四舍五入的整数
                base_send = round(total_pkts)
                
                # 考虑附近的几个整数，寻找最优组合
                best_error = float('inf')
                best_send = base_send
                best_loss = round(base_send * loss_rate)
                
                # 检查base_send附近的5个整数（避免极端情况）
                for send_pkts in range(max(1, base_send - 2), base_send + 3):
                    # 计算可能的丢包数
                    loss_pkts = round(send_pkts * loss_rate)
                    # 确保丢包数在合理范围内
                    loss_pkts = max(0, min(loss_pkts, send_pkts))
                    
                    # 计算误差
                    if send_pkts > 0:
                        calc_loss_rate = loss_pkts / send_pkts
                        error = abs(calc_loss_rate - loss_rate)
                        
                        # 找到误差最小的组合
                        if error < best_error:
                            best_error = error
                            best_send = send_pkts
                            best_loss = loss_pkts
                
                return best_send, best_loss
            
            # 对每个行计算最优的发包数和丢包数
            df["send_pkts_up"] = 0
            df["loss_pkts_up"] = 0
            df["send_pkts_down"] = 0
            df["loss_pkts_down"] = 0
            
            for i in range(len(df)):
                # 上行
                total_pkts_up = df.loc[i, "total_pkts_up"]
                loss_up = df.loc[i, "loss_up"]
                df.loc[i, "send_pkts_up"], df.loc[i, "loss_pkts_up"] = find_best_packet_count(total_pkts_up, loss_up)
                
                # 下行
                total_pkts_down = df.loc[i, "total_pkts_down"]
                loss_down = df.loc[i, "loss_down"]
                df.loc[i, "send_pkts_down"], df.loc[i, "loss_pkts_down"] = find_best_packet_count(total_pkts_down, loss_down)
        except Exception as e:
            print(f"警告：解析配置文件 {txt_path} 时出错：{e}")
            # 如果配置解析失败，添加默认值
            df["total_pkts_up"] = 0.0
            df["total_pkts_down"] = 0.0
            df["send_pkts_up"] = 0
            df["send_pkts_down"] = 0
            df["loss_pkts_up"] = 0
            df["loss_pkts_down"] = 0
    else:
        # 如果没有配置文件，添加默认值
        df["total_pkts_up"] = 0.0
        df["total_pkts_down"] = 0.0
        df["send_pkts_up"] = 0
        df["send_pkts_down"] = 0
        df["loss_pkts_up"] = 0
        df["loss_pkts_down"] = 0

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
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float], List[float]]:
    """解析数据行
    
    Args:
        data_lines: 数据行列表
        start_timestamp: 开始时间戳
        interval_sec: 间隔秒数
        
    Returns:
        (timestamps, delay_up, loss_up, delay_down, loss_down, bandwidth_up, bandwidth_down) 七元组
    """
    timestamps = []
    delay_up = []
    loss_up = []
    delay_down = []
    loss_down = []
    bandwidth_up = []
    bandwidth_down = []

    for i, line in enumerate(data_lines):
        if not line.strip():
            continue

        parts = line.strip().split(",")
        if len(parts) < 6:
            continue

        # 解析单行数据
        parsed_data = _parse_single_line(parts, start_timestamp, i, interval_sec)
        if parsed_data is not None:
            ts, du, lu, dd, ld, bu, bd = parsed_data
            timestamps.append(ts)
            delay_up.append(du)
            loss_up.append(lu)
            delay_down.append(dd)
            loss_down.append(ld)
            bandwidth_up.append(bu)
            bandwidth_down.append(bd)

    return timestamps, delay_up, loss_up, delay_down, loss_down, bandwidth_up, bandwidth_down


def _parse_single_line(
    parts: List[str], 
    start_timestamp: float, 
    line_idx: int, 
    interval_sec: float
) -> Tuple[float, float, float, float, float, float, float] or None:
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

    return timestamp, delay_up_val, loss_up_val, delay_down_val, loss_down_val, bandwidth1, bandwidth2


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

    # 从前往后扫描，首次出现 delay_up ≥ max_delay_ms 或 delay_down ≥ max_delay_ms → 丢弃该行及之后所有行
    truncate_idx = len(df)
    for i in range(len(df)):
        if df.iloc[i]["delay_up"] >= max_delay_ms or df.iloc[i]["delay_down"] >= max_delay_ms:
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
    resampled["delay_up"] = resampled["delay_up"].interpolate(method="linear")
    resampled["delay_down"] = resampled["delay_down"].interpolate(method="linear")

    # 对原始丢包率使用前向填充
    resampled["loss_up"] = resampled["loss_up"].ffill().bfill()
    resampled["loss_down"] = resampled["loss_down"].ffill().bfill()
    
    # 处理新添加的列
    if "bandwidth_up" in resampled.columns:
        # 对带宽使用前向填充
        resampled["bandwidth_up"] = resampled["bandwidth_up"].ffill().bfill()
        resampled["bandwidth_down"] = resampled["bandwidth_down"].ffill().bfill()
        
        # 对总包数、发包数和丢包数使用前向填充
        for col in ["total_pkts_up", "total_pkts_down", "send_pkts_up", "send_pkts_down", "loss_pkts_up", "loss_pkts_down"]:
            if col in resampled.columns:
                resampled[col] = resampled[col].ffill().bfill()
    
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
    
    使用训练集数据拟合RobustScaler，然后对所有数据集应用标准化变换。
    基于行为ID和方向（上行/下行）进行鲁棒性归一化，共需要16个归一化器：
    8个有效行为ID × 2个方向（上行/下行）。
    
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
    if not train:
        raise ValueError("训练集为空")

    # 1. 按行为ID和方向分组训练数据
    # 准备存储分组数据的字典
    train_data_grouped = {}
    for behavior_id in range(8):  # 0-7共8个有效行为ID
        train_data_grouped[f"up_{behavior_id}"] = []
        train_data_grouped[f"down_{behavior_id}"] = []
    
    # 收集训练数据，按行为ID和方向分组
    for meta in train:
        window_df = meta["window"]
        behavior_id = meta.get("behavior_id", 0)  # 默认使用0号行为ID
        if behavior_id >= 8:  # 跳过INVALID行为
            continue
        
        # 提取原始延迟特征
        delays_up = window_df["delay_up"].values
        delays_down = window_df["delay_down"].values
        
        # 添加到对应的分组
        train_data_grouped[f"up_{behavior_id}"].append(delays_up)
        train_data_grouped[f"down_{behavior_id}"].append(delays_down)

    # 2. 拟合16个RobustScaler
    rs_dict = {}
    
    for behavior_id in range(8):
        # 拟合上行归一化器
        up_key = f"up_{behavior_id}"
        up_data = np.concatenate(train_data_grouped[up_key]) if train_data_grouped[up_key] else np.array([[0.0]])
        
        rs_up = RobustScaler()
        rs_up.fit(up_data.reshape(-1, 1))
        rs_dict[up_key] = rs_up
        
        # 拟合下行归一化器
        down_key = f"down_{behavior_id}"
        down_data = np.concatenate(train_data_grouped[down_key]) if train_data_grouped[down_key] else np.array([[0.0]])
        
        rs_down = RobustScaler()
        rs_down.fit(down_data.reshape(-1, 1))
        rs_dict[down_key] = rs_down

    # 3. 处理数据集的辅助函数
    def process_dataset(dataset):
        renamed_metas = []
        for meta in dataset:
            window_df = meta["window"].copy()
            behavior_id = meta.get("behavior_id", 0)  # 默认使用0号行为ID
            if behavior_id >= 8:  # 处理INVALID行为，使用默认行为ID
                behavior_id = 0
            
            # 检查窗口内的延迟是否恒定（标准差为0）
            window_std_up = window_df['delay_up'].std()
            window_std_down = window_df['delay_down'].std()
            
            # 如果上行或下行延迟在窗口内是恒定的，则跳过此窗口
            if window_std_up == 0 or window_std_down == 0:
                continue  # 跳过这个无效窗口
            
            # 提取原始延迟列
            raw_delay_up = window_df["delay_up"].values
            raw_delay_down = window_df["delay_down"].values
            
            # 选择对应的归一化器
            rs_up = rs_dict[f"up_{behavior_id}"]
            rs_down = rs_dict[f"down_{behavior_id}"]
            
            # 应用归一化
            delay_up_rs = rs_up.transform(raw_delay_up.reshape(-1, 1)).flatten()
            delay_down_rs = rs_down.transform(raw_delay_down.reshape(-1, 1)).flatten()
            
            # 保存RobustScaler归一化结果，用于条件特征计算
            window_df["delay_up_rs"] = delay_up_rs
            window_df["delay_down_rs"] = delay_down_rs
            
            # 直接使用RobustScaler归一化结果作为模型训练数据
            window_df["delay_up"] = delay_up_rs
            window_df["delay_down"] = delay_down_rs
            
            # 重命名loss列
            window_df = window_df.rename(columns={
                "loss_down": "loss_dn",
            })

            # 构造新的元数据，保留behavior_id字段
            renamed_meta = {
                "window": window_df,
                "is_first": meta["is_first"],
                "start_time": meta["start_time"],
                "trace_id": meta["trace_id"],
                "behavior_id": behavior_id  # 确保保留行为ID
            }

            renamed_metas.append(renamed_meta)

        return renamed_metas

    # 4. 处理所有数据集
    train_renamed = process_dataset(train)
    val_renamed = process_dataset(val)
    test_renamed = process_dataset(test)

    # 5. 构造返回结果
    renamed_datasets = {
        "train": train_renamed,
        "val": val_renamed,
        "test": test_renamed,
    }

    # 保存所有归一化器到assets中
    assets = {
        "rs_dict": rs_dict,  # 保存所有16个归一化器
    }

    return renamed_datasets, assets




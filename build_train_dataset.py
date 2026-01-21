import os
import argparse  # ← 新增
import numpy as np
import pickle
from tqdm import tqdm
from clustering.cluster_labels import two_stage_clustering, predict_labels_with_model

# ================== 配置 ==================
DATA_DIR = "data"
OUTPUT_DIR = "dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)
WINDOW_SIZE = 64
STRIDE = 32
MAX_DELAY_MS = 2000

def parse_holowan_file(filepath):
    """解析 HoloWAN .txt 文件，跳过元信息直到 '--------'"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                return None
            data_start = None
            for i, line in enumerate(lines):
                if line.strip().startswith("--------"):
                    data_start = i + 1
                    break
            if data_start is None or data_start >= len(lines):
                return None
            data_lines = [line for line in lines[data_start:] if line.strip()]
            if not data_lines:
                return None
            from io import StringIO
            data_str = "".join(data_lines)
            data = np.loadtxt(StringIO(data_str), delimiter=",", dtype=float)
            if data.size == 0:
                return None
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] != 6:
                return None
            return data
    except Exception:
        return None

def log_transform(x):
    return np.log(x + 1.0)

def inverse_log_transform(log_x):
    return np.exp(log_x) - 1.0

def extract_continuous_condition(window):
    d1_log, p1, d2_log, p2 = window.T
    d1 = inverse_log_transform(d1_log)
    d2 = inverse_log_transform(d2_log)
    return np.array([
        np.mean(d1), np.std(d1),
        np.mean(p1), np.mean(p1 > 0.05),
        np.mean(d2), np.std(d2),
        np.mean(p2), np.mean(p2 > 0.05),
    ], dtype=np.float32)

def extract_clustering_features(windows):
    features = []
    for win in windows:
        d1_log, p1, d2_log, p2 = win.T
        d1 = inverse_log_transform(d1_log)
        d2 = inverse_log_transform(d2_log)
        feat = [
            np.mean(d1), np.std(d1),
            np.mean(p1), np.sum(p1 > 0.05),
            np.mean(d2), np.std(d2),
            np.mean(p2), np.sum(p2 > 0.05),
        ]
        features.append(feat)
    return np.array(features, dtype=np.float32)

def build_windows_from_trace(trace):
    from utils.sliding_window import generate_windows
    
    d1, l1, bw1, d2, l2, bw2 = trace.T
    valid_mask = (d1 <= MAX_DELAY_MS) & (d2 <= MAX_DELAY_MS)
    d1 = d1[valid_mask]
    l1 = l1[valid_mask]
    d2 = d2[valid_mask]
    l2 = l2[valid_mask]
    bw1 = bw1[valid_mask]
    bw2 = bw2[valid_mask]
    l1 = np.where(bw1 == 0, 100.0, l1)
    l2 = np.where(bw2 == 0, 100.0, l2)
    p1 = l1 / 100.0
    p2 = l2 / 100.0
    d1_log = log_transform(d1)
    d2_log = log_transform(d2)
    
    # 构建序列数据
    sequence = np.column_stack([d1_log, p1, d2_log, p2])
    
    # 使用通用窗口生成函数
    windows, _ = generate_windows(
        sequence, 
        window_size=WINDOW_SIZE, 
        stride=STRIDE,
        filter_identical=True, 
        sample_interval_ms=100,  # 假设采样间隔为100ms
        max_identical_seconds=10
    )
    
    return windows

def main():
    # ← 新增：命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="gmm_three_class",
                        choices=["kmeans_two_stage", "isolation_forest_two_stage", "gmm_three_class", 
                                 "gmm_four_class", "auto_gmm_bic", "hierarchical", "dbscan"])
    args = parser.parse_args()

    print("🔍 扫描 data/ 目录中的 HoloWAN 回放文件...")
    all_traces = []
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    if not files:
        raise ValueError("未找到 .txt 文件！请检查 data/ 目录")
    for filename in tqdm(files, desc="解析文件"):
        filepath = os.path.join(DATA_DIR, filename)
        trace = parse_holowan_file(filepath)
        if trace is not None and trace.shape[0] > 0:
            all_traces.append(trace)
    if not all_traces:
        raise ValueError("未找到有效 trace！请检查 data/ 目录")
    print(f"✅ 成功加载 {len(all_traces)} 条 trace")

    print("🪟 构建滑动窗口...")
    all_windows = []
    for trace in tqdm(all_traces, desc="生成窗口"):
        windows = build_windows_from_trace(trace)
        all_windows.extend(windows)
    if not all_windows:
        raise ValueError("未生成任何窗口！")
    print(f"🪟 训练窗口数: {len(all_windows)}")

    train_windows = np.array(all_windows, dtype=np.float32)

    # ← 修改：传入 strategy 参数
    discrete_labels, cluster_models = two_stage_clustering(
        train_windows,
        strategy=args.strategy,  # ← 关键修改
        random_state=42,
        output_dir="clustering_analysis"
    )

    print("🔄 提取连续条件...")
    continuous_conditions = []
    for win in tqdm(train_windows, desc="计算条件"):
        cond = extract_continuous_condition(win)
        continuous_conditions.append(cond)
    continuous_conditions = np.array(continuous_conditions, dtype=np.float32)

    n_total = len(train_windows)
    n_test = int(n_total * 0.2)
    indices = np.random.RandomState(42).permutation(n_total)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    dataset = {
        "train": {
            "windows": train_windows[train_idx],
            "labels": [discrete_labels[i] for i in train_idx],
            "conditions": continuous_conditions[train_idx],
        },
        "test": {
            "windows": train_windows[test_idx],
            "labels": [discrete_labels[i] for i in test_idx],
            "conditions": continuous_conditions[test_idx],
        },
    }

    output_pkl = os.path.join(OUTPUT_DIR, f"train_dataset_{args.strategy}.pkl")
    with open(output_pkl, "wb") as f:
        pickle.dump(dataset, f)
    with open(os.path.join(OUTPUT_DIR, f"cluster_models_{args.strategy}.pkl"), "wb") as f:
        pickle.dump(cluster_models, f)

    from collections import Counter
    train_dist = dict(Counter(dataset["train"]["labels"]))
    test_dist = dict(Counter(dataset["test"]["labels"]))
    print(f"\n📊 标签分布:")
    print(f" Train: {train_dist}")
    print(f" Test: {test_dist}")
    print(f"\n✅ 数据集构建完成！")
    print(f"📁 输出目录: {OUTPUT_DIR}/")


    # ========== 新增：打印聚类统计摘要 ==========
    # print("\n📊 Class-wise Statistical Summary:")
    # print("=" * 50)
    #
    # all_labels = discrete_labels  # 所有窗口的标签
    # unique_labels = sorted(set(all_labels))

    # ========== 新增：打印聚类统计摘要（语义化标签） ==========
    print("\n📊 Class-wise Statistical Summary:")
    print("=" * 50)

    all_labels = discrete_labels  # 所有窗口的标签
    unique_labels = sorted(set(all_labels))

    # 根据策略推断标签含义（适配不同聚类策略）
    if args.strategy == "gmm_three_class":
        label_names = {0: "S0", 1: "S1_light", 2: "S1_anomaly"}
    else:
        # kmeans_two_stage / isolation_forest_two_stage: 通常 0=S0, 1=S1_light, -1=S1_anomaly
        label_names = {}
        if -1 in unique_labels:
            label_names[-1] = "S1_anomaly"
            normal_labels = [l for l in unique_labels if l != -1]
            if len(normal_labels) == 2:
                label_names[normal_labels[0]] = "S0"
                label_names[normal_labels[1]] = "S1_light"
            elif len(normal_labels) == 1:
                label_names[normal_labels[0]] = "S0"
        else:
            # 无 anomaly 类
            if len(unique_labels) >= 1:
                label_names[unique_labels[0]] = "S0"
            if len(unique_labels) >= 2:
                label_names[unique_labels[1]] = "S1_light"

    for label in unique_labels:
        name = label_names.get(label, f"{label}")
        indices = [i for i, l in enumerate(all_labels) if l == label]
        subset = train_windows[indices]

        d1_log = subset[:, :, 0]
        p1 = subset[:, :, 1]
        d2_log = subset[:, :, 2]
        p2 = subset[:, :, 3]
        d1 = inverse_log_transform(d1_log)
        d2 = inverse_log_transform(d2_log)

        all_delays = np.concatenate([d1.ravel(), d2.ravel()])
        all_losses = np.concatenate([p1.ravel(), p2.ravel()])

        n_samples = len(indices)
        avg_delay = np.mean(all_delays)
        p95_delay = np.percentile(all_delays, 95)
        avg_loss_pct = np.mean(all_losses) * 100.0
        burst_freq_pct = np.mean(all_losses > 0.05) * 100.0

        print(f"{name}:")
        print(f"  样本数: {n_samples}")
        print(f"  平均延迟: {avg_delay:.2f} ms")
        print(f"  P95 延迟: {p95_delay:.2f} ms")
        print(f"  平均丢包: {avg_loss_pct:.2f} %")
        print(f"  Burst 频率 (>5%): {burst_freq_pct:.2f} %")
    # =========================================

if __name__ == "__main__":
    main()
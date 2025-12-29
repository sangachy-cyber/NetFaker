#!/usr/bin/env python3
"""
展示三个典型类别的窗口数据，包括反归一化后的时延和卡顿情况
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_resources():
    """加载预处理过程中保存的资源文件
    
    Returns:
        dict: 包含qt_up、qt_down以及z-score均值和标准差的字典
    """
    assets_dir = Path("output/assets")
    
    # 加载QuantileTransformer模型
    with open(assets_dir / "qt_up.pkl", "rb") as f:
        qt_up = pickle.load(f)
    
    with open(assets_dir / "qt_down.pkl", "rb") as f:
        qt_down = pickle.load(f)
    
    # 加载z-score均值和标准差
    delay_up_mean = np.load(assets_dir / "delay_up_mean.npy").item()
    delay_up_std = np.load(assets_dir / "delay_up_std.npy").item()
    delay_down_mean = np.load(assets_dir / "delay_down_mean.npy").item()
    delay_down_std = np.load(assets_dir / "delay_down_std.npy").item()
    
    return {
        "qt_up": qt_up,
        "qt_down": qt_down,
        "delay_up_mean": delay_up_mean,
        "delay_up_std": delay_up_std,
        "delay_down_mean": delay_down_mean,
        "delay_down_std": delay_down_std
    }


def load_cluster_results():
    """加载聚类结果
    
    Returns:
        tuple: (cluster_labels, split_labels)
    """
    cluster_dir = Path("output/10d_clustering_analysis")
    
    cluster_labels = np.load(cluster_dir / "cluster_labels.npy")
    split_labels = np.load(cluster_dir / "split_labels.npy")
    
    return cluster_labels, split_labels


def load_window_data():
    """加载窗口数据
    
    Returns:
        list: 窗口数据列表
    """
    datasets_dir = Path("output/datasets")
    window_data = []
    
    # 遍历所有数据集文件
    for split in ["train", "val", "test"]:
        file_path = datasets_dir / f"{split}.jsonl"
        if not file_path.exists():
            continue
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                window_data.append(record)
    
    return window_data


def inverse_transform_delay(delays, qt_model, mean=0.0, std=1.0):
    """对时延数据进行反归一化，包括z-score反变换和QuantileTransformer反变换
    
    Args:
        delays: 归一化后的时延数据
        qt_model: QuantileTransformer模型
        mean: z-score均值
        std: z-score标准差
        
    Returns:
        np.array: 反归一化后的时延数据
    """
    # 1. 先进行z-score反变换
    delays_zscore_inv = delays * std + mean
    # 2. 确保输入是二维数组
    delays_2d = delays_zscore_inv.reshape(-1, 1)
    # 3. 反QuantileTransformer变换
    inverse_delays = qt_model.inverse_transform(delays_2d)
    # 4. 转换回一维数组
    return inverse_delays.flatten()


def select_typical_cases(window_data, cluster_labels, num_cases=3):
    """从每个聚类中选择一个典型案例
    
    Args:
        window_data: 窗口数据列表
        cluster_labels: 聚类标签数组
        num_cases: 每个聚类选择的案例数量
        
    Returns:
        dict: 每个聚类对应的典型案例
    """
    typical_cases = {}
    unique_clusters = np.unique(cluster_labels)
    
    for cluster_id in unique_clusters[:num_cases]:
        # 找出当前聚类的所有窗口
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        
        # 选择第一个窗口作为典型案例
        idx = cluster_indices[0]
        typical_cases[cluster_id] = {
            "window": window_data[idx],
            "cluster_id": cluster_id
        }
    
    return typical_cases


def plot_typical_case(ax, case_data, assets, title=""):
    """绘制典型案例的时延和卡顿情况
    
    Args:
        ax: matplotlib轴对象
        case_data: 典型案例数据
        assets: 包含qt_up和qt_down的字典
        title: 图表标题
    """
    # 提取窗口数据
    window_df = pd.DataFrame(case_data["window"]["window"])
    # 只取前100个点
    window_df = window_df.head(100)
    
    # 反归一化时延数据
    del_up = window_df["del_up"].values
    del_dn = window_df["del_dn"].values
    loss_up = window_df["loss_up"].values
    loss_dn = window_df["loss_dn"].values
    
    inverse_del_up = inverse_transform_delay(del_up, assets["qt_up"], 
                                          assets["delay_up_mean"], assets["delay_up_std"])
    inverse_del_dn = inverse_transform_delay(del_dn, assets["qt_down"], 
                                          assets["delay_down_mean"], assets["delay_down_std"])
    
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


def main():
    """主函数"""
    print("=== 展示典型类别案例 ===")
    
    # 1. 加载资源文件
    print("\n1. 加载资源文件...")
    assets = load_resources()
    
    # 2. 加载聚类结果
    print("2. 加载聚类结果...")
    cluster_labels, split_labels = load_cluster_results()
    
    # 3. 加载窗口数据
    print("3. 加载窗口数据...")
    window_data = load_window_data()
    
    # 4. 选择典型案例
    print("4. 选择典型案例...")
    typical_cases = select_typical_cases(window_data, cluster_labels, num_cases=3)
    
    # 5. 绘制典型案例
    print("5. 绘制典型案例...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    for i, (cluster_id, case_data) in enumerate(typical_cases.items()):
        if i >= len(axes):
            break
        
        title = f"类别 {cluster_id} - 典型案例"
        plot_typical_case(axes[i], case_data, assets, title)
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = Path("output/typical_cases")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "typical_cases.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"图表已保存到: {output_path}")
    
    # 显示图表
    plt.show()
    
    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()

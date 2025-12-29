#!/usr/bin/env python3
"""
网络行为发现可视化模块，提供可复用的可视化功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any


# 行为名称映射
BEHAVIOR_NAMES = {
    0: "STABLE",
    1: "WEAK_BURST",
    2: "FREQUENT_FLUCTUATION",
    3: "HIGH_DELAY_NO_LOSS",
    4: "HIGH_LOSS_STEADY",
    5: "LOW_DELAY_HIGH_LOSS",
    6: "STRONG_BURST",
    7: "INSTANT_SPIKE",
    8: "INVALID"
}


def calculate_window_stats(window_df: pd.DataFrame) -> Dict[str, Any]:
    """计算窗口的统计信息
    
    Args:
        window_df: 窗口数据DataFrame
        
    Returns:
        dict: 包含统计信息的字典
    """
    # 获取时延和丢包率数据
    if "delay_up_origin" in window_df.columns and "delay_down_origin" in window_df.columns:
        delay_up = window_df["delay_up_origin"].values
        delay_down = window_df["delay_down_origin"].values
        loss_up = window_df.get("loss_up_origin", np.zeros(len(window_df))).values
        loss_dn = window_df.get("loss_down_origin", np.zeros(len(window_df))).values
    elif "delay_up" in window_df.columns and "delay_down" in window_df.columns:
        delay_up = window_df["delay_up"].values
        delay_down = window_df["delay_down"].values
        loss_up = window_df.get("loss_up", np.zeros(len(window_df))).values
        loss_dn = window_df.get("loss_dn", np.zeros(len(window_df))).values
    elif "delay1" in window_df.columns and "delay2" in window_df.columns:
        delay_up = window_df["delay1"].values
        delay_down = window_df["delay2"].values
        loss_up = window_df.get("loss_rate1", np.zeros(len(window_df))).values
        loss_dn = window_df.get("loss_rate2", np.zeros(len(window_df))).values
    else:
        return None
    
    # 计算统计信息
    return {
        "delay_up": {
            "mean": np.mean(delay_up),
            "std": np.std(delay_up),
            "max": np.max(delay_up),
            "min": np.min(delay_up)
        },
        "delay_down": {
            "mean": np.mean(delay_down),
            "std": np.std(delay_down),
            "max": np.max(delay_down),
            "min": np.min(delay_down)
        },
        "loss_up": {
            "mean": np.mean(loss_up),
            "std": np.std(loss_up),
            "max": np.max(loss_up)
        },
        "loss_down": {
            "mean": np.mean(loss_dn),
            "std": np.std(loss_dn),
            "max": np.max(loss_dn)
        },
        "delay_up_data": delay_up,
        "delay_down_data": delay_down,
        "loss_up_data": loss_up,
        "loss_down_data": loss_dn
    }


def plot_behavior_case(ax: plt.Axes, case_data: Dict, stats: Dict[str, Any], title: str = "") -> None:
    """绘制单个行为案例的时延和丢包情况，并显示统计信息
    
    Args:
        ax: matplotlib轴对象
        case_data: 典型案例数据
        stats: 窗口统计信息
        title: 图表标题
    """
    if stats is None:
        ax.text(0.5, 0.5, "无法获取统计信息", ha='center', va='center', transform=ax.transAxes)
        return
    
    # 创建上下两个子图
    ax1 = ax
    ax2 = ax1.twinx()
    
    # 绘制上行和下行时延
    l1, = ax1.plot(stats["delay_up_data"], label="上行时延 (ms)", color="blue", alpha=0.8)
    l2, = ax1.plot(stats["delay_down_data"], label="下行时延 (ms)", color="red", alpha=0.8)
    
    # 绘制上行和下行丢包率，解决数值相同导致的重叠问题
    # 为下行丢包率添加微小偏移，确保即使数值相同也能显示两条线
    loss_up = stats["loss_up_data"]
    loss_dn = stats["loss_down_data"]
    loss_dn_offset = loss_dn + 0.005  # 添加0.5%的偏移
    
    # 上行丢包率：绿色实线 + 圆形标记 + 适中线宽 + 高频标记
    l3, = ax2.plot(loss_up, label="上行丢包率", color="#00FF00", linestyle="-", alpha=0.9, 
                   marker='o', markersize=4, markevery=4, linewidth=1.2)
    # 下行丢包率：红色虚线 + 方形标记 + 适中线宽 + 高频标记 + 微小偏移
    l4, = ax2.plot(loss_dn_offset, label="下行丢包率", color="#FF0000", linestyle="--", alpha=0.9, 
                   marker='s', markersize=4, markevery=4, linewidth=1.2)
    
    # 添加丢包标记（丢包时）
    loss_up_indices = np.where(loss_up > 0)[0]
    loss_dn_indices = np.where(loss_dn > 0)[0]
    
    if len(loss_up_indices) > 0:
        ax2.scatter(loss_up_indices, loss_up[loss_up_indices], 
                  color="green", marker="x", s=30, alpha=0.8)
    if len(loss_dn_indices) > 0:
        ax2.scatter(loss_dn_indices, loss_dn[loss_dn_indices], 
                  color="red", marker="x", s=30, alpha=0.8)
    
    # 设置轴标签和标题
    ax1.set_xlabel("时间点")
    ax1.set_ylabel("时延 (ms)")
    ax2.set_ylabel("丢包率")
    ax1.set_title(title, fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 统一时延坐标范围：0-2000ms，调整半对数刻度使0-100ms分布更合理
    ax1.set_ylim(0, 2000)
    ax1.set_yscale('symlog', linthresh=100, linscale=0.8)
    ax1.set_yticks([0, 100, 200, 1000, 2000])
    ax1.set_yticklabels(['0', '100', '200', '1000', '2000'])
    
    # 统一丢包率坐标范围
    ax2.set_ylim(-0.01, 1.01)
    
    # 合并图例，只在标记为first的图表显示
    if "first" in title:
        lines = [l1, l2, l3, l4]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="upper right", fontsize=8)
    
    # 添加统计信息文本框
    stats_text = f"上行统计:\n" + \
                f"  平均时延: {stats['delay_up']['mean']:.1f} ms\n" + \
                f"  时延标准差: {stats['delay_up']['std']:.1f} ms\n" + \
                f"  平均丢包率: {stats['loss_up']['mean']:.3f}\n" + \
                f"\n下行统计:\n" + \
                f"  平均时延: {stats['delay_down']['mean']:.1f} ms\n" + \
                f"  时延标准差: {stats['delay_down']['std']:.1f} ms\n" + \
                f"  平均丢包率: {stats['loss_down']['mean']:.3f}"
    
    # 在图中添加统计信息
    ax1.text(0.02, 0.02, stats_text, transform=ax1.transAxes, fontsize=8, 
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def generate_behavior_typical_cases(typical_cases: Dict[int, List[Dict]], 
                                  output_path: Path = None, 
                                  figsize: tuple = (18, None)) -> plt.Figure:
    """生成行为典型案例的可视化图表
    
    Args:
        typical_cases: 典型案例数据，键为行为ID，值为该行为的案例列表
        output_path: 输出文件路径，若为None则不保存
        figsize: 图表尺寸，默认(18, None)，None表示自动计算高度
        
    Returns:
        plt.Figure: 生成的图表对象
    """
    # 设计布局：每类行为一行，一行3个案例
    num_behaviors = len(typical_cases)
    num_cases_per_behavior = 3
    
    # 计算图表高度
    if figsize[1] is None:
        fig_height = 4 * num_behaviors  # 每行4英寸
        figsize = (figsize[0], fig_height)
    
    # 创建图表：num_behaviors行，3列
    fig, axes = plt.subplots(num_behaviors, num_cases_per_behavior, figsize=figsize)
    
    # 确保axes是2D数组
    if num_behaviors == 1:
        axes = [axes]
    
    # 遍历每个行为类型
    for behavior_row, (behavior_id, cases) in enumerate(typical_cases.items()):
        behavior_name = BEHAVIOR_NAMES.get(behavior_id, f"未知 ({behavior_id})")
        
        # 遍历每个案例
        for case_col, case_data in enumerate(cases):
            if case_col >= num_cases_per_behavior:
                break
            
            # 获取当前轴
            ax = axes[behavior_row][case_col]
            
            # 提取窗口数据
            window_data_item = case_data["window"]
            if isinstance(window_data_item, dict) and "window" in window_data_item:
                window_df = pd.DataFrame(window_data_item["window"])
            else:
                window_df = pd.DataFrame(window_data_item)
            
            # 计算统计信息
            stats = calculate_window_stats(window_df.head(100))
            
            # 设置标题，在第一个案例添加"first"标记
            case_title = f"行为类型: {behavior_id} - {behavior_name} - 案例 {case_col+1}"
            if case_col == 0:
                case_title = f"first - {case_title}"  # 用于标记第一个案例
            
            # 绘制案例
            plot_behavior_case(ax, case_data, stats, case_title)
    
    # 调整图表间距，减少空白
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3, wspace=0.2)
    
    # 保存图表
    if output_path is not None:
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    
    return fig


def select_behavior_typical_cases(window_data: List[Dict], 
                                 behavior_labels: List[int], 
                                 num_cases: int = 3) -> Dict[int, List[Dict]]:
    """从每个行为类型中选择典型案例
    
    Args:
        window_data: 窗口数据列表
        behavior_labels: 行为标签数组
        num_cases: 每个行为类型选择的案例数量
        
    Returns:
        dict: 每个行为类型对应的典型案例
    """
    typical_cases = {}
    unique_behaviors = np.unique(behavior_labels)
    
    for behavior_id in unique_behaviors:
        if behavior_id == 8:  # 跳过INVALID类型
            continue
            
        # 找出当前行为类型的所有窗口
        behavior_indices = np.where(behavior_labels == behavior_id)[0]
        if len(behavior_indices) == 0:
            continue
        
        # 为了确保选择的案例更具代表性，我们随机选择样本
        np.random.seed(42)  # 设置随机种子，确保结果可复现
        selected_indices = np.random.choice(behavior_indices, 
                                           size=min(num_cases, len(behavior_indices)), 
                                           replace=False)
        
        cases = []
        for idx in selected_indices:
            cases.append({
                "window": window_data[idx],
                "behavior_id": behavior_id
            })
        
        typical_cases[behavior_id] = cases
    
    return typical_cases


def main():
    """主函数，用于测试可视化功能"""
    pass


if __name__ == "__main__":
    main()

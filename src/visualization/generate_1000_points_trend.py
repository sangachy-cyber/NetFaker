#!/usr/bin/env python3

"""将1000个原始数据点画到一起展示趋势
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 读取生成样本数据
def load_generated_samples(file_path):
    """加载生成样本数据"""
    generated_samples = np.load(file_path)
    # 生成样本格式: (num_samples, 100, 5)
    # 特征0: 时间步（0-9.9）
    # 特征1: 上行时延（实际值，秒）
    # 特征2: 下行时延（实际值，秒）
    # 特征3: 上行丢包率
    # 特征4: 下行丢包率
    return generated_samples

# 生成1000个点的趋势图
def generate_1000_points_trend(generated_samples, output_path):
    """生成1000个点的趋势图"""
    # 合并所有样本，得到1000个连续的数据点
    # 将(10, 100, 5)转换为(1000, 5)
    all_points = generated_samples.reshape(-1, 5)

    # 数据点索引（0-999）
    point_indices = np.arange(all_points.shape[0])

    # 时延数据已经是毫秒单位
    up_delay_ms = all_points[:, 1]
    down_delay_ms = all_points[:, 2]

    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(20, 15))
    fig.suptitle("1000个原始数据点的时延趋势（实际尺度，时延单位：毫秒）", fontsize=20, fontweight="bold")

    # 上行时延趋势
    ax1 = axes[0]
    ax1.plot(point_indices, up_delay_ms, color="blue", linewidth=1, alpha=0.8)
    ax1.set_title("上行时延趋势", fontsize=16, fontweight="bold")
    ax1.set_xlabel("数据点索引（0-999）", fontsize=12)
    ax1.set_ylabel("上行时延 (毫秒)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=20, color="red", linestyle="--", linewidth=1.5, label="20ms阈值")
    ax1.legend(fontsize=12)

    # 下行时延趋势
    ax2 = axes[1]
    ax2.plot(point_indices, down_delay_ms, color="green", linewidth=1, alpha=0.8)
    ax2.set_title("下行时延趋势", fontsize=16, fontweight="bold")
    ax2.set_xlabel("数据点索引（0-999）", fontsize=12)
    ax2.set_ylabel("下行时延 (毫秒)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=20, color="red", linestyle="--", linewidth=1.5, label="20ms阈值")
    ax2.legend(fontsize=12)

    # 调整布局
    plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.97])

    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"1000个点的趋势图已保存到: {output_path}")

# 生成包含丢包率的综合趋势图
def generate_comprehensive_trend(generated_samples, output_path):
    """生成包含丢包率的综合趋势图"""
    # 合并所有样本，得到1000个连续的数据点
    all_points = generated_samples.reshape(-1, 5)

    # 数据点索引（0-999）
    point_indices = np.arange(all_points.shape[0])

    # 将时延从秒转换为毫秒
    up_delay_ms = all_points[:, 1] * 1000
    down_delay_ms = all_points[:, 2] * 1000

    # 丢包率
    up_loss = all_points[:, 3]
    down_loss = all_points[:, 4]

    # 创建图表
    fig, axes = plt.subplots(4, 1, figsize=(20, 25))
    fig.suptitle("1000个原始数据点的综合趋势", fontsize=20, fontweight="bold")

    # 1. 上行时延趋势
    ax1 = axes[0]
    ax1.plot(point_indices, up_delay_ms, color="blue", linewidth=1, alpha=0.8)
    ax1.set_title("上行时延趋势（毫秒）", fontsize=16, fontweight="bold")
    ax1.set_xlabel("数据点索引（0-999）", fontsize=12)
    ax1.set_ylabel("上行时延 (毫秒)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=20, color="red", linestyle="--", linewidth=1.5, label="20ms阈值")
    ax1.legend(fontsize=12)

    # 2. 下行时延趋势
    ax2 = axes[1]
    ax2.plot(point_indices, down_delay_ms, color="green", linewidth=1, alpha=0.8)
    ax2.set_title("下行时延趋势（毫秒）", fontsize=16, fontweight="bold")
    ax2.set_xlabel("数据点索引（0-999）", fontsize=12)
    ax2.set_ylabel("下行时延 (毫秒)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=20, color="red", linestyle="--", linewidth=1.5, label="20ms阈值")
    ax2.legend(fontsize=12)

    # 3. 上行丢包率趋势
    ax3 = axes[2]
    ax3.plot(point_indices, up_loss, color="orange", linewidth=1, alpha=0.8)
    ax3.set_title("上行丢包率趋势", fontsize=16, fontweight="bold")
    ax3.set_xlabel("数据点索引（0-999）", fontsize=12)
    ax3.set_ylabel("上行丢包率", fontsize=12)
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3)

    # 4. 下行丢包率趋势
    ax4 = axes[3]
    ax4.plot(point_indices, down_loss, color="purple", linewidth=1, alpha=0.8)
    ax4.set_title("下行丢包率趋势", fontsize=16, fontweight="bold")
    ax4.set_xlabel("数据点索引（0-999）", fontsize=12)
    ax4.set_ylabel("下行丢包率", fontsize=12)
    ax4.set_ylim([0, 1])
    ax4.grid(True, alpha=0.3)

    # 调整布局
    plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.97])

    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"综合趋势图已保存到: {output_path}")

# 主函数
def main():
    # 数据文件路径
    generated_file = Path("output/run_20251223_020748_UTC/visualization/generated_samples.npy")
    output_dir = Path("output/run_20251223_020748_UTC/visualization")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载生成样本数据
    print("加载生成样本数据...")
    generated_samples = load_generated_samples(generated_file)
    print(f"生成样本形状: {generated_samples.shape}")
    print(f"总数据点数量: {generated_samples.shape[0] * generated_samples.shape[1]}")

    # 生成1000个点的时延趋势图
    output_path1 = output_dir / "1000_points_latency_trend.png"
    generate_1000_points_trend(generated_samples, output_path1)

    # 生成综合趋势图
    output_path2 = output_dir / "1000_points_comprehensive_trend.png"
    generate_comprehensive_trend(generated_samples, output_path2)

    print("\n所有趋势图生成完成!")

if __name__ == "__main__":
    main()

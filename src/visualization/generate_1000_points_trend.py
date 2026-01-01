#!/usr/bin/env python3

"""将1000个原始数据点画到一起展示趋势
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 读取生成样本数据
def load_generated_samples(file_path):
    """加载生成样本数据"""
    generated_samples = np.load(file_path)
    # 生成样本格式: (3, 4, 100) - 3类样本，每类4个特征，100个时间步
    # 特征0: 上行时延（毫秒）
    # 特征1: 下行时延（毫秒）
    # 特征2: 上行丢包率
    # 特征3: 下行丢包率
    return generated_samples

# 生成3类样本的趋势图
def generate_3_samples_trend(generated_samples, output_path):
    """生成3类样本每类100个点的趋势图"""
    # 样本形状: (3, 4, 100) - 3类样本，4个特征，100个时间步
    # 特征0: 上行时延（毫秒）
    # 特征1: 下行时延（毫秒）
    # 特征2: 上行丢包率
    # 特征3: 下行丢包率
    
    # 时间步索引（0-99）
    time_steps = np.arange(generated_samples.shape[2])
    
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(20, 15))
    fig.suptitle("3类样本的时延趋势（每类100个点，时延单位：毫秒）", fontsize=20, fontweight="bold")
    
    # 样本颜色
    colors = ["blue", "green", "red"]
    sample_labels = ["样本1", "样本2", "样本3"]
    
    # 上行时延趋势 - 每类样本分开绘制
    ax1 = axes[0]
    for i in range(generated_samples.shape[0]):
        up_delay_ms = generated_samples[i, 0, :]  # 第i类样本，上行时延特征，所有时间步
        ax1.plot(time_steps, up_delay_ms, color=colors[i], linewidth=2, alpha=0.8, label=sample_labels[i])
    ax1.set_title("上行时延趋势", fontsize=16, fontweight="bold")
    ax1.set_xlabel("时间步（0-99）", fontsize=12)
    ax1.set_ylabel("上行时延 (毫秒)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=20, color="red", linestyle="--", linewidth=1.5, label="20ms阈值")
    ax1.legend(fontsize=12)
    
    # 下行时延趋势 - 每类样本分开绘制
    ax2 = axes[1]
    for i in range(generated_samples.shape[0]):
        down_delay_ms = generated_samples[i, 1, :]  # 第i类样本，下行时延特征，所有时间步
        ax2.plot(time_steps, down_delay_ms, color=colors[i], linewidth=2, alpha=0.8, label=sample_labels[i])
    ax2.set_title("下行时延趋势", fontsize=16, fontweight="bold")
    ax2.set_xlabel("时间步（0-99）", fontsize=12)
    ax2.set_ylabel("下行时延 (毫秒)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=20, color="red", linestyle="--", linewidth=1.5, label="20ms阈值")
    ax2.legend(fontsize=12)
    
    # 调整布局
    plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.97])
    
    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"3类样本的趋势图已保存到: {output_path}")

# 生成包含丢包率的综合趋势图
def generate_comprehensive_trend(generated_samples, output_path):
    """生成包含丢包率的综合趋势图"""
    # 样本形状: (3, 4, 100) - 3类样本，4个特征，100个时间步
    # 特征0: 上行时延（毫秒）
    # 特征1: 下行时延（毫秒）
    # 特征2: 上行丢包率
    # 特征3: 下行丢包率
    
    # 时间步索引（0-99）
    time_steps = np.arange(generated_samples.shape[2])
    
    # 创建图表
    fig, axes = plt.subplots(4, 1, figsize=(20, 25))
    fig.suptitle("3类样本的综合趋势（每类100个点）", fontsize=20, fontweight="bold")
    
    # 样本颜色
    colors = ["blue", "green", "red"]
    sample_labels = ["样本1", "样本2", "样本3"]
    
    # 1. 上行时延趋势 - 每类样本分开绘制
    ax1 = axes[0]
    for i in range(generated_samples.shape[0]):
        up_delay_ms = generated_samples[i, 0, :]  # 第i类样本，上行时延特征
        ax1.plot(time_steps, up_delay_ms, color=colors[i], linewidth=2, alpha=0.8, label=sample_labels[i])
    ax1.set_title("上行时延趋势（毫秒）", fontsize=16, fontweight="bold")
    ax1.set_xlabel("时间步（0-99）", fontsize=12)
    ax1.set_ylabel("上行时延 (毫秒)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=20, color="red", linestyle="--", linewidth=1.5, label="20ms阈值")
    ax1.legend(fontsize=12)
    
    # 2. 下行时延趋势 - 每类样本分开绘制
    ax2 = axes[1]
    for i in range(generated_samples.shape[0]):
        down_delay_ms = generated_samples[i, 1, :]  # 第i类样本，下行时延特征
        ax2.plot(time_steps, down_delay_ms, color=colors[i], linewidth=2, alpha=0.8, label=sample_labels[i])
    ax2.set_title("下行时延趋势（毫秒）", fontsize=16, fontweight="bold")
    ax2.set_xlabel("时间步（0-99）", fontsize=12)
    ax2.set_ylabel("下行时延 (毫秒)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=20, color="red", linestyle="--", linewidth=1.5, label="20ms阈值")
    ax2.legend(fontsize=12)
    
    # 3. 上行丢包率趋势 - 每类样本分开绘制
    ax3 = axes[2]
    for i in range(generated_samples.shape[0]):
        up_loss = generated_samples[i, 2, :]  # 第i类样本，上行丢包率特征
        ax3.plot(time_steps, up_loss, color=colors[i], linewidth=2, alpha=0.8, label=sample_labels[i])
    ax3.set_title("上行丢包率趋势", fontsize=16, fontweight="bold")
    ax3.set_xlabel("时间步（0-99）", fontsize=12)
    ax3.set_ylabel("上行丢包率", fontsize=12)
    ax3.set_ylim([-0.01, 1.01])
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=12)
    
    # 4. 下行丢包率趋势 - 每类样本分开绘制
    ax4 = axes[3]
    for i in range(generated_samples.shape[0]):
        down_loss = generated_samples[i, 3, :]  # 第i类样本，下行丢包率特征
        ax4.plot(time_steps, down_loss, color=colors[i], linewidth=2, alpha=0.8, label=sample_labels[i])
    ax4.set_title("下行丢包率趋势", fontsize=16, fontweight="bold")
    ax4.set_xlabel("时间步（0-99）", fontsize=12)
    ax4.set_ylabel("下行丢包率", fontsize=12)
    ax4.set_ylim([-0.01, 1.01])
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=12)
    
    # 调整布局
    plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.97])
    
    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"综合趋势图已保存到: {output_path}")

# 为了保持兼容性，保留generate_1000_points_trend函数名
def generate_1000_points_trend(generated_samples, output_path):
    """生成3类样本的时延趋势图（保持函数名兼容性）"""
    generate_3_samples_trend(generated_samples, output_path)

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
    print(f"总数据点数量: {generated_samples.shape[0] * generated_samples.shape[2]}")
    print(f"样本类别数量: {generated_samples.shape[0]}")
    print(f"每个样本的时间步数量: {generated_samples.shape[2]}")

    # 生成3类样本的时延趋势图
    output_path1 = output_dir / "3_samples_latency_trend.png"
    generate_3_samples_trend(generated_samples, output_path1)
    
    # 生成1000个点的时延趋势图（保持兼容）
    output_path1_compat = output_dir / "1000_points_latency_trend.png"
    generate_1000_points_trend(generated_samples, output_path1_compat)

    # 生成综合趋势图
    output_path2 = output_dir / "3_samples_comprehensive_trend.png"
    generate_comprehensive_trend(generated_samples, output_path2)
    
    # 生成1000个点的综合趋势图（保持兼容）
    output_path2_compat = output_dir / "1000_points_comprehensive_trend.png"
    generate_comprehensive_trend(generated_samples, output_path2_compat)

    print("\n所有趋势图生成完成!")

if __name__ == "__main__":
    main()

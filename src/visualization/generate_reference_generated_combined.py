#!/usr/bin/env python3

"""将参考样本和生成样本的1000个点绘制在一起，展示趋势对比
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 加载参考样本的1000个点
def load_reference_1000_points(reference_file, qt_up, qt_down):
    """从参考样本中加载1000个点，选取连续的10个窗口，每个窗口100个点
    
    Args:
        reference_file: 参考样本文件路径
        qt_up: 上行延迟反归一化器
        qt_down: 下行延迟反归一化器
        
    Returns:
        np.ndarray: 连续的1000个数据点，shape (1000, 4)
    """
    # 读取所有有效窗口及其起始时间
    all_windows = []
    all_start_times = []
    with open(reference_file) as f:
        for line in f:
            data = json.loads(line)
            if data.get("keep", True):
                window_data = data["window"]
                start_time = data["start_time"]
                # 确保窗口数据完整（100个点）
                if len(window_data) >= 100:
                    # 提取窗口的前100个点
                    window_points = []
                    for row in window_data[:100]:
                        window_points.append([
                                row["delay_up"],
                                row["delay_down"],
                                row["loss_up"],
                                row["loss_dn"],
                            ])
                    all_windows.append(window_points)
                    all_start_times.append(start_time)
    
    # 确保有足够的连续窗口
    import numpy as np
    required_windows = 10
    if len(all_windows) < required_windows:
        # 如果窗口数量不足，使用所有可用窗口
        selected_windows = all_windows
        print(f"警告：测试集中有效窗口数量不足 {required_windows} 个，仅使用 {len(all_windows)} 个")
    else:
        # 随机选择起始位置，确保有连续的10个窗口
        max_start_idx = len(all_windows) - required_windows
        start_idx = np.random.randint(0, max_start_idx + 1)
        # 选取连续的10个窗口
        selected_windows = all_windows[start_idx:start_idx + required_windows]
    
    # 将连续窗口合并为1000个点
    reference_samples = []
    for window in selected_windows:
        reference_samples.extend(window)
    
    # 转换为numpy数组
    reference_samples = np.array(reference_samples)
    # 调整形状为 (1, 1000, 4)
    reference_samples = reference_samples.reshape(1, -1, 4)

    # 注意：此处的del_up和del_dn是归一化值，可以为负
    # 反变换后会转换为实际时延，不需要在此处修复负值

    # 执行反归一化，将参考样本转换为实际时延值
    reference_samples_actual = reference_samples.copy()

    # 对每个样本进行反归一化
    for i in range(reference_samples.shape[0]):
        # 提取延迟特征 - 这里使用的是归一化后的del_up和del_dn
        # 注意：这些是已经归一化过的数据，需要用对应的变换器反变换
        del_up_norm = reference_samples[i, :, 0].reshape(-1, 1)
        del_dn_norm = reference_samples[i, :, 1].reshape(-1, 1)

        # 执行反变换 - 上行时延用qt_up，下行时延用qt_down
        del_up_actual = qt_up.inverse_transform(del_up_norm).flatten()
        del_dn_actual = qt_down.inverse_transform(del_dn_norm).flatten()

        # 更新样本数据
        reference_samples_actual[i, :, 0] = del_up_actual
        reference_samples_actual[i, :, 1] = del_dn_actual

    # 将(1, 1000, 4)转换为(1000, 4)
    reference_samples_actual = reference_samples_actual.reshape(-1, 4)

    return reference_samples_actual

# 加载生成样本的1000个点
def load_generated_1000_points(generated_file):
    """从生成样本中加载1000个点"""
    generated_samples = np.load(generated_file)
    # 生成样本格式: (10, 100, 5)
    # 特征0: 时间步（0-9.9）
    # 特征1: 上行时延（实际值，毫秒）
    # 特征2: 下行时延（实际值，毫秒）
    # 特征3: 上行丢包率
    # 特征4: 下行丢包率

    # 将(10, 100, 5)转换为(1000, 5)
    all_points = generated_samples.reshape(-1, 5)

    # 提取实际的时延值和丢包率，去掉时间步
    # 转换为：上行时延(秒), 下行时延(秒), 上行丢包率, 下行丢包率
    generated_samples_actual = all_points[:, 1:5]

    return generated_samples_actual

# 生成参考样本和生成样本的1000点对比图
def generate_combined_1000_points(reference_points, generated_points, output_path):
    """生成参考样本和生成样本的1000点对比图"""
    # 确保参考样本和生成样本点数一致
    min_points = min(reference_points.shape[0], generated_points.shape[0])
    
    # 为参考样本和生成样本生成相同的时间索引
    # 假设每个点间隔0.1秒，总时长100秒（1000点）
    time_seconds = np.arange(min_points) * 0.1
    
    # 截取相同数量的点
    ref_up_delay_ms = reference_points[:min_points, 0]
    ref_down_delay_ms = reference_points[:min_points, 1]
    gen_up_delay_ms = generated_points[:min_points, 0]
    gen_down_delay_ms = generated_points[:min_points, 1]

    # 丢包率 - 本函数仅绘制时延对比，不使用丢包率数据

    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(20, 15))
    fig.suptitle(f"参考样本与生成样本的{min_points}个点时延趋势对比（实际尺度，时延单位：毫秒）", fontsize=20, fontweight="bold")

    # 1. 上行时延对比
    ax1 = axes[0]
    ax1.plot(time_seconds, ref_up_delay_ms, color="blue", linewidth=1, alpha=0.8, label="参考样本")
    ax1.plot(time_seconds, gen_up_delay_ms, color="orange", linewidth=1, alpha=0.8, label="生成样本")
    ax1.set_title("上行时延趋势对比", fontsize=16, fontweight="bold")
    ax1.set_xlabel("时间 (秒)", fontsize=12)
    ax1.set_ylabel("上行时延 (毫秒)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=20, color="red", linestyle="--", linewidth=1.5, label="20ms阈值")
    ax1.legend(fontsize=12)

    # 2. 下行时延对比
    ax2 = axes[1]
    ax2.plot(time_seconds, ref_down_delay_ms, color="green", linewidth=1, alpha=0.8, label="参考样本")
    ax2.plot(time_seconds, gen_down_delay_ms, color="purple", linewidth=1, alpha=0.8, label="生成样本")
    ax2.set_title("下行时延趋势对比", fontsize=16, fontweight="bold")
    ax2.set_xlabel("时间 (秒)", fontsize=12)
    ax2.set_ylabel("下行时延 (毫秒)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=20, color="red", linestyle="--", linewidth=1.5, label="20ms阈值")
    ax2.legend(fontsize=12)

    # 调整布局
    plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.97])

    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"参考样本与生成样本1000点对比图已保存到: {output_path}")

# 生成包含丢包率的综合对比图
def generate_combined_comprehensive(reference_points, generated_points, output_path):
    """生成包含丢包率的综合对比图"""
    # 确保参考样本和生成样本点数一致
    min_points = min(reference_points.shape[0], generated_points.shape[0])
    
    # 为参考样本和生成样本生成相同的时间索引
    # 假设每个点间隔0.1秒，总时长100秒（1000点）
    time_seconds = np.arange(min_points) * 0.1
    
    # 截取相同数量的点
    # 直接使用毫秒单位的时延数据
    ref_up_delay_ms = reference_points[:min_points, 0]
    ref_down_delay_ms = reference_points[:min_points, 1]
    gen_up_delay_ms = generated_points[:min_points, 0]
    gen_down_delay_ms = generated_points[:min_points, 1]

    # 丢包率
    ref_up_loss = reference_points[:min_points, 2]
    ref_down_loss = reference_points[:min_points, 3]
    gen_up_loss = generated_points[:min_points, 2]
    gen_down_loss = generated_points[:min_points, 3]

    # 创建图表
    fig, axes = plt.subplots(4, 1, figsize=(20, 25))
    fig.suptitle("参考样本与生成样本的1000个点综合趋势对比", fontsize=20, fontweight="bold")

    # 1. 上行时延对比
    ax1 = axes[0]
    ax1.plot(time_seconds, ref_up_delay_ms, color="blue", linewidth=1, alpha=0.8, label="参考样本")
    ax1.plot(time_seconds, gen_up_delay_ms, color="orange", linewidth=1, alpha=0.8, label="生成样本")
    ax1.set_title("上行时延趋势对比（毫秒）", fontsize=16, fontweight="bold")
    ax1.set_xlabel("时间 (秒)", fontsize=12)
    ax1.set_ylabel("上行时延 (毫秒)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=20, color="red", linestyle="--", linewidth=1.5, label="20ms阈值")
    ax1.legend(fontsize=12)

    # 2. 下行时延对比
    ax2 = axes[1]
    ax2.plot(time_seconds, ref_down_delay_ms, color="green", linewidth=1, alpha=0.8, label="参考样本")
    ax2.plot(time_seconds, gen_down_delay_ms, color="purple", linewidth=1, alpha=0.8, label="生成样本")
    ax2.set_title("下行时延趋势对比（毫秒）", fontsize=16, fontweight="bold")
    ax2.set_xlabel("时间 (秒)", fontsize=12)
    ax2.set_ylabel("下行时延 (毫秒)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=20, color="red", linestyle="--", linewidth=1.5, label="20ms阈值")
    ax2.legend(fontsize=12)

    # 3. 上行丢包率对比
    ax3 = axes[2]
    ax3.plot(time_seconds, ref_up_loss, color="blue", linewidth=1, alpha=0.8, label="参考样本")
    ax3.plot(time_seconds, gen_up_loss, color="orange", linewidth=1, alpha=0.8, label="生成样本")
    ax3.set_title("上行丢包率趋势对比", fontsize=16, fontweight="bold")
    ax3.set_xlabel("时间 (秒)", fontsize=12)
    ax3.set_ylabel("上行丢包率", fontsize=12)
    ax3.set_ylim([-0.01, 1.01])
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=12)

    # 4. 下行丢包率对比
    ax4 = axes[3]
    ax4.plot(time_seconds, ref_down_loss, color="green", linewidth=1, alpha=0.8, label="参考样本")
    ax4.plot(time_seconds, gen_down_loss, color="purple", linewidth=1, alpha=0.8, label="生成样本")
    ax4.set_title("下行丢包率趋势对比", fontsize=16, fontweight="bold")
    ax4.set_xlabel("时间 (秒)", fontsize=12)
    ax4.set_ylabel("下行丢包率", fontsize=12)
    ax4.set_ylim([-0.01, 1.01])
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=12)

    # 调整布局
    plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.97])

    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"综合对比图已保存到: {output_path}")

# 主函数
def main():
    # 数据文件路径
    reference_file = Path("output/run_20251223_020748_UTC/datasets/train.jsonl")
    generated_file = Path("output/run_20251223_020748_UTC/visualization/generated_samples.npy")
    assets_dir = Path("output/run_20251223_020748_UTC/assets")
    output_dir = Path("output/run_20251223_020748_UTC/visualization")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载反归一化所需的QuantileTransformer
    print("加载QuantileTransformer对象...")
    qt_up = joblib.load(assets_dir / "qt_up.pkl")
    qt_down = joblib.load(assets_dir / "qt_down.pkl")

    # 加载参考样本的1000个点
    print("加载参考样本的1000个点...")
    reference_points = load_reference_1000_points(reference_file, qt_up, qt_down)
    print(f"参考样本数据点数量: {reference_points.shape[0]}")

    # 加载生成样本的1000个点
    print("加载生成样本的1000个点...")
    generated_points = load_generated_1000_points(generated_file)
    print(f"生成样本数据点数量: {generated_points.shape[0]}")

    # 生成时延对比图
    output_path1 = output_dir / "reference_generated_1000_points_latency.png"
    generate_combined_1000_points(reference_points, generated_points, output_path1)

    # 生成综合对比图
    output_path2 = output_dir / "reference_generated_1000_points_comprehensive.png"
    generate_combined_comprehensive(reference_points, generated_points, output_path2)

    print("\n所有对比图表生成完成!")

if __name__ == "__main__":
    main()

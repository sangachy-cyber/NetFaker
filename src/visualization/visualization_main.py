#!/usr/bin/env python3

"""统一的可视化主脚本, 用于生成各种图表."""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from .generate_1000_points_trend import (
    generate_1000_points_trend,
    generate_comprehensive_trend,
)
from .generate_reference_generated_combined import (
    generate_combined_1000_points,
    generate_combined_comprehensive,
    load_generated_1000_points,
    load_reference_1000_points,
)


class Visualizer:
    """可视化类, 用于生成各种图表."""

    def __init__(self, assets_dir: str, output_dir: str) -> None:
        """初始化可视化器.

        Args:
            assets_dir: 资源文件目录路径
            output_dir: 输出目录路径

        """
        self.assets_dir = Path(assets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载反归一化所需的QuantileTransformer
        self.qt_up = joblib.load(self.assets_dir / "qt_up.pkl")
        self.qt_down = joblib.load(self.assets_dir / "qt_down.pkl")

    def generate_1000_points_trend(self, generated_file: str) -> None:
        """生成1000个点的趋势图.

        Args:
            generated_file: 生成样本文件路径

        """
        generated_samples = np.load(generated_file)

        output_path1 = self.output_dir / "1000_points_latency_trend.png"
        generate_1000_points_trend(generated_samples, output_path1)

        output_path2 = self.output_dir / "1000_points_comprehensive_trend.png"
        generate_comprehensive_trend(generated_samples, output_path2)
        
    def generate_3scenarios_6charts(self, generated_file: str, reference_file: str) -> None:
        """生成3个场景的上下行时延的6张图，每个标题带着行为id.

        Args:
            generated_file: 生成样本文件路径
            reference_file: 参考样本文件路径

        """
        import json
        from src.behavior_discovery.pattern_identifier import LABEL_TO_NAME
        
        generated_samples = np.load(generated_file)
        # 样本形状: (3, 4, 100) - 3类样本，4个特征，100个时间步
        # 特征0: 上行时延（毫秒）
        # 特征1: 下行时延（毫秒）
        # 特征2: 上行丢包率
        # 特征3: 下行丢包率
        
        # 时间步索引（0-99）
        time_steps = np.arange(generated_samples.shape[2])
        
        # 读取参考样本文件，获取行为ID和参考数据
        behavior_names = []
        behavior_ids_list = []
        reference_data = []
        
        # 检查参考样本文件是否是我们保存的reference_samples_info.json或reference_samples_info.npz
        if reference_file.endswith("reference_samples_info.json"):
            # 处理JSON格式
            import json
            # 读取我们保存的参考样本信息
            with open(reference_file, 'r') as f:
                reference_samples_info = json.load(f)
            
            for i, sample_info in enumerate(reference_samples_info[:3]):
                # 获取行为ID
                behavior_label = sample_info.get('behavior_id', 0)
                behavior_id = int(behavior_label)
                behavior_name = LABEL_TO_NAME.get(behavior_id, f"未知行为_{behavior_id}")
                behavior_names.append(behavior_name)
                behavior_ids_list.append(behavior_id)  # 保存真实行为ID
                
                # 提取参考数据
                ref_window_df = sample_info.get('window', {})
                
                # 提取上行和下行时延数据
                ref_up_delay_qt = []
                ref_down_delay_qt = []
                
                # 处理不同类型的window数据
                if isinstance(ref_window_df, dict):
                    # 如果是字典，可能是从JSON序列化的DataFrame
                    # 这种情况需要特殊处理，但目前我们保存的是原始数据，所以暂时不处理
                    continue
                elif isinstance(ref_window_df, list):
                    # 处理列表格式的窗口数据
                    for row in ref_window_df[:100]:
                        if isinstance(row, dict):
                            ref_up_delay_qt.append(row.get('delay_up_qt', 0.0))
                            ref_down_delay_qt.append(row.get('delay_down_qt', 0.0))
                        else:
                            # 如果是其他格式，跳过
                            continue
                elif hasattr(ref_window_df, 'values'):  # 处理DataFrame
                    # 如果是DataFrame，直接提取值
                    ref_up_delay_qt = ref_window_df['delay_up_qt'].values
                    ref_down_delay_qt = ref_window_df['delay_down_qt'].values
                else:
                    # 处理从JSON加载的DataFrame
                    try:
                        # 尝试解析JSON格式的DataFrame
                        df = ref_window_df
                        ref_up_delay_qt = df['delay_up_qt'].values
                        ref_down_delay_qt = df['delay_down_qt'].values
                    except Exception:
                        continue
                
                if len(ref_up_delay_qt) >= 100 and len(ref_down_delay_qt) >= 100:
                    # 将参考数据转换为numpy数组并进行反归一化
                    ref_up_delay_np = np.array(ref_up_delay_qt[:100]).reshape(-1, 1)
                    ref_down_delay_np = np.array(ref_down_delay_qt[:100]).reshape(-1, 1)
                    
                    # 执行反归一化，将QT归一化值转换回原始毫秒值
                    ref_up_delay_denorm = self.qt_up.inverse_transform(ref_up_delay_np).flatten()
                    ref_down_delay_denorm = self.qt_down.inverse_transform(ref_down_delay_np).flatten()
                    
                    reference_data.append({
                        'up_delay': ref_up_delay_denorm,
                        'down_delay': ref_down_delay_denorm
                    })
        elif reference_file.endswith(".npz"):
            # 处理npz格式
            # 读取npz文件
            with np.load(reference_file) as data:
                behavior_ids = data['behavior_ids']
                ref_trajectories = data['ref_trajectories']
            
            for i in range(min(3, len(behavior_ids))):
                # 获取行为ID
                behavior_id = int(behavior_ids[i])
                behavior_name = LABEL_TO_NAME.get(behavior_id, f"未知行为_{behavior_id}")
                behavior_names.append(behavior_name)
                behavior_ids_list.append(behavior_id)  # 保存真实行为ID
                
                # 提取参考轨迹数据
                ref_trajectory = ref_trajectories[i]
                ref_up_delay_qt = ref_trajectory[0][:100]  # 上行时延（QT归一化）
                ref_down_delay_qt = ref_trajectory[1][:100]  # 下行时延（QT归一化）
                
                if len(ref_up_delay_qt) >= 100 and len(ref_down_delay_qt) >= 100:
                    # 将参考数据转换为numpy数组并进行反归一化
                    ref_up_delay_np = np.array(ref_up_delay_qt[:100]).reshape(-1, 1)
                    ref_down_delay_np = np.array(ref_down_delay_qt[:100]).reshape(-1, 1)
                    
                    # 执行反归一化，将QT归一化值转换回原始毫秒值
                    ref_up_delay_denorm = self.qt_up.inverse_transform(ref_up_delay_np).flatten()
                    ref_down_delay_denorm = self.qt_down.inverse_transform(ref_down_delay_np).flatten()
                    
                    reference_data.append({
                        'up_delay': ref_up_delay_denorm,
                        'down_delay': ref_down_delay_denorm
                    })
        else:
            # 处理原始的JSONL格式参考样本文件
            with open(reference_file, 'r') as f:
                # 读取前3个有效样本的行为ID和数据
                count = 0
                for line in f:
                    data = json.loads(line)
                    if data.get('keep', True):
                        # 获取行为标签
                        behavior_label = data.get('behavior_label', 0)
                        behavior_name = LABEL_TO_NAME.get(behavior_label, f"未知行为_{behavior_label}")
                        behavior_names.append(behavior_name)
                        behavior_ids_list.append(behavior_label)  # 保存真实行为ID
                        
                        # 获取参考数据（前100个点）
                        window_data = data.get('window', [])[:100]
                        if len(window_data) >= 100:
                            # 提取上行和下行时延数据
                            # window_data中的每行是字典，包含delay_up和delay_down等键
                            ref_up_delay_qt = []
                            ref_down_delay_qt = []
                            for row in window_data:
                                if isinstance(row, dict):
                                    # 从字典中提取归一化后的时延数据
                                    # 注意：在预处理阶段，delay_up和delay_down字段已被替换为QT归一化后的值
                                    ref_up_delay_qt.append(row.get('delay_up', 0))  # 上行时延（QT归一化后）
                                    ref_down_delay_qt.append(row.get('delay_down', 0))  # 下行时延（QT归一化后）
                                else:
                                    # 如果是列表或元组，使用索引
                                    ref_up_delay_qt.append(row[1] if len(row) > 1 else 0)  # 上行时延
                                    ref_down_delay_qt.append(row[2] if len(row) > 2 else 0)  # 下行时延
                            
                            # 将参考数据转换为numpy数组并进行反归一化
                            ref_up_delay_np = np.array(ref_up_delay_qt).reshape(-1, 1)
                            ref_down_delay_np = np.array(ref_down_delay_qt).reshape(-1, 1)
                            
                            # 执行反归一化，将QT归一化值转换回原始毫秒值
                            ref_up_delay_denorm = self.qt_up.inverse_transform(ref_up_delay_np).flatten()
                            ref_down_delay_denorm = self.qt_down.inverse_transform(ref_down_delay_np).flatten()
                            
                            reference_data.append({
                                'up_delay': ref_up_delay_denorm,
                                'down_delay': ref_down_delay_denorm
                            })
                        
                        count += 1
                        if count >= 3:
                            break
        
        # 如果参考样本中没有足够的行为ID，使用默认的
        if len(behavior_names) < 3:
            # 从pattern_identifier.py中获取的真实行为名称
            default_behavior_names = [
                "STABLE", "WEAK_BURST", "FREQUENT_FLUCTUATION", 
                "HIGH_DELAY_NO_LOSS", "HIGH_LOSS_STEADY", "LOW_DELAY_HIGH_LOSS",
                "STRONG_BURST", "INSTANT_SPIKE"
            ]
            behavior_names = default_behavior_names[:3]
            # 使用默认行为ID
            behavior_ids_list = list(range(len(behavior_names)))  # 默认行为ID为0,1,2
        
        # 为每个场景生成上下行时延图
        for i in range(generated_samples.shape[0]):
            # 获取当前场景数据
            scenario_data = generated_samples[i]
            
            # 上行时延数据
            gen_up_delay_ms = scenario_data[0, :]
            # 下行时延数据
            gen_down_delay_ms = scenario_data[1, :]
            
            # 获取真实行为ID和名称
            real_behavior_id = behavior_ids_list[i] if i < len(behavior_ids_list) else i
            behavior_name = behavior_names[i] if i < len(behavior_names) else f"未知行为_{i}"
            
            # 生成上行时延图 - 包含参考样本和生成样本对比
            plt.figure(figsize=(15, 8))
            
            # 如果有对应的参考数据，绘制参考样本
            if i < len(reference_data):
                ref_up_delay = reference_data[i]['up_delay']
                plt.plot(time_steps, ref_up_delay, color="orange", linewidth=2, alpha=0.8, label="参考样本")
            
            # 绘制生成样本
            plt.plot(time_steps, gen_up_delay_ms, color="blue", linewidth=2, alpha=0.8, label="生成样本")
            
            plt.title(f"行为ID {real_behavior_id} - {behavior_name} - 上行时延趋势对比（100个点，时延单位：毫秒）", 
                     fontsize=18, fontweight="bold")
            plt.xlabel("时间步（0-99）", fontsize=14)
            plt.ylabel("上行时延 (毫秒)", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.axhline(y=20, color="red", linestyle="--", linewidth=1.5, label="20ms阈值")
            plt.legend(fontsize=12)
            plt.tight_layout()
            output_path = self.output_dir / f"behavior_{real_behavior_id}_{behavior_name}_up_latency_comparison.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"行为ID {real_behavior_id}上行时延对比图已保存到: {output_path}")
            
            # 生成下行时延图 - 包含参考样本和生成样本对比
            plt.figure(figsize=(15, 8))
            
            # 如果有对应的参考数据，绘制参考样本
            if i < len(reference_data):
                ref_down_delay = reference_data[i]['down_delay']
                plt.plot(time_steps, ref_down_delay, color="orange", linewidth=2, alpha=0.8, label="参考样本")
            
            # 绘制生成样本
            plt.plot(time_steps, gen_down_delay_ms, color="green", linewidth=2, alpha=0.8, label="生成样本")
            
            plt.title(f"行为ID {real_behavior_id} - {behavior_name} - 下行时延趋势对比（100个点，时延单位：毫秒）", 
                     fontsize=18, fontweight="bold")
            plt.xlabel("时间步（0-99）", fontsize=14)
            plt.ylabel("下行时延 (毫秒)", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.axhline(y=20, color="red", linestyle="--", linewidth=1.5, label="20ms阈值")
            plt.legend(fontsize=12)
            plt.tight_layout()
            output_path = self.output_dir / f"behavior_{real_behavior_id}_{behavior_name}_down_latency_comparison.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"行为ID {real_behavior_id}下行时延对比图已保存到: {output_path}")

    def generate_reference_generated_comparison(
        self,
        reference_file: str,
        generated_file: str,
    ) -> None:
        """生成参考样本和生成样本的对比图.

        Args:
            reference_file: 参考样本文件路径
            generated_file: 生成样本文件路径

        """
        # 加载参考样本的1000个点
        reference_points = load_reference_1000_points(
            reference_file,
            self.qt_up,
            self.qt_down,
        )

        # 加载生成样本的1000个点
        generated_points = load_generated_1000_points(generated_file)

        # 生成时延对比图
        output_path1 = self.output_dir / "reference_generated_1000_points_latency.png"
        generate_combined_1000_points(
            reference_points,
            generated_points,
            output_path1,
        )

        # 生成综合对比图
        output_path2 = self.output_dir / (
            "reference_generated_1000_points_comprehensive.png"
        )
        generate_combined_comprehensive(
            reference_points,
            generated_points,
            output_path2,
        )

    def generate_all_charts(self, reference_file: str, generated_file: str) -> None:
        """生成所有图表.

        Args:
            reference_file: 参考样本文件路径
            generated_file: 生成样本文件路径

        """
        # 只生成3个场景的对比图，移除1000个点和100个点图片的生成
        self.generate_3scenarios_6charts(generated_file, reference_file)

# 命令行入口

# 将命令行逻辑封装到main函数中
def main() -> None:
    """命令行入口函数, 用于处理命令行参数并执行可视化操作."""
    parser = argparse.ArgumentParser(description="生成可视化图表")
    parser.add_argument(
        "--reference-file",
        type=str,
        default="./output/datasets/test.jsonl",
        help="参考样本文件路径",
    )
    parser.add_argument(
        "--generated-file",
        type=str,
        default="./output/visualization/generated_samples.npy",
        help="生成样本文件路径",
    )
    parser.add_argument(
        "--assets-dir",
        type=str,
        default="./output/assets",
        help="资源文件目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/visualization",
        help="输出目录",
    )
    parser.add_argument("--all", action="store_true", help="生成所有图表")
    parser.add_argument(
        "--scenarios",
        action="store_true",
        help="生成3个场景的上下行时延的6张图，每个标题带着行为id",
    )

    args = parser.parse_args()

    visualizer = Visualizer(args.assets_dir, args.output_dir)

    if args.all or args.scenarios:
        visualizer.generate_all_charts(args.reference_file, args.generated_file)
    else:
        # 默认只生成3个场景的对比图
        visualizer.generate_all_charts(args.reference_file, args.generated_file)

# 命令行入口
if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一的可视化主脚本，用于生成各种图表
"""

import numpy as np
import joblib
from pathlib import Path
from .generate_1000_points_trend import generate_1000_points_trend, generate_comprehensive_trend
from .generate_reference_generated_combined import generate_combined_1000_points, generate_combined_comprehensive, load_reference_1000_points, load_generated_1000_points

class Visualizer:
    """可视化类，用于生成各种图表"""
    
    def __init__(self, assets_dir, output_dir):
        """初始化可视化器"""
        self.assets_dir = Path(assets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载反归一化所需的QuantileTransformer
        self.qt_up = joblib.load(self.assets_dir / "qt_up.pkl")
        self.qt_down = joblib.load(self.assets_dir / "qt_down.pkl")
    
    def generate_1000_points_trend(self, generated_file):
        """生成1000个点的趋势图"""
        generated_samples = np.load(generated_file)
        
        output_path1 = self.output_dir / "1000_points_latency_trend.png"
        generate_1000_points_trend(generated_samples, output_path1)
        
        output_path2 = self.output_dir / "1000_points_comprehensive_trend.png"
        generate_comprehensive_trend(generated_samples, output_path2)
    
    def generate_reference_generated_comparison(self, reference_file, generated_file):
        """生成参考样本和生成样本的对比图"""
        # 加载参考样本的1000个点
        reference_points = load_reference_1000_points(reference_file, self.qt_up, self.qt_down)
        
        # 加载生成样本的1000个点
        generated_points = load_generated_1000_points(generated_file)
        
        # 生成时延对比图
        output_path1 = self.output_dir / "reference_generated_1000_points_latency.png"
        generate_combined_1000_points(reference_points, generated_points, output_path1)
        
        # 生成综合对比图
        output_path2 = self.output_dir / "reference_generated_1000_points_comprehensive.png"
        generate_combined_comprehensive(reference_points, generated_points, output_path2)
    
    def generate_all_charts(self, reference_file, generated_file):
        """生成所有图表"""
        print("生成1000个点的趋势图...")
        self.generate_1000_points_trend(generated_file)
        
        print("\n生成参考样本和生成样本的对比图...")
        self.generate_reference_generated_comparison(reference_file, generated_file)
        
        print("\n所有图表生成完成!")

# 命令行入口

# 将命令行逻辑封装到main函数中
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="生成可视化图表")
    parser.add_argument("--reference-file", type=str, default="output/run_20251223_020748_UTC/datasets/train.jsonl", help="参考样本文件路径")
    parser.add_argument("--generated-file", type=str, default="output/run_20251223_020748_UTC/visualization/generated_samples.npy", help="生成样本文件路径")
    parser.add_argument("--assets-dir", type=str, default="output/run_20251223_020748_UTC/assets", help="资源文件目录")
    parser.add_argument("--output-dir", type=str, default="output/run_20251223_020748_UTC/visualization", help="输出目录")
    parser.add_argument("--all", action="store_true", help="生成所有图表")
    parser.add_argument("--trend", action="store_true", help="生成1000个点的趋势图")
    parser.add_argument("--comparison", action="store_true", help="生成参考样本和生成样本的对比图")
    
    args = parser.parse_args()
    
    visualizer = Visualizer(args.assets_dir, args.output_dir)
    
    if args.all:
        visualizer.generate_all_charts(args.reference_file, args.generated_file)
    elif args.trend:
        visualizer.generate_1000_points_trend(args.generated_file)
    elif args.comparison:
        visualizer.generate_reference_generated_comparison(args.reference_file, args.generated_file)
    else:
        # 默认生成所有图表
        visualizer.generate_all_charts(args.reference_file, args.generated_file)

# 命令行入口
if __name__ == "__main__":
    main()

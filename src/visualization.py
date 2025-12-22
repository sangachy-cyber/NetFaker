#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据可视化模块，用于分析预处理后的数据分布
"""

import platform
import matplotlib
import matplotlib.font_manager as fm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List, Any
import warnings

# 忽略绘图警告
warnings.filterwarnings("ignore")


def setup_chinese_font():
    """
    根据操作系统设置中文字体
    Mac: 使用系统默认中文字体
    Windows: 使用微软雅黑
    Linux: 使用文泉驿正黑
    """
    system = platform.system()
    
    if system == "Darwin":  # macOS
        # macOS使用系统中文字体
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']
    elif system == "Windows":
        # Windows使用微软雅黑
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    elif system == "Linux":
        # Linux使用文泉驿正黑
        try:
            # 检查是否存在文泉驿正黑字体
            font_names = [f.name for f in fm.fontManager.ttflist]
            if any('wqy' in name.lower() for name in font_names):
                plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei']
            else:
                # 如果没有文泉驿字体，使用系统默认
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        except:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False

# 设置中文字体
setup_chinese_font()


def load_processed_data(data_path: Path) -> Dict[str, List[Any]]:
    """
    加载预处理后的数据
    
    Args:
        data_path: 数据路径
        
    Returns:
        包含训练、验证、测试数据的字典
    """
    import json
    
    datasets = {}
    for split in ['train', 'val', 'test']:
        file_path = data_path / f"{split}.jsonl"
        if file_path.exists():
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            datasets[split] = data
        else:
            datasets[split] = []
            
    return datasets


def extract_features(datasets: Dict[str, List[Any]]) -> Dict[str, np.ndarray]:
    """
    从数据集中提取特征用于可视化
    
    Args:
        datasets: 数据集字典
        
    Returns:
        特征字典
    """
    features = {}
    
    for split_name, data in datasets.items():
        if not data:
            continue
            
        # 提取条件向量特征
        cond_vectors = []
        window_data = []
        
        for item in data:
            if 'cond' in item:
                cond_vectors.append(item['cond'])
            if 'window' in item:
                # window是一个列表，需要展开
                window_data.extend(item['window'])
        
        if cond_vectors:
            features[f'{split_name}_cond'] = np.array(cond_vectors)
            
        if window_data:
            df = pd.DataFrame(window_data)
            features[f'{split_name}_window'] = df
            
    return features


def plot_histograms(features: Dict[str, np.ndarray], output_dir: Path):
    """
    绘制直方图对比原始数据、归一化后数据与标准正态分布
    
    Args:
        features: 特征字典
        output_dir: 输出目录
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘制条件向量的直方图
    for key, data in features.items():
        if 'cond' in key and len(data.shape) == 2:
            n_features = min(data.shape[1], 6)  # 最多绘制6个特征
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'{key} - 条件向量特征分布直方图', fontsize=16)
            
            for i in range(n_features):
                row, col = i // 3, i % 3
                ax = axes[row, col]
                
                # 绘制实际数据分布
                feature_data = data[:, i]
                ax.hist(feature_data, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black', linewidth=0.5)
                
                # 绘制标准正态分布对比
                x = np.linspace(feature_data.min(), feature_data.max(), 100)
                standard_normal = stats.norm.pdf(x, 0, 1)
                ax.plot(x, standard_normal, 'r-', linewidth=2, label='标准正态分布N(0,1)')
                
                ax.set_title(f'特征 {i}')
                ax.set_xlabel('值')
                ax.set_ylabel('密度')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 隐藏多余的子图
            for i in range(n_features, 6):
                row, col = i // 3, i % 3
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{key}_histogram.png', dpi=300, bbox_inches='tight')
            plt.close()


def plot_qq_plots(features: Dict[str, np.ndarray], output_dir: Path):
    """
    绘制Q-Q图评估数据分布与正态分布的吻合程度
    
    Args:
        features: 特征字典
        output_dir: 输出目录
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘制条件向量的Q-Q图
    for key, data in features.items():
        if 'cond' in key and len(data.shape) == 2:
            n_features = min(data.shape[1], 6)  # 最多绘制6个特征
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'{key} - 条件向量特征 Q-Q 图', fontsize=16)
            
            for i in range(n_features):
                row, col = i // 3, i % 3
                ax = axes[row, col]
                
                # 绘制Q-Q图
                feature_data = data[:, i]
                stats.probplot(feature_data, dist="norm", plot=ax)
                ax.set_title(f'特征 {i}')
                ax.grid(True, alpha=0.3)
                
                # 添加R²值
                # 计算理论分位数和实际分位数的相关性
                theoretical_quantiles = np.sort(stats.norm.ppf(np.linspace(0.01, 0.99, len(feature_data))))
                sample_quantiles = np.sort(feature_data)
                # 只取相同长度的部分
                min_len = min(len(theoretical_quantiles), len(sample_quantiles))
                r_squared = np.corrcoef(theoretical_quantiles[:min_len], sample_quantiles[:min_len])[0, 1]**2
                ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 隐藏多余的子图
            for i in range(n_features, 6):
                row, col = i // 3, i % 3
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{key}_qq_plot.png', dpi=300, bbox_inches='tight')
            plt.close()


def plot_window_features(features: Dict[str, pd.DataFrame], output_dir: Path):
    """
    绘制窗口数据特征的分布图
    
    Args:
        features: 特征字典
        output_dir: 输出目录
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘制窗口数据的特征分布
    for key, df in features.items():
        if 'window' in key and isinstance(df, pd.DataFrame):
            # 选择数值型列
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            n_features = min(len(numeric_columns), 6)  # 最多绘制6个特征
            
            if n_features > 0:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                fig.suptitle(f'{key} - 窗口数据特征分布', fontsize=16)
                
                for i in range(n_features):
                    row, col = i // 3, i % 3
                    ax = axes[row, col]
                    
                    column = numeric_columns[i]
                    data = df[column].dropna()
                    
                    # 绘制直方图
                    ax.hist(data, bins=50, alpha=0.7, color='lightgreen', edgecolor='black', linewidth=0.5)
                    ax.set_title(f'{column}')
                    ax.set_xlabel('值')
                    ax.set_ylabel('频次')
                    ax.grid(True, alpha=0.3)
                
                # 隐藏多余的子图
                for i in range(n_features, 6):
                    row, col = i // 3, i % 3
                    axes[row, col].set_visible(False)
                
                plt.tight_layout()
                plt.savefig(output_dir / f'{key}_features.png', dpi=300, bbox_inches='tight')
                plt.close()


def generate_visualizations(data_path: Path, output_dir: Path = None):
    """
    生成完整的数据可视化报告
    
    Example:
        >>> generate_visualizations(Path("output/run_20251221_161125_UTC/datasets"), 
        ...                        Path("output/run_20251221_161125_UTC/visualizations"))
    
    Args:
        data_path: 数据集路径
        output_dir: 输出目录，如果为None则使用data_path/../visualizations
    """
    if output_dir is None:
        output_dir = data_path.parent / "visualizations"
    
    print(f"正在从 {data_path} 加载数据...")
    datasets = load_processed_data(data_path)
    
    print("正在提取特征...")
    features = extract_features(datasets)
    
    if not features:
        print("警告: 未找到可用于可视化的特征数据")
        return
    
    print("正在生成直方图...")
    plot_histograms(features, output_dir)
    
    print("正在生成Q-Q图...")
    plot_qq_plots(features, output_dir)
    
    print("正在生成窗口特征图...")
    plot_window_features(features, output_dir)
    
    print(f"可视化结果已保存到: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成数据可视化报告")
    parser.add_argument("--data_path", type=str, required=True, help="数据集路径")
    parser.add_argument("--output_dir", type=str, help="输出目录")
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    generate_visualizations(data_path, output_dir)
# preprocess.py (v1.3)
import os
import sys

# 添加项目根目录到Python搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import yaml
import matplotlib.pyplot as plt

from src.preprocessing import PreprocessingPipeline
from src.visualization.behavior_visualization import select_behavior_typical_cases, generate_behavior_typical_cases


def main():
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="数据预处理脚本")
    args = parser.parse_args()
    
    # 加载配置文件
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    # 使用配置文件中指定的固定输出目录
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建预处理流水线实例
    pipeline = PreprocessingPipeline(cfg)
    
    # 执行预处理流程
    result = pipeline.run()
    
    # 保存窗口数据用于调试
    import pandas as pd
    import os
    
    # 创建assets目录
    assets_dir = Path("assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有窗口数据
    all_windows = []
    for dataset in result['datasets'].values():
        all_windows.extend(dataset)
    
    # 保存窗口数据
    if all_windows:
        windows_df = pd.DataFrame(all_windows)
        windows_df.to_pickle(assets_dir / "windows_meta.pkl")
        print(f"\n调试信息: 已保存 {len(all_windows)} 个窗口数据到 {assets_dir / 'windows_meta.pkl'}")
    
    # 打印结果统计
    stats = result["stats"]
    assets = result["assets"]
    
    print("\n=== 预处理完成 ===")
    print(f"总样本数: {stats['total_windows']}")
    print(f"训练集: {stats['train_count']}个样本")
    print(f"验证集: {stats['val_count']}个样本")
    print(f"测试集: {stats['test_count']}个样本")
    print(f"处理文件总数: {stats['processed_files']}")
    print(f"失败文件数: {stats['failed_files']}")
    print(f"空文件数: {stats['empty_files']}")
    
    # 打印网络状态ID分布
    if "state_id_stats" in assets:
        print("\n=== 网络状态ID分布统计 ===")
        for dataset_name, stats_item in assets["state_id_stats"].items():
            print(f"\n{dataset_name.upper()}集:")
            total_samples = sum(stat["count"] for stat in stats_item.values())
            print(f"  总样本数: {total_samples}")
            print("  状态ID分布:")
            for state_id in sorted(stats_item.keys()):
                stat = stats_item[state_id]
                print(f"    状态ID {state_id}: {stat['count']}个样本 ({stat['percentage']:.2f}%)")
    
    # 打印行为发现统计
    if "behavior_stats" in assets:
        print("\n=== 行为发现统计 ===")
        behavior_stats = assets["behavior_stats"]
        total_samples = sum(stat["count"] for stat in behavior_stats.values())
        print(f"总样本数: {total_samples}")
        print("行为类型分布:")
        for behavior_name in sorted(behavior_stats.keys()):
            stat = behavior_stats[behavior_name]
            print(f"  {behavior_name}: {stat['count']}个样本 ({stat['percentage']:.2f}%)")
    
    # 生成行为典型案例可视化
    print("\n=== 生成行为典型案例可视化 ===")
    
    # 收集所有窗口数据和行为标签
    all_windows = []
    behavior_labels = []
    
    for dataset_type, windows in result['datasets'].items():
        for window_meta in windows:
            # 只处理被保留的窗口
            if window_meta['keep']:
                all_windows.append(window_meta['window'])
                # 行为标签位于条件向量的第11维（索引11）
                behavior_label = int(window_meta['cond'][11])
                behavior_labels.append(behavior_label)
    
    # 选择典型案例
    typical_cases = select_behavior_typical_cases(all_windows, behavior_labels, num_cases=3)
    
    # 生成可视化图表
    if typical_cases:
        # 创建可视化输出目录（使用已定义的out_dir变量）
        viz_dir = out_dir / "visualization"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成图表
        output_path = viz_dir / "behavior_typical_cases.png"
        print(f"正在生成行为典型案例图表...")
        fig = generate_behavior_typical_cases(typical_cases, output_path=output_path)
        plt.close(fig)  # 关闭图表，释放资源
        print(f"行为典型案例图表已保存到: {output_path}")
    else:
        print("没有找到有效的行为案例用于可视化")


if __name__ == "__main__":
    main()
# preprocess.py (v1.3)
import os
import sys

# 添加项目根目录到Python搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import yaml

from src.preprocessing import PreprocessingPipeline


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


if __name__ == "__main__":
    main()
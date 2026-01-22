#!/usr/bin/env python3
"""HoloWAN 状态序列提取工具。

从真实HoloWAN仿真文件中自动提取结构化的状态序列（segments），
用于回放、分析或再生成。

Usage:
    python scripts/extract_segments.py --input data/raw/real_trace.txt --output data/outputs/segments.json
"""

import argparse
import json
import os
import sys

from loguru import logger

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# 导入核心分析器
from netfaker.simcore.trace_analyzer import TraceAnalyzer


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        解析后的命令行参数
    """
    parser = argparse.ArgumentParser(description="HoloWAN状态序列提取工具")
    parser.add_argument("--input", required=True, help="输入HoloWAN文件路径")
    parser.add_argument("--output", required=True, help="输出segments文件路径")
    parser.add_argument("--model-path", default="data/clusters/gmm_model.joblib",
                      help="聚类模型路径")
    parser.add_argument("--scaler-path", default="data/clusters/feature_scaler.joblib",
                      help="特征缩放器路径")
    parser.add_argument("--metadata-path", default="data/clusters/state_metadata.json",
                      help="状态元数据路径")
    return parser.parse_args()


def main() -> None:
    """主函数。
    """
    # 配置日志
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")

    # 解析命令行参数
    args = parse_args()

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        # 初始化状态序列提取器
        analyzer = TraceAnalyzer(
            model_path=args.model_path,
            scaler_path=args.scaler_path,
            metadata_path=args.metadata_path
        )

        # 分析文件并生成segments
        logger.info(f"开始分析文件: {args.input}")
        segments = analyzer.analyze(args.input)

        # 保存结果
        logger.info(f"保存结果到: {args.output}")
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        logger.success("状态序列提取完成！")
        logger.success(f"输入文件: {args.input}")
        logger.success(f"输出文件: {args.output}")
        logger.success(f"生成segments数量: {len(segments)}")

    except Exception as e:
        logger.exception(f"处理失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

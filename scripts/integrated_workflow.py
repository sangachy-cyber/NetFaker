#!/usr/bin/env python3
"""NetFaker 完整工作流整合脚本。

实现从真实文件提取segments → 生成合成文件 → 评估合成文件的完整工作流。

Usage:
    python scripts/integrated_workflow.py --real data/raw/real_trace.txt --output data/outputs
"""

import argparse
import json
import os
import sys
from datetime import datetime

from loguru import logger

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# 导入核心模块
from netfaker.simcore.trace_analyzer import TraceAnalyzer
from netfaker.simcore.synthesizer import RuleSynthesizer
from netfaker.evaluator import HoloWANEvaluator


def parse_args() -> argparse.Namespace:
    """解析命令行参数。
    
    Returns:
        解析后的命令行参数
    """
    parser = argparse.ArgumentParser(description="NetFaker 完整工作流整合脚本")
    parser.add_argument("--real", default="data/raw/20260118_231050_VAj-playback.txt", help="真实HoloWAN文件路径")
    parser.add_argument("--output", default="data/outputs", help="输出目录路径")
    parser.add_argument("--window-pool", default="data/clusters/train_with_state.parquet",
                      help="已标注窗口数据的Parquet文件路径")
    parser.add_argument("--model-path", default="data/clusters/gmm_model.joblib",
                      help="聚类模型路径")
    parser.add_argument("--scaler-path", default="data/clusters/feature_scaler.joblib",
                      help="特征缩放器路径")
    parser.add_argument("--metadata-path", default="data/clusters/state_metadata.json",
                      help="状态元数据路径")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--verbose", action="store_true", help="启用详细日志")
    return parser.parse_args()


def main() -> None:
    """主函数。
    """
    # 配置日志
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    logger.info("=== NetFaker 完整工作流开始 ===")
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    
    # 生成时间戳用于区分不同运行
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # 步骤1：提取segments
        logger.info("\n1. 开始从真实文件提取segments...")
        segments_file = os.path.join(args.output, f"segments_{timestamp}.json")
        
        analyzer = TraceAnalyzer(
            model_path=args.model_path,
            scaler_path=args.scaler_path,
            metadata_path=args.metadata_path
        )
        
        segments = analyzer.analyze(args.real)
        
        with open(segments_file, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        
        logger.success(f"✅ segments提取完成：{segments_file}")
        logger.info(f"   生成segments数量：{len(segments)}")
        
        # 步骤2：基于segments生成合成文件
        logger.info("\n2. 开始基于segments生成合成文件...")
        syn_file = os.path.join(args.output, f"syn_{timestamp}.txt")
        
        synthesizer = RuleSynthesizer(args.window_pool)
        synthesizer.generate(segments, syn_file, args.seed)
        
        logger.success(f"✅ 合成文件生成完成：{syn_file}")
        
        # 步骤3：评估合成文件
        logger.info("\n3. 开始评估合成文件...")
        eval_output_dir = os.path.join(args.output, f"eval_{timestamp}")
        
        evaluator = HoloWANEvaluator()
        eval_result = evaluator.evaluate(args.real, syn_file, eval_output_dir)
        
        logger.success(f"✅ 评估完成，报告已生成：{eval_output_dir}")
        
        # 输出评估摘要
        logger.info("\n4. 评估结果摘要：")
        logger.info(f"   总窗口数：{eval_result['total_windows']}")
        logger.info("   通过率：")
        for metric, pass_rate in eval_result["pass_rates"].items():
            logger.info(f"     {metric}: {pass_rate['pass_count']}/{pass_rate['total_count']} ({pass_rate['rate']:.1%})")
        
        # 计算整体通过率
        overall_pass_rate = sum(pr['rate'] for pr in eval_result["pass_rates"].values()) / len(eval_result["pass_rates"])
        logger.info(f"   整体通过率：{overall_pass_rate:.1%}")
        
        logger.info("\n=== NetFaker 完整工作流结束 ===")
        
    except Exception as e:
        logger.exception(f"❌ 工作流执行失败：{e}")
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    main()
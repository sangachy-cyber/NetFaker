#!/usr/bin/env python3
"""NetFaker 仿真参数评估 CLI 脚本。

用于评估合成 HoloWAN 文件与真实文件的相似度，
输出详细指标、汇总报告和可视化图表。
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

from netfaker.evaluator import HoloWANEvaluator


def parse_args() -> argparse.Namespace:
    """解析命令行参数。
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="NetFaker 仿真参数评估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例用法：
  python scripts/evaluate_simulation.py \
    --real data/raw/campus_real.txt \
    --syn data/outputs/sim_rule_task123.txt \
    --output data/outputs/eval/
        """
    )

    parser.add_argument(
        "--real",
        type=str,
        required=True,
        help="真实 HoloWAN 文件路径"
    )

    parser.add_argument(
        "--syn",
        type=str,
        required=True,
        help="合成 HoloWAN 文件路径"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/outputs/eval/",
        help="输出目录路径（默认：data/outputs/eval/）"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="评估配置文件路径（默认：使用内置配置）"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="启用详细日志"
    )

    return parser.parse_args()


def main() -> int:
    """主函数。
    
    Returns:
        int: 退出码，0表示成功，非0表示失败
    """
    # 解析命令行参数
    args = parse_args()

    # 配置日志
    logger.remove()  # 移除默认处理器
    logger.add(
        sys.stderr,
        level="DEBUG" if args.verbose else "INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    try:
        # 验证文件路径
        real_path = Path(args.real)
        syn_path = Path(args.syn)

        if not real_path.exists():
            logger.error(f"真实文件不存在：{real_path}")
            return 1

        if not syn_path.exists():
            logger.error(f"合成文件不存在：{syn_path}")
            return 1

        # 初始化评估器
        evaluator = HoloWANEvaluator(args.config)

        # 执行评估
        report = evaluator.evaluate(
            real_file=str(real_path),
            syn_file=str(syn_path),
            output_dir=args.output
        )

        # 输出结果摘要
        logger.info("\n=== 评估结果摘要 ===")
        logger.info(f"总窗口数: {report['total_windows']}")

        logger.info("\n指标通过率:")
        for metric, pass_rate in report['pass_rates'].items():
            logger.info(f"  {metric}: {pass_rate['pass_count']}/{pass_rate['total_count']} ({pass_rate['rate']:.1%})")

        # 计算整体通过率
        overall_pass_rate = sum(pr['rate'] for pr in report['pass_rates'].values()) / len(report['pass_rates'])
        logger.info(f"\n整体通过率: {overall_pass_rate:.1%}")

        return 0
    except Exception as e:
        logger.error(f"评估失败：{e}")
        if args.verbose:
            logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())

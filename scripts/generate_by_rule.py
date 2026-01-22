#!/usr/bin/env python3
"""基于规则的状态序列生成入口脚本。

用于根据用户提供的状态序列模板，生成仿真流量文件。
"""

import argparse
import json
from typing import Any, Dict, List

from loguru import logger

from netfaker.simcore.synthesizer import RuleSynthesizer


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 解析后的命令行参数
    """
    parser = argparse.ArgumentParser(
        description="基于规则的状态序列生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        示例用法：
            python scripts/generate_by_rule.py \
                --template '[{"type": "s0", "duration": 10}, {"type": "s3", "duration": 20}]' \
                --output output/rule_seq_01.txt
                --window-pool data/clusters/train_with_state.parquet
                --seed 42
        """
    )

    parser.add_argument(
        "--template",
        type=str,
        required=True,
        help='状态序列模板，JSON格式的列表，例如："[{\"type\": \"s0\", \"duration\": 10}]"'
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出文件路径"
    )

    parser.add_argument(
        "--window-pool",
        type=str,
        default="data/clusters/train_with_state.parquet",
        help="已标注窗口数据的Parquet文件路径（默认：data/clusters/train_with_state.parquet）"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，用于控制抽样的可复现性（默认：None）"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="启用详细日志输出"
    )

    return parser.parse_args()


def main() -> None:
    """主函数。
    """
    # 解析命令行参数
    args = parse_args()

    # 配置日志
    if args.verbose:
        logger.remove()
        logger.add(
            sink=lambda msg: print(msg, end=""),
            level="DEBUG",
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
    else:
        logger.remove()
        logger.add(
            sink=lambda msg: print(msg, end=""),
            level="INFO",
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
        )

    logger.info("基于规则的状态序列生成工具")
    logger.info(f"输出路径: {args.output}")
    logger.info(f"窗口数据路径: {args.window_pool}")
    logger.info(f"随机种子: {args.seed}")

    try:
        # 解析模板JSON
        template: List[Dict[str, Any]] = json.loads(args.template)
        logger.info(f"解析模板: {template}")

        # 创建RuleSynthesizer实例
        synthesizer = RuleSynthesizer(args.window_pool)

        # 生成仿真流量文件
        output_file = synthesizer.generate(template, args.output, args.seed)

        logger.success(f"仿真流量文件生成成功: {output_file}")

    except json.JSONDecodeError as e:
        logger.error(f"JSON模板解析失败: {str(e)}")
        exit(1)
    except ValueError as e:
        logger.error(f"参数验证失败: {str(e)}")
        exit(1)
    except FileNotFoundError as e:
        logger.error(f"文件不存在: {str(e)}")
        exit(1)
    except Exception as e:
        logger.error(f"生成过程失败: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    main()

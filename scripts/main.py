"""NetFaker 命令行入口脚本。

提供命令行界面，用于生成网络仿真参数。
"""

import argparse
import json

from netfaker.core.logging import logger
from netfaker.simcore.generator import (
    generate_simulation_params,
    list_available_strategies,
)


def main():
    """脚本主函数。"""
    parser = argparse.ArgumentParser(description="NetFaker 网络仿真参数生成工具")

    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # generate 命令
    generate_parser = subparsers.add_parser("generate", help="生成仿真参数")
    generate_parser.add_argument(
        "--strategy",
        required=True,
        help="使用的仿真策略名称",
    )
    generate_parser.add_argument(
        "--params",
        type=str,
        help="仿真参数（JSON格式）",
    )
    generate_parser.add_argument(
        "--scenario",
        type=str,
        help="预定义场景名称（如：video_streaming, online_gaming）",
    )

    # list 命令
    subparsers.add_parser("list", help="列出所有可用策略")

    # 解析参数
    args = parser.parse_args()

    try:
        if args.command == "generate":
            # 处理生成命令
            params = {}

            # 如果提供了params参数，解析JSON
            if args.params:
                params = json.loads(args.params)

            # 如果提供了scenario参数，添加到params
            if args.scenario:
                params["scenario"] = args.scenario

            # 生成仿真参数
            result = generate_simulation_params(args.strategy, params)

            # 输出结果
            print(json.dumps(result, indent=2, ensure_ascii=False))

        elif args.command == "list":
            # 处理列出策略命令
            strategies = list_available_strategies()
            print("可用策略列表：")
            for name, info in strategies.items():
                print(f"- {name}: {info['description']}")
        else:
            parser.print_help()

    except Exception as e:
        logger.error(f"命令执行失败: {e}")
        print(f"错误: {e}")
        exit(1)


if __name__ == "__main__":
    main()

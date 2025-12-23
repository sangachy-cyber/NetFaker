#!/usr/bin/env python3

"""可视化脚本，执行可视化模块的命令行入口
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 直接执行可视化模块的命令行入口
from src.visualization import visualization_main

if __name__ == "__main__":
    # 调用可视化模块的命令行主函数
    visualization_main.main()


"""仿真核心工具模块初始化文件。

提供网络仿真核心功能相关的工具函数和辅助类，包括文件处理、数据转换等。
"""

import os
import platform
from typing import Dict, Optional


def setup_matplotlib_font() -> Dict[str, str]:
    """设置Matplotlib字体，确保在不同操作系统上正常显示中文。

    该函数会根据操作系统自动选择合适的字体配置，解决中文显示问题。

    Returns:
        Dict[str, str]: 当前使用的字体配置
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # 获取当前操作系统
    system = platform.system()
    font_config = {}

    try:
        # 根据操作系统选择字体
        if system == "Darwin":  # macOS
            # macOS 系统使用系统内置的中文字体
            mpl.rcParams["font.family"] = "Arial Unicode MS"
            font_config["font.family"] = "Arial Unicode MS"
            font_config["system"] = "macOS"
        elif system == "Windows":  # Windows
            # Windows 系统使用系统内置的中文字体
            mpl.rcParams["font.family"] = "SimHei"
            font_config["font.family"] = "SimHei"
            font_config["system"] = "Windows"
        else:  # Linux 及其他系统
            # Linux 系统尝试使用常用的中文字体
            # 优先查找 Noto Sans SC、WenQuanYi Micro Hei、Microsoft YaHei 等字体
            linux_fonts = [
                "Noto Sans SC",
                "WenQuanYi Micro Hei",
                "Microsoft YaHei",
                "DejaVu Sans"
            ]

            # 检查字体是否可用
            available_fonts = mpl.font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
            selected_font = None

            for font in linux_fonts:
                for f_path in available_fonts:
                    if font in f_path:
                        selected_font = font
                        break
                if selected_font:
                    break

            # 如果找到合适的字体，设置它
            if selected_font:
                mpl.rcParams["font.family"] = selected_font
                font_config["font.family"] = selected_font
            else:
                # 默认使用 DejaVu Sans
                mpl.rcParams["font.family"] = "DejaVu Sans"
                font_config["font.family"] = "DejaVu Sans"
            font_config["system"] = "Linux"

        # 设置字体大小和其他属性
        mpl.rcParams["font.size"] = 10
        mpl.rcParams["axes.titlesize"] = 12
        mpl.rcParams["axes.labelsize"] = 10
        mpl.rcParams["xtick.labelsize"] = 8
        mpl.rcParams["ytick.labelsize"] = 8
        mpl.rcParams["legend.fontsize"] = 9

        # 解决负号显示问题
        mpl.rcParams["axes.unicode_minus"] = False

    except Exception as e:
        # 如果字体设置失败，使用默认配置
        mpl.rcParams["font.family"] = "DejaVu Sans"
        font_config["font.family"] = "DejaVu Sans (fallback)"
        font_config["error"] = str(e)
        font_config["system"] = system

    return font_config

"""仿真核心工具模块初始化文件。

提供网络仿真核心功能相关的工具函数和辅助类，包括文件处理、数据转换等。
"""

import platform
import logging
from typing import Dict


logger = logging.getLogger(__name__)


def setup_matplotlib_font() -> Dict:
    """设置Matplotlib字体，确保在不同操作系统上正常显示中文。

    该函数会根据操作系统自动选择合适的字体配置，解决中文显示问题。

    Returns:
        Dict: 当前使用的字体配置，包含字体属性对象
    """
    import matplotlib as mpl
    from matplotlib.font_manager import FontManager, FontProperties

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
            # 创建字体属性对象
            font_config["font_properties"] = FontProperties(family="Arial Unicode MS")
        elif system == "Windows":  # Windows
            # Windows 系统使用系统内置的中文字体
            mpl.rcParams["font.family"] = "SimHei"
            font_config["font.family"] = "SimHei"
            font_config["system"] = "Windows"
            # 创建字体属性对象
            font_config["font_properties"] = FontProperties(family="SimHei")
        else:  # Linux 及其他系统
                # Linux 系统尝试使用常用的中文字体
                # 动态搜索系统中的中文字体文件
                import os
                from matplotlib.font_manager import FontProperties, findSystemFonts
                
                # 常见的中文字体名称和文件名模式
                chinese_font_patterns = [
                    "wqy-microhei",
                    "wqy-zenhei",
                    "notosanscjk",
                    "notoserifcjk",
                    "ukai",
                    "uming",
                    "simhei",
                    "simsun",
                    "microsoftyahei",
                ]
                
                # 搜索系统字体目录
                system_font_dirs = [
                    "/usr/share/fonts",
                    "/usr/local/share/fonts",
                    os.path.expanduser("~/.fonts"),
                    os.path.expanduser("~/.local/share/fonts"),
                ]
                
                # 动态查找可用的中文字体文件
                selected_font_file = None
                found_fonts = []
                
                # 方法1: 使用 findSystemFonts 查找所有系统字体
                all_system_fonts = findSystemFonts(fontpaths=None, fontext="ttf")
                all_system_fonts.extend(findSystemFonts(fontpaths=None, fontext="otf"))
                
                # 遍历所有系统字体，查找中文字体
                for font_path in all_system_fonts:
                    font_name = os.path.basename(font_path).lower()
                    for pattern in chinese_font_patterns:
                        if pattern in font_name:
                            found_fonts.append(font_path)
                            if selected_font_file is None:
                                selected_font_file = font_path
                                logger.info(f"找到可用中文字体文件: {font_path}")
                
                # 方法2: 如果方法1失败，遍历常见字体目录
                if not selected_font_file:
                    for font_dir in system_font_dirs:
                        if os.path.exists(font_dir):
                            for root, _, files in os.walk(font_dir):
                                for file in files:
                                    if file.lower().endswith((".ttf", ".ttc", ".otf")):
                                        font_path = os.path.join(root, file)
                                        font_name = file.lower()
                                        for pattern in chinese_font_patterns:
                                            if pattern in font_name:
                                                found_fonts.append(font_path)
                                                if selected_font_file is None:
                                                    selected_font_file = font_path
                                                    logger.info(f"找到可用中文字体文件: {font_path}")
                
                if selected_font_file:
                    # 直接设置字体
                    font_prop = FontProperties(fname=selected_font_file)
                    mpl.rcParams["font.family"] = font_prop.get_family()
                    mpl.rcParams["font.sans-serif"] = [font_prop.get_name()] + mpl.rcParams["font.sans-serif"]
                    font_config["font.family"] = font_prop.get_family()
                    font_config["font.file"] = selected_font_file
                    font_config["found_fonts"] = found_fonts[:5]  # 只记录前5个找到的字体
                    font_config["font_properties"] = font_prop
                else:
                    # 默认使用 sans-serif 字体家族
                    mpl.rcParams["font.family"] = ["sans-serif"]
                    mpl.rcParams["font.sans-serif"] = [
                        "WenQuanYi Micro Hei",
                        "WenQuanYi Zen Hei",
                        "Noto Sans CJK SC",
                        "Noto Sans CJK JP",
                        "DejaVu Sans",
                        "Arial",
                        "Helvetica",
                        "Verdana"
                    ]
                    font_config["font.family"] = "sans-serif"
                    font_config["font.sans-serif"] = mpl.rcParams["font.sans-serif"]
                    font_config["font_properties"] = None
                    logger.warning("未找到可用中文字体文件，使用默认字体配置")
                
                font_config["system"] = "Linux"
                logger.info(f"设置Linux字体配置: {font_config}")

        # 设置字体大小和其他属性
        mpl.rcParams["font.size"] = 10
        mpl.rcParams["axes.titlesize"] = 12
        mpl.rcParams["axes.labelsize"] = 10
        mpl.rcParams["xtick.labelsize"] = 8
        mpl.rcParams["ytick.labelsize"] = 8
        mpl.rcParams["legend.fontsize"] = 9

        # 解决负号显示问题
        mpl.rcParams["axes.unicode_minus"] = False

        logger.info(f"Matplotlib字体配置完成: {font_config}")

    except Exception as e:
        # 如果字体设置失败，使用默认配置
        mpl.rcParams["font.family"] = "DejaVu Sans"
        font_config["font.family"] = "DejaVu Sans (fallback)"
        font_config["error"] = str(e)
        font_config["system"] = system
        font_config["font_properties"] = None
        logger.error(f"字体设置失败: {e}")

    return font_config

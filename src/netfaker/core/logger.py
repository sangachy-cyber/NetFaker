"""日志初始化模块。

基于 loguru 实现统一、安全、可配置的日志系统。
"""

import sys
from pathlib import Path
from loguru import logger
from .config import config


def setup_logger() -> None:
    """初始化全局日志器。

    配置日志输出格式、级别和位置，支持控制台和文件双通道输出。
    """
    logger.remove()

    # 控制台输出（带颜色）
    logger.add(
        sys.stderr,
        level=config.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        enqueue=True,
        colorize=True,
    )

    # 文件输出（按需开启）
    if config.log_to_file:
        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / "netfaker_{time:YYYYMMDD}.log",
            level=config.log_level,
            rotation="100 MB",
            retention="7 days",
            encoding="utf-8",
            enqueue=True,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        )

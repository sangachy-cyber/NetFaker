"""日志管理模块。

负责配置和管理项目的日志系统。
"""

import logging
from typing import Optional

from netfaker.core.config import config


class Logger:
    """日志记录器类。

    封装了Python标准库的logging模块，提供统一的日志记录接口。
    """

    def __init__(self, name: str = "netfaker", level: Optional[str] = None):
        """初始化日志记录器。

        Args:
            name: 日志记录器名称
            level: 日志级别
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level or config.log_level)

        # 如果没有处理器，添加控制台处理器
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def debug(self, message: str, *args, **kwargs):
        """记录调试级别日志。

        Args:
            message: 日志消息
            args: 格式化参数
            kwargs: 额外参数
        """
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """记录信息级别日志。

        Args:
            message: 日志消息
            args: 格式化参数
            kwargs: 额外参数
        """
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """记录警告级别日志。

        Args:
            message: 日志消息
            args: 格式化参数
            kwargs: 额外参数
        """
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """记录错误级别日志。

        Args:
            message: 日志消息
            args: 格式化参数
            kwargs: 额外参数
        """
        self.logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        """记录严重错误级别日志。

        Args:
            message: 日志消息
            args: 格式化参数
            kwargs: 额外参数
        """
        self.logger.critical(message, *args, **kwargs)

    def exception(self, message: str, *args, **kwargs):
        """记录异常信息。

        Args:
            message: 日志消息
            args: 格式化参数
            kwargs: 额外参数
        """
        self.logger.exception(message, *args, **kwargs)


# 创建默认日志记录器
logger = Logger()

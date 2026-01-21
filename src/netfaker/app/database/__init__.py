"""数据库初始化模块。

负责初始化数据库连接和表结构。
"""

from .task_db import TaskDatabase

__all__ = ["TaskDatabase"]

"""任务数据库操作模块。

封装数据库连接和任务相关的CRUD操作。
"""

import json
import os
import sqlite3
from typing import Any, Dict, Optional

from loguru import logger


class TaskDatabase:
    """任务数据库操作类。

    封装数据库连接和任务相关的CRUD操作。

    Attributes:
        db_path: 数据库文件路径
        conn: 数据库连接对象
    """

    def __init__(self, db_path: str):
        """初始化数据库连接。

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.conn = self._create_connection()
        self._init_tables()

    def _create_connection(self) -> sqlite3.Connection:
        """创建数据库连接。

        Returns:
            sqlite3.Connection: 数据库连接对象
        """
        try:
            # 确保数据库路径是绝对路径
            db_path = self.db_path
            if db_path.startswith("sqlite:///"):
                # 处理SQLite URL格式，提取相对路径部分
                relative_path = db_path[10:]  # 移除 "sqlite:///"
                # 使用项目根目录作为基准路径
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
                db_path = os.path.join(project_root, relative_path)
                # 确保数据目录存在
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                logger.info(f"使用绝对数据库路径: {db_path}")
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            logger.info(f"成功连接到数据库: {db_path}")
            return conn
        except sqlite3.Error as e:
            logger.error(f"数据库连接失败: {e}")
            raise

    def _init_tables(self):
        """初始化数据库表。

        创建tasks表（如果不存在）。
        """
        create_tasks_table = """
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            description TEXT,
            strategy TEXT NOT NULL,
            target_ip TEXT NOT NULL,
            segments TEXT NOT NULL,
            result TEXT
        );
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(create_tasks_table)
            self.conn.commit()
            logger.info("成功初始化tasks表")
        except sqlite3.Error as e:
            logger.error(f"初始化数据库表失败: {e}")
            raise

    def insert_task(self, task_id: str, status: str, strategy: str,
                    target_ip: str, segments: Dict[str, Any],
                    description: Optional[str] = None) -> bool:
        """插入新任务。

        Args:
            task_id: 任务ID
            status: 任务状态
            strategy: 生成策略
            target_ip: 目标设备IP
            segments: 仿真阶段列表
            description: 任务描述（可选）

        Returns:
            bool: 插入是否成功
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO tasks (task_id, status, description, strategy,
                               target_ip, segments, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (task_id, status, description, strategy,
                 target_ip, json.dumps(segments))
            )
            self.conn.commit()
            logger.info(f"成功插入任务: {task_id}")
            return True
        except sqlite3.Error as e:
            logger.error(f"插入任务失败: {e}")
            return False

    def update_task_status(self, task_id: str, status: str,
                          result: Optional[Dict[str, Any]] = None) -> bool:
        """更新任务状态。

        Args:
            task_id: 任务ID
            status: 新的任务状态
            result: 仿真结果（可选）

        Returns:
            bool: 更新是否成功
        """
        try:
            cursor = self.conn.cursor()
            result_json = json.dumps(result) if result else None
            cursor.execute(
                """
                UPDATE tasks
                SET status = ?, result = ?, updated_at = CURRENT_TIMESTAMP
                WHERE task_id = ?
                """,
                (status, result_json, task_id)
            )
            self.conn.commit()
            logger.info(f"成功更新任务状态: {task_id} -> {status}")
            return True
        except sqlite3.Error as e:
            logger.error(f"更新任务状态失败: {e}")
            return False

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """根据任务ID获取任务。

        Args:
            task_id: 任务ID

        Returns:
            Optional[Dict[str, Any]]: 任务信息，如果不存在返回None
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT * FROM tasks WHERE task_id = ?",
                (task_id,)
            )
            row = cursor.fetchone()
            if row:
                task = dict(row)
                # 解析JSON字段
                task["segments"] = json.loads(task["segments"])
                if task["result"]:
                    task["result"] = json.loads(task["result"])
                return task
            return None
        except sqlite3.Error as e:
            logger.error(f"获取任务失败: {e}")
            return None

    def close(self):
        """关闭数据库连接。
        """
        if self.conn:
            self.conn.close()
            logger.info("数据库连接已关闭")

    def __del__(self):
        """析构函数，关闭数据库连接。
        """
        self.close()

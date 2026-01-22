"""任务管理服务模块。

封装任务管理的业务逻辑，包括任务创建、状态更新和结果生成。
"""

import asyncio
import os
import random
import string
from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger

from netfaker.app.database.task_db import TaskDatabase
from netfaker.core.config import config
from netfaker.simcore.generator import generate_simulation_params


class TaskManager:
    """任务管理服务类。

    封装任务管理的业务逻辑，包括任务创建、状态更新和结果生成。

    Attributes:
        db: TaskDatabase实例，用于数据库操作
        output_dir: 结果文件输出目录
    """

    def __init__(self):
        """初始化任务管理服务。
        """
        self.db = TaskDatabase(config.database_url)
        self.output_dir = config.output_dir
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_task_id(self) -> str:
        """生成唯一任务ID。

        Returns:
            str: 任务ID，格式：task_${timestamp}_${random}
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        return f"task_{timestamp}_{random_str}"

    async def create_task(self, strategy: str, target_ip: str,
                         segments: Dict[str, Any], description: Optional[str] = None) -> str:
        """创建新任务。

        Args:
            strategy: 生成策略
            target_ip: 目标设备IP
            segments: 仿真阶段列表
            description: 任务描述（可选）

        Returns:
            str: 生成的任务ID
        """
        task_id = self.generate_task_id()

        # 插入任务到数据库
        self.db.insert_task(
            task_id=task_id,
            status="queued",
            strategy=strategy,
            target_ip=target_ip,
            segments=segments,
            description=description
        )

        # 异步处理任务
        asyncio.create_task(self.process_task(task_id, strategy, target_ip, segments, description))

        return task_id

    async def process_task(self, task_id: str, strategy: str, target_ip: str,
                         segments: Dict[str, Any], description: Optional[str] = None):
        """处理仿真任务。

        Args:
            task_id: 任务ID
            strategy: 生成策略
            target_ip: 目标设备IP
            segments: 仿真阶段列表
            description: 任务描述（可选）
        """
        try:
            # 更新任务状态为processing
            self.db.update_task_status(task_id, "processing")

            # 策略名称映射，将API文档中的策略名映射到实际注册的策略名
            strategy_map = {
                "rule": "rule_based",
                "model": "model_based"  # 暂时使用占位符，后续会实现
            }

            # 获取实际策略名称
            actual_strategy = strategy_map.get(strategy, strategy)

            # 生成输出文件路径
            filename = f"sim_{task_id}.txt"
            output_path = os.path.join(self.output_dir, filename)

            # 准备策略参数
            strategy_params = {
                "segments": segments,
                "output_path": output_path,
                "target_ip": target_ip,
                "description": description
            }

            # 生成仿真参数
            logger.info(f"开始生成仿真参数: {task_id}")
            logger.info(f"使用策略: {actual_strategy}, 参数: {strategy_params}")
            simulation_params = generate_simulation_params(actual_strategy, strategy_params)
            logger.info(f"仿真参数生成成功: {simulation_params}")

            # 获取文件路径
            holowan_file_path = simulation_params["output_file"]
            logger.info(f"HoloWAN文件生成成功: {holowan_file_path}")

            # 更新任务结果
            result = {
                "config_file_url": os.path.basename(holowan_file_path)
            }
            self.db.update_task_status(task_id, "completed", result)

            logger.info(f"任务处理完成: {task_id}")
        except Exception as e:
            logger.error(f"任务处理失败: {task_id}, 错误: {e}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            self.db.update_task_status(task_id, "failed")

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态。

        Args:
            task_id: 任务ID

        Returns:
            Optional[Dict[str, Any]]: 任务状态信息，如果任务不存在返回None
        """
        return self.db.get_task(task_id)

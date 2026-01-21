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
from netfaker.simcore.utils.holowan import HoloWANFile


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

            # 准备策略参数，根据策略类型转换segments格式
            strategy_params = {}
            if actual_strategy == "rule_based":
                # rule_based策略需要scenario参数，从segments中提取或使用默认值
                # 使用第一个segment的type作为scenario，或者使用默认值"video_streaming"
                # 如果场景不支持，使用默认值
                scenario = segments[0]["type"] if segments else "video_streaming"
                logger.info(f"处理任务 {task_id}: 提取场景 {scenario}")
                # 支持的场景列表
                supported_scenarios = ["video_streaming", "online_gaming", "web_browsing", "file_download"]
                # 如果场景不支持，使用默认值
                if scenario not in supported_scenarios:
                    logger.warning(f"场景 {scenario} 不支持，使用默认值 video_streaming")
                    scenario = "video_streaming"
                strategy_params["scenario"] = scenario
                logger.info(f"处理任务 {task_id}: 使用场景 {scenario}")
            else:
                # 其他策略直接使用segments
                strategy_params["segments"] = segments

            # 生成仿真参数
            logger.info(f"开始生成仿真参数: {task_id}")
            logger.info(f"使用策略: {actual_strategy}, 参数: {strategy_params}")
            simulation_params = generate_simulation_params(actual_strategy, strategy_params)
            logger.info(f"仿真参数生成成功: {simulation_params}")

            # 生成HoloWAN文件
            holowan_file_path = await self._generate_holowan_file(
                task_id, target_ip, simulation_params, description
            )
            logger.info(f"HoloWAN文件生成成功: {holowan_file_path}")

            # 更新任务结果
            result = {
                "config_file_url": holowan_file_path
            }
            self.db.update_task_status(task_id, "completed", result)

            logger.info(f"任务处理完成: {task_id}")
        except Exception as e:
            logger.error(f"任务处理失败: {task_id}, 错误: {e}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            self.db.update_task_status(task_id, "failed")

    async def _generate_holowan_file(self, task_id: str, target_ip: str,
                                     simulation_params: Dict[str, Any],
                                     description: Optional[str] = None) -> str:
        """生成HoloWAN文件。

        Args:
            task_id: 任务ID
            target_ip: 目标设备IP
            simulation_params: 仿真参数
            description: 任务描述（可选）

        Returns:
            str: 生成的文件相对路径
        """
        # 创建HoloWANFile实例
        holowan_file = HoloWANFile()

        # 设置文件属性
        holowan_file.test_name = description or f"generated_{task_id}"
        holowan_file.destination = f"{target_ip}:8081"

        # 添加数据点
        # 这里需要根据simulation_params生成具体的HoloWAN数据点
        # 目前简化实现，生成示例数据
        for i in range(10):  # 生成10个数据点
            holowan_file._add_data(
                ul_delay=100.0 + i * 2,
                ul_loss=0.1 + i * 0.01,
                ul_bw=10.0 - i * 0.5,
                dl_delay=95.0 + i * 1.5,
                dl_loss=0.05 + i * 0.005,
                dl_bw=12.0 - i * 0.3
            )

        # 生成文件名
        filename = f"sim_{task_id}.txt"
        file_path = os.path.join(self.output_dir, filename)

        # 写入文件
        holowan_file.write_to_file(file_path)

        # 返回相对路径
        return os.path.relpath(file_path)

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态。

        Args:
            task_id: 任务ID

        Returns:
            Optional[Dict[str, Any]]: 任务状态信息，如果任务不存在返回None
        """
        return self.db.get_task(task_id)

"""API路由定义模块。

定义NetFaker API的所有路由端点。
"""

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from loguru import logger

from netfaker.app.api.schemas import SimulationRequest, TaskStatus
from netfaker.app.services.task_manager import TaskManager
from netfaker.core.config import config

# 创建路由实例
router = APIRouter()

# 初始化任务管理器
task_manager = TaskManager()


@router.post("/simulate", response_model=dict)
async def simulate_network(request: SimulationRequest):
    """生成网络仿真参数。

    Args:
        request: 仿真请求参数

    Returns:
        dict: 仿真结果
    """
    try:
        logger.info(f"收到仿真请求: {request}")

        # 转换为字典格式
        request_dict = request.model_dump()
        segments = request_dict.pop("segments")

        # 创建任务
        task_id = await task_manager.create_task(
            strategy=request_dict["strategy"],
            target_ip=str(request_dict["target_ip"]),
            segments=segments,
            description=request_dict["description"]
        )

        return {
            "task_id": task_id,
            "status": "queued",
            "message": "任务已提交"
        }
    except ValueError as e:
        logger.error(f"仿真请求参数验证失败: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"仿真请求处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """获取仿真任务状态。

    Args:
        task_id: 任务ID

    Returns:
        TaskStatus: 任务状态信息
    """
    try:
        logger.info(f"查询任务状态: {task_id}")

        # 查询任务状态
        task = task_manager.get_task_status(task_id)

        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")

        # 转换为TaskStatus模型
        return TaskStatus(
            task_id=task["task_id"],
            status=task["status"],
            created_at=task["created_at"],
            updated_at=task["updated_at"],
            result=task["result"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"任务状态查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/outputs/{filename}.txt")
async def download_file(filename: str):
    """下载配置文件。

    Args:
        filename: 文件名（不含扩展名）

    Returns:
        FileResponse: 配置文件
    """
    try:
        logger.info(f"下载文件请求: {filename}.txt")

        # 构建完整文件路径
        file_path = os.path.join(config.output_dir, f"{filename}.txt")

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="文件不存在")

        # 返回文件响应
        return FileResponse(
            path=file_path,
            media_type="text/plain; charset=utf-8",
            filename=f"{filename}.txt"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件下载失败: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

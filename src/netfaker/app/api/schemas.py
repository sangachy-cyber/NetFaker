"""API请求响应模型模块。

使用Pydantic定义API请求和响应的数据结构。
"""

from typing import List, Optional

from pydantic import BaseModel, Field, IPvAnyAddress


class SimulationSegment(BaseModel):
    """仿真阶段模型。

    Attributes:
        type: 阶段类型标识
        duration: 持续时间（秒），必须是10的正整数倍
        condition: 预留字段，当前可选且无作用
    """
    type: str = Field(..., description="阶段类型标识")
    duration: int = Field(..., gt=0, description="持续时间（秒），必须是10的正整数倍")
    condition: Optional[str] = Field(None, description="预留字段，当前可选且无作用")

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "s0",
                "duration": 60
            }
        }
    }

    def validate_duration(self):
        """验证duration是否为10的倍数。

        Raises:
            ValueError: 如果duration不是10的倍数
        """
        if self.duration % 10 != 0:
            raise ValueError("duration必须是10的正整数倍")

    def __init__(self, **data):
        """初始化并验证duration。
        """
        super().__init__(**data)
        self.validate_duration()


class SimulationRequest(BaseModel):
    """仿真任务请求模型。

    Attributes:
        strategy: 生成策略，固定值："model" 或 "rule"
        target_ip: 目标设备IPv4地址
        description: 任务描述（可选）
        segments: 仿真阶段列表
    """
    strategy: str = Field(..., description="生成策略，固定值：'model' 或 'rule'")
    target_ip: IPvAnyAddress = Field(..., description="目标设备IPv4地址")
    description: Optional[str] = Field(None, description="任务描述")
    segments: List[SimulationSegment] = Field(..., description="仿真阶段列表")

    model_config = {
        "json_schema_extra": {
            "example": {
                "strategy": "rule",
                "target_ip": "172.30.153.236",
                "description": "Xicen yuanshen(12-04 00:12:04)",
                "segments": [
                    {"type": "s0", "duration": 60}
                ]
            }
        }
    }

    def validate_strategy(self):
        """验证strategy是否为有效值。

        Raises:
            ValueError: 如果strategy不是"model"或"rule"
        """
        if self.strategy not in ["model", "rule"]:
            raise ValueError("strategy必须是'model'或'rule'")

    def __init__(self, **data):
        """初始化并验证strategy。
        """
        super().__init__(**data)
        self.validate_strategy()


class TaskResult(BaseModel):
    """任务结果模型。

    Attributes:
        config_file_url: 配置文件URL
    """
    config_file_url: str = Field(..., description="配置文件URL")

    model_config = {
        "json_schema_extra": {
            "example": {
                "config_file_url": "/api/v1/outputs/sim_task_20251204_xyz789.txt"
            }
        }
    }


class TaskStatus(BaseModel):
    """任务状态响应模型。

    Attributes:
        task_id: 任务ID
        status: 任务状态
        created_at: 创建时间
        updated_at: 更新时间
        result: 任务结果（仅当状态为completed时）
    """
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    created_at: str = Field(..., description="创建时间")
    updated_at: str = Field(..., description="更新时间")
    result: Optional[TaskResult] = Field(None, description="任务结果")

    model_config = {
        "json_schema_extra": {
            "example": {
                "task_id": "task_20251204_xyz789",
                "status": "completed",
                "created_at": "2025-12-04T00:12:00Z",
                "updated_at": "2025-12-04T00:12:10Z",
                "result": {
                    "config_file_url": "/api/v1/outputs/sim_task_20251204_xyz789.txt"
                }
            }
        }
    }


class ErrorResponse(BaseModel):
    """错误响应模型。

    Attributes:
        error: 错误信息
        code: 错误码
    """
    error: str = Field(..., description="错误信息")
    code: int = Field(..., description="错误码")

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "Invalid request parameters",
                "code": 400
            }
        }
    }


class SimulationResponse(BaseModel):
    """仿真任务提交响应模型。

    Attributes:
        task_id: 任务ID
        status: 任务状态
        message: 响应消息
    """
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    message: str = Field(..., description="响应消息")

    model_config = {
        "json_schema_extra": {
            "example": {
                "task_id": "task_20251204_xyz789",
                "status": "queued",
                "message": "任务已提交"
            }
        }
    }

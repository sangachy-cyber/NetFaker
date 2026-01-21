"""工具函数模块。

提供项目中常用的工具函数。
"""

import os
import uuid


def generate_uuid() -> str:
    """生成UUID字符串。

    Returns:
        str: UUID字符串
    """
    return str(uuid.uuid4())


def ensure_dir(path: str) -> None:
    """确保目录存在，如果不存在则创建。

    Args:
        path: 目录路径
    """
    os.makedirs(path, exist_ok=True)


def get_file_extension(file_path: str) -> str:
    """获取文件扩展名。

    Args:
        file_path: 文件路径

    Returns:
        str: 文件扩展名（不含点）
    """
    return os.path.splitext(file_path)[1][1:]


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """安全除法，避免除以零错误。

    Args:
        a: 被除数
        b: 除数
        default: 除数为零时的默认返回值

    Returns:
        float: 除法结果
    """
    if b == 0:
        return default
    return a / b


def validate_range(value: float, min_val: float, max_val: float, name: str) -> float:
    """验证数值是否在指定范围内。

    Args:
        value: 要验证的数值
        min_val: 最小值
        max_val: 最大值
        name: 参数名称

    Returns:
        float: 验证后的数值

    Raises:
        ValueError: 当数值超出范围时
    """
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} 必须在 [{min_val}, {max_val}] 范围内，当前值: {value}")
    return value

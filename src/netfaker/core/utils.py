"""工具函数模块。

提供项目中常用的工具函数，包括UUID生成、目录管理、文件操作和数值验证等。
"""

import os
import uuid


def generate_uuid() -> str:
    """生成UUID字符串。

    生成一个唯一的UUID字符串，用于标识任务、文件等。

    Returns:
        str: 生成的UUID字符串

    Examples:
        >>> generate_uuid()
        '123e4567-e89b-12d3-a456-426614174000'
    """
    return str(uuid.uuid4())


def ensure_dir(path: str) -> None:
    """确保目录存在，如果不存在则创建。

    递归创建目录结构，确保指定的路径存在。

    Args:
        path: 目录路径

    Examples:
        >>> ensure_dir('data/output')
        # 如果目录不存在，会创建 data/output 目录
    """
    os.makedirs(path, exist_ok=True)


def get_file_extension(file_path: str) -> str:
    """获取文件扩展名。

    从文件路径中提取扩展名，不包含点号。

    Args:
        file_path: 文件路径

    Returns:
        str: 文件扩展名（不含点）

    Examples:
        >>> get_file_extension('file.txt')
        'txt'
        >>> get_file_extension('path/to/file.csv')
        'csv'
    """
    return os.path.splitext(file_path)[1][1:]


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """安全除法，避免除以零错误。

    执行除法运算，当除数为零时返回默认值。

    Args:
        a: 被除数
        b: 除数
        default: 除数为零时的默认返回值

    Returns:
        float: 除法结果

    Examples:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0)
        0.0
        >>> safe_divide(10, 0, default=1.0)
        1.0
    """
    if b == 0:
        return default
    return a / b


def validate_range(value: float, min_val: float, max_val: float, name: str) -> float:
    """验证数值是否在指定范围内。

    检查数值是否在给定的最小值和最大值之间。

    Args:
        value: 要验证的数值
        min_val: 最小值
        max_val: 最大值
        name: 参数名称，用于错误消息

    Returns:
        float: 验证后的数值

    Raises:
        ValueError: 当数值超出范围时

    Examples:
        >>> validate_range(5, 0, 10, 'test')
        5
        >>> validate_range(-1, 0, 10, 'test')
        Traceback (most recent call last):
            ...
        ValueError: test 必须在 [0, 10] 范围内，当前值: -1
    """
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} 必须在 [{min_val}, {max_val}] 范围内，当前值: {value}")
    return value

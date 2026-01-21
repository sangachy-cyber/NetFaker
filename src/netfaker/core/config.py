"""配置管理模块。

负责读取和管理项目的配置信息。
"""

import os
from dataclasses import dataclass


@dataclass
class NetFakerConfig:
    """NetFaker 项目配置类。

    Attributes:
        debug: 是否开启调试模式
        host: 服务监听地址
        port: 服务监听端口
        log_level: 日志级别
        data_dir: 数据目录路径
    """
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    data_dir: str = "data"


def get_config() -> NetFakerConfig:
    """获取配置实例。

    Returns:
        NetFakerConfig: 配置实例
    """
    return NetFakerConfig(
        debug=os.getenv("NETFAKER_DEBUG", "True").lower() == "true",
        host=os.getenv("NETFAKER_HOST", "0.0.0.0"),
        port=int(os.getenv("NETFAKER_PORT", "8000")),
        log_level=os.getenv("NETFAKER_LOG_LEVEL", "INFO"),
        data_dir=os.getenv("NETFAKER_DATA_DIR", "data"),
    )


config = get_config()

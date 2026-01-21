"""配置管理模块。

负责读取和管理项目的配置信息。
"""

from pydantic_settings import BaseSettings


class NetFakerConfig(BaseSettings):
    """NetFaker 项目配置类。

    Attributes:
        debug: 是否开启调试模式
        host: 服务监听地址
        port: 服务监听端口
        log_level: 日志级别
        data_dir: 数据目录路径
        holowan_api_url: HoloWAN API地址
        holowan_api_key: HoloWAN API密钥
        holowan_engine_id: HoloWAN引擎ID
        database_url: 数据库URL
        model_path: 模型文件路径
    """
    # 应用配置
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    data_dir: str = "data"
    
    # HoloWAN API 配置
    holowan_api_url: str = "http://localhost:8080"
    holowan_api_key: str = ""
    holowan_engine_id: int = 1
    
    # 数据库配置
    database_url: str = "sqlite:///./netfaker.db"
    
    # 模型配置
    model_path: str = "data/models/default_model.pt"
    
    class Config:
        """Pydantic 配置类。
        
        设置从环境变量读取配置，支持.env文件。
        """
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_config() -> NetFakerConfig:
    """获取配置实例。

    Returns:
        NetFakerConfig: 配置实例
    """
    return NetFakerConfig()


config = get_config()

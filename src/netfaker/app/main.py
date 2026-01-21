"""FastAPI应用初始化模块。

负责创建和配置FastAPI应用实例。
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from netfaker.app.api import endpoints
from netfaker.core.config import config
from netfaker.core.logger import setup_logger

# 初始化日志系统（必须在 app 创建前调用）
setup_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理器。

    处理应用的启动和关闭事件。
    """
    # 启动时执行
    logger.info("NetFaker API服务启动成功")
    logger.info(f"服务地址: http://{config.host}:{config.port}")
    logger.info(f"调试模式: {'开启' if config.debug else '关闭'}")
    yield
    # 关闭时执行
    logger.info("NetFaker API服务已关闭")


# 创建FastAPI应用实例
app = FastAPI(
    title="NetFaker API",
    description="网络仿真参数生成API",
    version="0.1.0",
    debug=config.debug,
    lifespan=lifespan,
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 根路径路由
@app.get("/")
async def root():
    """根路径。

    返回API基本信息。
    """
    return {
        "name": "NetFaker API",
        "version": "0.1.0",
        "description": "网络仿真参数生成API",
    }


# 健康检查路由
@app.get("/health")
async def health_check():
    """健康检查。

    用于检查API服务是否正常运行。
    """
    return {"status": "healthy"}


app.include_router(endpoints.router, prefix="/api/v1")

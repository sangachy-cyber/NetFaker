"""仿真参数生成器模块。

集成所有仿真策略，提供统一的参数生成接口。
"""

from typing import Any, Dict

from loguru import logger

# 导入策略实现以触发注册
import netfaker.simcore.strategies.rule_based  # noqa: F401

# 导入本地模块
from netfaker.core.exceptions import StrategyError
from netfaker.simcore.strategy import SimulationStrategy, strategy_registry

# 临时添加model_based策略占位符，避免策略不存在错误


class ModelBasedStrategy(SimulationStrategy):
    """基于模型的仿真策略占位符。

    实际实现将在后续添加。
    """

    @property
    def name(self) -> str:
        return "model_based"

    @property
    def description(self) -> str:
        return "基于机器学习模型的仿真参数生成策略"

    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # 简单实现，返回固定参数
        return {
            "strategy": self.name,
            "parameters": {
                "delay_ms": 100.0,
                "loss_percent": 0.1,
                "jitter_ms": 5.0,
                "bandwidth_mbps": 10.0
            }
        }

# 注册model_based策略
strategy_registry.register(ModelBasedStrategy())


def generate_simulation_params(strategy_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """生成网络仿真参数。

    Args:
        strategy_name: 策略名称
        params: 生成请求参数

    Returns:
        Dict[str, Any]: 生成的仿真参数

    Raises:
        StrategyError: 当策略执行失败时
    """
    try:
        logger.info(f"使用策略 '{strategy_name}' 生成仿真参数")

        # 获取指定策略
        strategy = strategy_registry.get_strategy(strategy_name)

        # 生成参数
        result = strategy.generate(params)

        logger.info(f"仿真参数生成成功: {result}")
        return result
    except Exception as e:
        logger.error(f"仿真参数生成失败: {e}")
        raise StrategyError(f"策略执行失败: {str(e)}") from e


def list_available_strategies() -> Dict[str, Dict[str, str]]:
    """获取所有可用的仿真策略。

    Returns:
        Dict[str, Dict[str, str]]: 可用策略列表
    """
    return strategy_registry.list_strategies()

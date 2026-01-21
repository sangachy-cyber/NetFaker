"""仿真参数生成器模块。

集成所有仿真策略，提供统一的参数生成接口。
"""

from typing import Any, Dict

from netfaker.core.exceptions import StrategyError
from netfaker.core.logging import logger

# 导入所有策略（自动注册）
from netfaker.simcore.strategy import strategy_registry
from netfaker.simcore.strategies import rule_based


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

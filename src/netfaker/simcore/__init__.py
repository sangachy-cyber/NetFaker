"""仿真核心模块。

包含网络仿真参数生成的核心逻辑和策略。
"""

from netfaker.simcore.generator import (
    generate_simulation_params,
    list_available_strategies,
)
from netfaker.simcore.strategy import SimulationStrategy, StrategyRegistry

__all__ = [
    "generate_simulation_params",
    "list_available_strategies",
    "SimulationStrategy",
    "StrategyRegistry",
]

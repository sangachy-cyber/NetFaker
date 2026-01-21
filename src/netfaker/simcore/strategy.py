"""仿真策略基类模块。

定义仿真策略的基类，所有具体策略都应继承此类。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class SimulationStrategy(ABC):
    """仿真策略抽象基类。

    定义了仿真策略的基本接口，所有具体策略都应实现这些接口。
    """

    @abstractmethod
    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """生成仿真参数。

        Args:
            params: 生成请求参数

        Returns:
            Dict[str, Any]: 生成的仿真参数

        Raises:
            StrategyError: 当策略执行失败时
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称。

        Returns:
            str: 策略名称
        """

    @property
    @abstractmethod
    def description(self) -> str:
        """策略描述。

        Returns:
            str: 策略描述
        """


class StrategyRegistry:
    """策略注册表。

    用于注册和管理所有可用的仿真策略。
    """

    def __init__(self):
        """初始化策略注册表。"""
        self._strategies: Dict[str, SimulationStrategy] = {}

    def register(self, strategy: SimulationStrategy) -> None:
        """注册策略。

        Args:
            strategy: 要注册的策略实例
        """
        self._strategies[strategy.name] = strategy

    def get_strategy(self, name: str) -> SimulationStrategy:
        """获取指定名称的策略。

        Args:
            name: 策略名称

        Returns:
            SimulationStrategy: 策略实例

        Raises:
            ValueError: 当策略不存在时
        """
        if name not in self._strategies:
            raise ValueError(f"策略 '{name}' 不存在")
        return self._strategies[name]

    def list_strategies(self) -> Dict[str, Dict[str, str]]:
        """获取所有可用策略的列表。

        Returns:
            Dict[str, Dict[str, str]]: 策略列表，包含名称和描述
        """
        return {
            name: {
                "name": name,
                "description": strategy.description,
            }
            for name, strategy in self._strategies.items()
        }


# 创建全局策略注册表实例
strategy_registry = StrategyRegistry()

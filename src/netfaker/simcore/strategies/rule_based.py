"""基于规则的仿真参数生成策略。

支持常见网络场景（如视频流、在线游戏）的硬编码参数映射。
适用于无历史数据或需快速响应的仿真任务。
"""

from typing import Any, Dict

from netfaker.core.exceptions import ValidationError
from netfaker.simcore.strategy import SimulationStrategy, strategy_registry


class RuleBasedStrategy(SimulationStrategy):
    """基于规则的仿真策略。

    根据预定义的场景规则生成网络仿真参数。
    """

    # 预定义场景参数映射
    _SCENARIOS = {
        "video_streaming": {
            "delay_ms": 50.0,
            "loss_percent": 0.1,
            "jitter_ms": 10.0,
            "bandwidth_mbps": 5.0,
        },
        "online_gaming": {
            "delay_ms": 20.0,
            "loss_percent": 0.01,
            "jitter_ms": 2.0,
            "bandwidth_mbps": 10.0,
        },
        "web_browsing": {
            "delay_ms": 100.0,
            "loss_percent": 0.5,
            "jitter_ms": 20.0,
            "bandwidth_mbps": 2.0,
        },
        "file_download": {
            "delay_ms": 150.0,
            "loss_percent": 0.3,
            "jitter_ms": 30.0,
            "bandwidth_mbps": 100.0,
        },
    }

    @property
    def name(self) -> str:
        """策略名称。

        Returns:
            str: 策略名称
        """
        return "rule_based"

    @property
    def description(self) -> str:
        """策略描述。

        Returns:
            str: 策略描述
        """
        return "基于预定义规则的仿真参数生成策略，支持常见网络场景"

    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """生成仿真参数。

        Args:
            params: 生成请求参数，包含场景类型

        Returns:
            Dict[str, Any]: 生成的仿真参数

        Raises:
            ValidationError: 当输入参数不符合要求时
        """
        # 验证参数
        if "scenario" not in params:
            raise ValidationError("缺少必要参数: scenario")

        scenario = params["scenario"]
        if scenario not in self._SCENARIOS:
            raise ValidationError(f"不支持的场景: {scenario}")

        # 获取预定义参数
        scenario_params = self._SCENARIOS[scenario].copy()

        # 应用可选的参数调整
        if "adjustments" in params:
            scenario_params.update(params["adjustments"])

        return {
            "strategy": self.name,
            "scenario": scenario,
            "parameters": scenario_params,
        }


# 注册策略
strategy_registry.register(RuleBasedStrategy())

"""基于规则的仿真参数生成策略。

使用RuleSynthesizer从已标注的窗口数据中生成仿真流量文件。
"""

from typing import Any, Dict

from loguru import logger

from netfaker.core.config import config
from netfaker.core.exceptions import ValidationError
from netfaker.simcore.strategy import SimulationStrategy, strategy_registry
from netfaker.simcore.synthesizer.rule_synthesizer import RuleSynthesizer


class RuleBasedStrategy(SimulationStrategy):
    """基于规则的仿真策略。

    使用RuleSynthesizer从已标注的窗口数据中生成仿真流量文件。
    """

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
        return "基于规则的仿真策略，使用RuleSynthesizer生成仿真流量文件"

    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """生成仿真参数。

        Args:
            params: 生成请求参数，包含segments、output_path等

        Returns:
            Dict[str, Any]: 生成的仿真参数，包含文件路径

        Raises:
            ValidationError: 当输入参数不符合要求时
        """
        # 验证参数
        if "segments" not in params:
            raise ValidationError("缺少必要参数: segments")
        if "output_path" not in params:
            raise ValidationError("缺少必要参数: output_path")

        logger.info(f"RuleBasedStrategy.generate: 处理请求，segments={params['segments']}")

        # 初始化RuleSynthesizer
        synth = RuleSynthesizer(config.window_pool_path)

        # 生成仿真文件
        output_file = synth.generate(params["segments"], params["output_path"])
        logger.info(f"RuleBasedStrategy.generate: 生成文件成功，路径={output_file}")

        return {
            "strategy": self.name,
            "output_file": output_file,
        }


# 注册策略
strategy_registry.register(RuleBasedStrategy())

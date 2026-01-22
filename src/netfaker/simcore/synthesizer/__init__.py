"""基于规则的状态序列生成模块。

提供基于规则的网络状态序列生成功能，支持根据用户定义的状态-时长模板，
从已标注的窗口数据中抽样并生成仿真流量文件。
"""

from netfaker.simcore.synthesizer.rule_synthesizer import RuleSynthesizer

__all__ = ["RuleSynthesizer"]

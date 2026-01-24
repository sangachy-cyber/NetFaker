"""NetFaker 合成数据评估模块。

负责量化评估合成 HoloWAN 文件在单窗口粒度上对真实网络行为的复现能力。
"""

from netfaker.evaluator.holowan_evaluator import HoloWANEvaluator
from netfaker.evaluator.reporter import EvaluationReporter

__all__ = ["HoloWANEvaluator", "EvaluationReporter"]

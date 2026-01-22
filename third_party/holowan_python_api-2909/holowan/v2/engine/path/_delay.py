"""
HoloWAN Network Emulator
Python API
"""

import xml.etree.cElementTree as ET

from holowan.v2._holowan_types import check_parameter
from holowan.v2.engine.path._change_mode import ChangeMode, _CHANGE_MODE_TAG
from holowan.v2.engine.path._impairment_base import Delay, _IMPAIRMENT_TYPE

_DELAY_CONSTANT_MODE: int = 1
_DELAY_UNIFORM_MODE: int = 2
_DELAY_NORMAL_MODE: int = 3
_DELAY_CUSTOM_MODE: int = 4
_DELAY_JITTER_MODE: int = 5
_DELAY_GAMMA_MODE: int = 6
_DELAY_ACCUMULATE_MODE: int = 7
_DELAY_CUSTOMIZED_FILE_MODE: int = 8

_CONSTANT_TAG: str = r"co"
_UNIFORM_TAG: str = r"un"
_NORMAL_TAG: str = r"no"
_CUSTOM_TAG: str = r"cu"
_JITTER_TAG: str = r"ji"
_GAMMA_TAG: str = r"ga"
_ACCUMULATE_TAG: str = r"acc"
_CUSTOMIZED_FILE_TAG: str = r"customized"

import os
from holowan.v2.utils import my_util as mt
from holowan.v2._holowan_types import check_param_decimal, check_param_in_choices, check_param_in_range


class _DelayLimits(object):
    _CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
    _INI_PATH: str = os.path.join(_CURRENT_DIR, r"../../../resources/HoloWAN.ini")
    _INI = mt.open_ini(_INI_PATH)

    constant_delay_decimal = _INI.get("delay", "constantDelayDecimal")

    uniform_minimum_decimal = _INI.get("delay", "uniformMinimumDecimal")
    uniform_maximum_decimal = _INI.get("delay", "uniformMaximumDecimal")

    normal_min_decimal = _INI.get("delay", "normalMinDecimal")
    normal_mean_decimal = _INI.get("delay", "normalMeanDecimal")
    normal_std_deviation_decimal = _INI.get("delay", "normalStdDeviationDecimal")
    normal_advanced_period_decimal = _INI.get("delay", "normalAdvancedPeriodDecimal")
    normal_advanced_duration_decimal = _INI.get("delay", "normalAdvancedDurationDecimal")
    normal_advanced_min_decimal = _INI.get("delay", "normalAdvancedMinDecimal")
    normal_advanced_max_decimal = _INI.get("delay", "normalAdvancedMaxDecimal")

    custom_mean_delay_decimal = _INI.get("delay", "customMeanDelayDecimal")
    custom_min_delay_decimal = _INI.get("delay", "customMinDelayDecimal")
    custom_max_delay_decimal = _INI.get("delay", "customMaxDelayDecimal")
    custom_positive_delta_decimal = _INI.get("delay", "customPositiveDeltaDecimal")
    custom_negative_delta_decimal = _INI.get("delay", "customNegativeDeltaDecimal")
    custom_spread_delta_decimal = _INI.get("delay", "customSpreadDeltaDecimal")

    jitter_mean_delay_decimal = _INI.get("delay", "jitterMeanDelayDecimal")
    jitter_mean_jitter_decimal = _INI.get("delay", "jitterMeanJitterDecimal")

    enable_reo_list = [0, 1]

    _msg = 'Argument {argument} must be {expected!r}, but got {got!r}, value {value!r}'

    def check_delay_constant(self, kwargs: dict):
        for k, v in kwargs.items():
            if k == _CONSTANT_NORMAL_DELAY:
                check_param_decimal("delay", v, self.constant_delay_decimal)

    def check_delay_uniform(self, kwargs: dict):
        for k, v in kwargs.items():
            if k == _UNIFORM_MIN:
                check_param_decimal("minimum", v, self.uniform_minimum_decimal)
            elif k == _UNIFORM_MAX:
                check_param_decimal("maximum", v, self.uniform_maximum_decimal)
            elif k == _UNIFORM_ENABLE_REO:
                check_param_in_choices("enable_reordering", v, self.enable_reo_list)

    def check_delay_normal(self, kwargs: dict):
        for k, v in kwargs.items():
            if k == _NORMAL_MIN:
                check_param_decimal("minimum", v, self.normal_min_decimal)
            elif k == _NORMAL_MEAN:
                check_param_decimal("mean", v, self.normal_mean_decimal)
            elif k == _NORMAL_STD:
                check_param_decimal("std_deviation", v, self.normal_std_deviation_decimal)
            elif k == _UNIFORM_ENABLE_REO:
                check_param_in_choices("enable_reordering", v, self.enable_reo_list)
            elif k == _NORMAL_AD_MIN:
                check_param_decimal("advanced_min", v, self.normal_advanced_min_decimal)
            elif k == _NORMAL_AD_MAX:
                check_param_decimal("advanced_max", v, self.normal_advanced_max_decimal)
            elif k == _NORMAL_AD_PERIOD:
                check_param_decimal("advanced_period", v, self.normal_advanced_period_decimal)
            elif k == _NORMAL_AD_DURATION:
                check_param_decimal("advanced_duration", v, self.normal_advanced_duration_decimal)

    def check_delay_custom(self, kwargs: dict):
        for k, v in kwargs.items():
            if k == _CUSTOM_MEAN:
                check_param_decimal("mean_delay", v, self.custom_mean_delay_decimal)
            elif k == _CUSTOM_MIN:
                check_param_decimal("min_delay", v, self.custom_min_delay_decimal)
            elif k == _CUSTOM_MAX:
                check_param_decimal("max_delay", v, self.custom_max_delay_decimal)
            elif k == _CUSTOM_POS:
                check_param_decimal("pos_delta", v, self.custom_positive_delta_decimal)
            elif k == _CUSTOM_NEG:
                check_param_decimal("neg_delta", v, self.custom_negative_delta_decimal)
            elif k == _CUSTOM_SPREAD:
                check_param_decimal("spread", v, self.custom_spread_delta_decimal)
            elif k == _CUSTOM_ENABLE_REO:
                check_param_in_choices("enable_reordering", v, self.enable_reo_list)

    def check_delay_jitter(self, kwargs: dict):
        for k, v in kwargs.items():
            if k == _JITTER_DELAY:
                check_param_decimal("delay", v, self.jitter_mean_delay_decimal)
            elif k == _JITTER_JITTER:
                check_param_decimal("jitter", v, self.jitter_mean_jitter_decimal)
            elif k == _JITTER_REO:
                check_param_in_choices("enable_reordering", v, self.enable_reo_list)


_delay_checker = _DelayLimits()

_CONSTANT_NORMAL_DELAY: str = r"de"


class DelayConstant(Delay):
    """常量时延

    Notes:
        强制关键字传参

    Args:
        delay(float): 时延值

    """

    def __init__(self, *, delay: float):
        super().__init__(_DELAY_CONSTANT_MODE, _CONSTANT_TAG)
        self.impairment_params[_CONSTANT_NORMAL_DELAY] = delay
        self._param_tag2str = {
            _CONSTANT_NORMAL_DELAY: "delay"
        }
        _delay_checker.check_delay_constant(self.impairment_params)
        self._update_param()

    @property
    def delay(self) -> float:
        """时延值
        Notes:
            可读可写

        """
        return self.impairment_params[_CONSTANT_NORMAL_DELAY]

    @delay.setter
    @check_parameter
    def delay(self, value: float) -> None:
        self.impairment_params[_CONSTANT_NORMAL_DELAY] = value
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'DelayConstant':
        """由常量时延的 xml 节点构造 DelayConstant 对象

        Args:
            node: 常量时延的 xml 节点

        Returns:
            DelayConstant 对象
        """
        details = node.find(_CONSTANT_TAG)
        return DelayConstant(delay=float(details.findtext(_CONSTANT_NORMAL_DELAY)))

    def __str__(self):
        res = self.__class__.__name__ + ":{ "
        for k, v in self.impairment_params.items():
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)

        res += "}"
        return res


_UNIFORM_MIN: str = r"dmi"
_UNIFORM_MAX: str = r"dma"
_UNIFORM_ENABLE_REO: str = r"reo"
_UNIFORM_ADVANCE_TAG: str = r"shake"
_UNIFORM_ADVANCE_TYPE: str = r"type_id"
_UNIFORM_ADVANCE_MAX: str = r"max"
_UNIFORM_ADVANCE_MIN: str = r"min"
_UNIFORM_ADVANCE_CYCLE: str = r"cycle"
_UNIFORM_ADVANCE_RISE: str = r"proportion_up"
_UNIFORM_ADVANCE_FALL: str = r"proportion_down"


class DelayUniform(Delay):
    """均匀分布时延模式

    Notes:
        强制关键字传参

    Args:
        minimum(float): 时延的最小值
        maximum(float): 时延的最大值
        enable_reordering(int): 是否允许乱序
        advanced_setup(ChangeMode,optional): 变化模式

    """

    def __init__(self, *, minimum: float, maximum: float, enable_reordering: int, advanced_setup: ChangeMode = None):
        super().__init__(_DELAY_UNIFORM_MODE, _UNIFORM_TAG)
        self.impairment_params[_UNIFORM_MIN] = minimum
        self.impairment_params[_UNIFORM_MAX] = maximum
        self.impairment_params[_UNIFORM_ENABLE_REO] = enable_reordering
        self._param_tag2str = {
            _UNIFORM_MIN: "minimum",
            _UNIFORM_MAX: "maximum",
            _UNIFORM_ENABLE_REO: "enable_reordering"
        }
        _delay_checker.check_delay_uniform(self.impairment_params)
        if advanced_setup == None:
            self._advanced_setup = ChangeMode(mode=0)
        else:
            self._advanced_setup = advanced_setup

        self.add_child(self._advanced_setup)
        self._update_param()

    def set_advanced_setup(self, change_mode: ChangeMode):
        """设置均匀分布时延的高级模式

        Args:
            change_mode: 变化模式

        """
        self._advanced_setup = change_mode
        self._update_param()

    def disable_advanced_setup(self):
        """禁用均匀分布时延的高级模式

        """
        self._advanced_setup = ChangeMode(mode=0)
        self._update_param()

    @property
    def advanced_setup(self) -> ChangeMode:
        return self._advanced_setup

    def _update_param(self) -> None:
        self.clear_children()
        self.add_child(self._construct_details())

    def _construct_details(self) -> ET.Element:
        root = ET.Element(self._detail_tag)
        for k, v in self.impairment_params.items():
            node = ET.Element(k)
            node.text = str(v)
            root.append(node)

        root.append(self._advanced_setup.node)
        return root

    @property
    def min(self) -> float:
        return self.impairment_params[_UNIFORM_MIN]

    @min.setter
    @check_parameter
    def min(self, value: float) -> None:
        self.impairment_params[_UNIFORM_MIN] = value
        self._update_param()

    @property
    def max(self) -> float:
        return self.impairment_params[_UNIFORM_MAX]

    @max.setter
    @check_parameter
    def max(self, value: float) -> None:
        self.impairment_params[_UNIFORM_MAX] = value
        self._update_param()

    @property
    def enable_reordering(self) -> int:
        return self.impairment_params[_UNIFORM_ENABLE_REO]

    @enable_reordering.setter
    @check_parameter
    def enable_reordering(self, value: int) -> None:
        self.impairment_params[_UNIFORM_ENABLE_REO] = value
        self._update_param()

    def __str__(self):
        res = self.__class__.__name__ + ":{ "
        for k, v in self.impairment_params.items():
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)

        res += self._advanced_setup.__str__()

        res += "}"
        return res

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'DelayUniform':
        """由均匀分布时延 xml 节点构造 DelayUniform 对象

        Args:
            node: 均匀分布时延 xml 节点

        Returns:
            DelayUniform 对象
        """
        details = node.find(_UNIFORM_TAG)
        change_mode = ChangeMode.construct_from_node(details.find(_CHANGE_MODE_TAG))
        return DelayUniform(
            minimum=float(details.findtext(_UNIFORM_MIN)), maximum=float(details.findtext(_UNIFORM_MAX)),
            enable_reordering=int(details.findtext(_UNIFORM_ENABLE_REO)), advanced_setup=change_mode
        )


_NORMAL_MIN: str = r"de"
_NORMAL_MEAN: str = r"me"
_NORMAL_STD: str = r"sd"
_NORMAL_REO: str = r"reo"
_NORMAL_AD_PERIOD: str = r"p"
_NORMAL_AD_DURATION: str = r"d"
_NORMAL_AD_MIN: str = r"mi"
_NORMAL_AD_MAX: str = r"ma"
_NORMAL_AD_TAG: str = r"b"
_NORMAL_AD_TYPE: str = r"e"


class DelayNormal(Delay):
    """时延正态分布模式

    Notes:
        强制关键字传参

    Examples:
        >>> normal = DelayNormal(minimum=1.11, mean=55.55, std_deviation=11.11, enable_reordering=1)
        >>> normal.enable_advanced_setup(period=50, min=900.1, max=1000, duration=2)
    """

    def __init__(self, *, minimum: float, mean: float, std_deviation: float, enable_reordering: int):
        super().__init__(_DELAY_NORMAL_MODE, _NORMAL_TAG)
        self.impairment_params[_NORMAL_MIN] = minimum
        self.impairment_params[_NORMAL_MEAN] = mean
        self.impairment_params[_NORMAL_STD] = std_deviation
        self.impairment_params[_NORMAL_REO] = enable_reordering
        self._param_tag2str = {
            _NORMAL_MIN: "minimum",
            _NORMAL_MEAN: "mean",
            _NORMAL_STD: "std_deviation",
            _NORMAL_REO: "enable_reordering",
            _NORMAL_AD_MIN: "min",
            _NORMAL_AD_MAX: "max",
            _NORMAL_AD_PERIOD: "period",
            _NORMAL_AD_DURATION: "duration",
        }
        _delay_checker.check_delay_normal(self.impairment_params)
        self._advanced_params = {}
        self._update_param()

    def _update_param(self) -> None:
        self.clear_children()
        self.add_child(self._construct_details())

    def _construct_details(self) -> ET.Element:
        root = ET.Element(self._detail_tag)
        for k, v in self.impairment_params.items():
            node = ET.Element(k)
            node.text = str(v)
            root.append(node)

        root.append(self._construct_advanced_setup())
        return root

    def _construct_advanced_setup(self):
        as_node = ET.Element(_NORMAL_AD_TAG)
        if len(self._advanced_params) == 0:
            as_node.set(_NORMAL_AD_TYPE, "0")
        else:
            as_node.set(_NORMAL_AD_TYPE, "1")
            for k, v in self._advanced_params.items():
                node = ET.Element(k)
                node.text = str(v)
                as_node.append(node)

        return as_node

    @property
    def minimum(self) -> float:
        return self.impairment_params[_NORMAL_MIN]

    @minimum.setter
    @check_parameter
    def minimum(self, value: float) -> None:
        self.impairment_params[_NORMAL_MIN] = value
        self._update_param()

    @property
    def mean(self) -> float:
        return self.impairment_params[_NORMAL_MEAN]

    @mean.setter
    @check_parameter
    def mean(self, value: float) -> None:
        self.impairment_params[_NORMAL_MEAN] = value
        self._update_param()

    @property
    def std_deviation(self) -> float:
        return self.impairment_params[_NORMAL_STD]

    @std_deviation.setter
    @check_parameter
    def std_deviation(self, value: float) -> None:
        self.impairment_params[_NORMAL_STD] = value
        self._update_param()

    @property
    def enable_reordering(self) -> int:
        return self.impairment_params[_NORMAL_REO]

    @enable_reordering.setter
    @check_parameter
    def enable_reordering(self, value: int) -> None:
        self.impairment_params[_NORMAL_REO] = value
        self._update_param()

    def enable_advanced_setup(self, *, period: int, duration: int, min: float, max: float):
        self._advanced_params[_NORMAL_AD_MIN] = min
        self._advanced_params[_NORMAL_AD_MAX] = max
        self._advanced_params[_NORMAL_AD_PERIOD] = period
        self._advanced_params[_NORMAL_AD_DURATION] = duration
        self.impairment_params[_NORMAL_REO] = 1
        _delay_checker.check_delay_normal(self._advanced_params)
        self._update_param()

    def disable_advanced_setup(self):
        self.impairment_params[_NORMAL_REO] = 1
        self._advanced_params = {}
        self._update_param()

    def __str__(self):
        res = self.__class__.__name__ + ":{ "
        for k, v in self.impairment_params.items():
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)

        res += r"advanced_setup:{ "
        for k, v in self._advanced_params.items():
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)
        res += "}"
        res += "}"
        return res

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'DelayNormal':
        """由时延正态分布模式 xml 节点构造 DelayNormal 对象

        Args:
            node: 时延正态分布模式 xml 节点

        Returns:
            DelayNormal 对象
        """
        details = node.find(_NORMAL_TAG)
        normal = DelayNormal(
            minimum=float(details.findtext(_NORMAL_MIN)), mean=float(details.findtext(_NORMAL_MEAN)),
            std_deviation=float(details.findtext(_NORMAL_STD)), enable_reordering=int(details.findtext(_NORMAL_REO))
        )
        advance_node = details.find(_NORMAL_AD_TAG)
        if int(advance_node.get(_NORMAL_AD_TYPE)) == 1:
            normal.enable_advanced_setup(
                period=int(advance_node.findtext(_NORMAL_AD_PERIOD)),
                duration=int(advance_node.findtext(_NORMAL_AD_DURATION)),
                min=float(advance_node.findtext(_NORMAL_AD_MIN)),
                max=float(advance_node.findtext(_NORMAL_AD_MAX))
            )

        return normal


_CUSTOM_MEAN: str = r"mede"
_CUSTOM_MIN: str = r"mide"
_CUSTOM_MAX: str = r"made"
_CUSTOM_POS: str = r"pd"
_CUSTOM_NEG: str = r"nd"
_CUSTOM_SPREAD: str = r"spd"
_CUSTOM_ENABLE_REO: str = r"reo"


class DelayCustom(Delay):
    """时延自定义分布模式

    Notes；
        强制关键字传参

    Examples:
        >>>cust = DelayCustom(mean_delay=100.0, min_delay=1.0, max_delay=200.0, pos_delta=30.0, neg_delta=30.0,
                           spread=1.59,
                           enable_reordering=1)

    """

    def __init__(self, *, mean_delay: float, min_delay: float, max_delay: float, pos_delta: float, neg_delta: float,
                 spread: float, enable_reordering: int):
        super().__init__(_DELAY_CUSTOM_MODE, _CUSTOM_TAG)
        self.impairment_params[_CUSTOM_MEAN] = mean_delay
        self.impairment_params[_CUSTOM_MIN] = min_delay
        self.impairment_params[_CUSTOM_MAX] = max_delay
        self.impairment_params[_CUSTOM_POS] = pos_delta
        self.impairment_params[_CUSTOM_NEG] = neg_delta
        self.impairment_params[_CUSTOM_SPREAD] = spread
        self.impairment_params[_CUSTOM_ENABLE_REO] = enable_reordering
        self._param_tag2str = {
            _CUSTOM_MEAN: "mean_delay",
            _CUSTOM_MIN: "min_delay",
            _CUSTOM_MAX: "max_delay",
            _CUSTOM_POS: "pos_delta",
            _CUSTOM_NEG: "neg_delta",
            _CUSTOM_SPREAD: "spread",
            _CUSTOM_ENABLE_REO: "enable_reordering"
        }
        _delay_checker.check_delay_custom(self.impairment_params)
        self._update_param()

    @property
    def mean_delay(self) -> float:
        return self.impairment_params[_CUSTOM_MEAN]

    @mean_delay.setter
    @check_parameter
    def mean_delay(self, value: float) -> None:
        self.impairment_params[_CUSTOM_MEAN] = value
        self._update_param()

    @property
    def min_delay(self) -> float:
        return self.impairment_params[_CUSTOM_MIN]

    @min_delay.setter
    @check_parameter
    def min_delay(self, value: float) -> None:
        self.impairment_params[_CUSTOM_MIN] = value
        self._update_param()

    @property
    def max_delay(self) -> float:
        return self.impairment_params[_CUSTOM_MAX]

    @max_delay.setter
    @check_parameter
    def max_delay(self, value: float) -> None:
        self.impairment_params[_CUSTOM_MAX] = value
        self._update_param()

    @property
    def positive_delta(self) -> float:
        return self.impairment_params[_CUSTOM_POS]

    @positive_delta.setter
    @check_parameter
    def positive_delta(self, value: float) -> None:
        self.impairment_params[_CUSTOM_POS] = value
        self._update_param()

    @property
    def negative_delta(self) -> float:
        return self.impairment_params[_CUSTOM_NEG]

    @negative_delta.setter
    @check_parameter
    def negative_delta(self, value: float) -> None:
        self.impairment_params[_CUSTOM_NEG] = value
        self._update_param()

    @property
    def spread(self) -> float:
        return self.impairment_params[_CUSTOM_SPREAD]

    @spread.setter
    @check_parameter
    def spread(self, value: float) -> None:
        self.impairment_params[_CUSTOM_SPREAD] = value
        self._update_param()

    @property
    def enable_reordering(self) -> int:
        return self.impairment_params[_CUSTOM_ENABLE_REO]

    @enable_reordering.setter
    @check_parameter
    def enable_reordering(self, value: int) -> None:
        self.impairment_params[_CUSTOM_ENABLE_REO] = value
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'DelayCustom':
        """由自定义分布时延的 xml 节点构造 DelayCustom 对象

        Args:
            node: 自定义分布时延的 xml 节点

        Returns:
            DelayCustom 对象
        """
        details = node.find(_CUSTOM_TAG)
        return DelayCustom(
            mean_delay=float(details.findtext(_CUSTOM_MEAN)),
            min_delay=float(details.findtext(_CUSTOM_MIN)),
            max_delay=float(details.findtext(_CUSTOM_MAX)),
            pos_delta=float(details.findtext(_CUSTOM_POS)),
            neg_delta=float(details.findtext(_CUSTOM_NEG)),
            spread=float(details.findtext(_CUSTOM_SPREAD)),
            enable_reordering=int(details.findtext(_CUSTOM_ENABLE_REO))
        )


_JITTER_DELAY: str = r"md"
_JITTER_JITTER: str = r"j"
_JITTER_TYPE: str = r"s"
_JITTER_REO: str = r"reo"


class DelayJitter(Delay):
    """抖动时延模式

    Notes:
        强制关键字传参

    Examples:
        >>> jitter = DelayJitter(delay=99.9, jitter=51.1, type=1, enable_reordering=1)

    """

    def __init__(self, *, delay: float, jitter: float, type: int, enable_reordering: int) -> None:
        super().__init__(_DELAY_JITTER_MODE, _JITTER_TAG)
        self.impairment_params[_JITTER_DELAY] = delay
        self.impairment_params[_JITTER_JITTER] = jitter
        self.impairment_params[_JITTER_TYPE] = type
        self.impairment_params[_JITTER_REO] = enable_reordering
        self._param_tag2str = {
            _JITTER_DELAY: "delay",
            _JITTER_JITTER: "jitter",
            _JITTER_TYPE: "type",
            _JITTER_REO: "enable_reordering"
        }
        _delay_checker.check_delay_jitter(self.impairment_params)
        self._update_param()

    @property
    def delay(self) -> float:
        return self.impairment_params[_JITTER_DELAY]

    @delay.setter
    @check_parameter
    def delay(self, value: float) -> None:
        self.impairment_params[_JITTER_DELAY] = value
        self._update_param()

    @property
    def jitter(self) -> float:
        return self.impairment_params[_JITTER_JITTER]

    @jitter.setter
    @check_parameter
    def jitter(self, value: float) -> None:
        self.impairment_params[_JITTER_JITTER] = value
        self._update_param()

    @property
    def type(self) -> int:
        return self.impairment_params[_JITTER_TYPE]

    @type.setter
    @check_parameter
    def type(self, value: float) -> None:
        self.impairment_params[_JITTER_TYPE] = value
        self._update_param()

    @property
    def enable_reordering(self) -> int:
        return self.impairment_params[_JITTER_REO]

    @enable_reordering.setter
    @check_parameter
    def enable_reordering(self, value: int) -> None:
        self.impairment_params[_JITTER_REO] = value
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'DelayJitter':
        details = node.find(_JITTER_TAG)
        return DelayJitter(
            delay=float(details.findtext(_JITTER_DELAY)),
            type=int(details.findtext(_JITTER_TYPE)),
            enable_reordering=int(details.findtext(_JITTER_REO)),
            jitter=float(details.findtext(_JITTER_JITTER))
        )


_GAMMA_SHAPE: str = r"shape"
_GAMMA_SCALE: str = r"scale"
_GAMMA_REO: str = r"reo"


class DelayGamma(Delay):
    """伽马分布时延模式

    Notes:
        强制关键字传参

    Examples:
        >>> gamma = DelayGamma(shape=1.1,scale=2.1,enable_reordering=1)

    """

    def __init__(self, *, shape: float, scale: float, enable_reordering: int):
        super().__init__(_DELAY_GAMMA_MODE, _GAMMA_TAG)
        self.impairment_params[_GAMMA_SHAPE] = shape
        self.impairment_params[_GAMMA_SCALE] = scale
        self.impairment_params[_GAMMA_REO] = enable_reordering
        self._param_tag2str = {
            _GAMMA_SHAPE: "shape",
            _GAMMA_SCALE: "scale",
            _GAMMA_REO: "enable_reordering",
        }
        self._update_param()

    @property
    def shape(self) -> float:
        return self.impairment_params[_GAMMA_SHAPE]

    @shape.setter
    @check_parameter
    def shape(self, value: float) -> None:
        self.impairment_params[_GAMMA_SHAPE] = value
        self._update_param()

    @property
    def scale(self) -> float:
        return self.impairment_params[_GAMMA_SCALE]

    @scale.setter
    @check_parameter
    def scale(self, value: float) -> None:
        self.impairment_params[_GAMMA_SCALE] = value
        self._update_param()

    @property
    def enable_reordering(self) -> int:
        return self.impairment_params[_GAMMA_REO]

    @enable_reordering.setter
    @check_parameter
    def enable_reordering(self, value: int) -> None:
        self.impairment_params[_GAMMA_REO] = value
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'DelayGamma':
        """由 gamma分布时延 xml 节点构造 DelayGamma 对象

        Args:
            node: gamma分布时延 xml 节点

        Returns:
            DelayGamma 对象
        """
        details = node.find(_GAMMA_TAG)
        return DelayGamma(
            shape=float(details.findtext(_GAMMA_SHAPE)),
            scale=float(details.findtext(_GAMMA_SCALE)),
            enable_reordering=int(details.findtext(_GAMMA_REO))
        )


_AB_COUNT: str = r"count"
_AB_DELAY: str = r"delay"
_AB_EXTRA: str = r"extra_delay"


class DelayAccumulateBurst(Delay):
    """累积突发时延模式

    Notes:
        强制关键字传参

    Examples:
        >>> acc = DelayAccumulateBurst(delay_count=100, delay=99.9, extra_delay=50.1)

    """

    def __init__(self, *, delay_count: int, delay: float, extra_delay: float) -> None:
        super().__init__(_DELAY_ACCUMULATE_MODE, _ACCUMULATE_TAG)
        self.impairment_params[_AB_COUNT] = delay_count
        self.impairment_params[_AB_DELAY] = delay
        self.impairment_params[_AB_EXTRA] = extra_delay
        self._param_tag2str = {
            _AB_COUNT: "delay_count",
            _AB_DELAY: "delay",
            _AB_EXTRA: "extra_delay",
        }
        self._update_param()

    @property
    def delay_count(self) -> int:
        return self.impairment_params[_AB_COUNT]

    @delay_count.setter
    @check_parameter
    def delay_count(self, value: int) -> None:
        self.impairment_params[_AB_COUNT] = value
        self._update_param()

    @property
    def delay(self) -> float:
        return self.impairment_params[_AB_DELAY]

    @delay.setter
    @check_parameter
    def delay(self, value: float) -> None:
        self.impairment_params[_AB_DELAY] = value
        self._update_param()

    @property
    def extra_delay(self) -> float:
        return self.impairment_params[_AB_EXTRA]

    @extra_delay.setter
    @check_parameter
    def extra_delay(self, value: float) -> None:
        self.impairment_params[_AB_EXTRA] = value
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'DelayAccumulateBurst':
        """由累积突发时延模式 xml 节点构造 DelayAccumulateBurst 对象

        Args:
            node: 累积突发时延模式 xml 节点

        Returns:
            DelayAccumulateBurst 对象
        """
        details = node.find(_ACCUMULATE_TAG)
        return DelayAccumulateBurst(
            delay_count=int(details.findtext(_AB_COUNT)),
            delay=float(details.findtext(_AB_DELAY)),
            extra_delay=float(details.findtext(_AB_EXTRA))
        )

_CUSTOMIZED_FILENAME:str = r"filename"
_CUSTOMIZED_TYPE:str = r"type"
_CUSTOMIZED_INTERVAL:str = r"interval"

class DelayCustomizedFile(Delay):
    """自定义文件时延模式

    Notes:
        强制关键字传参

    Examples:
        >>> custom = DelayCustomizedFile(filename="delay.txt", type=1, interval=100)

    """

    def __init__(self, *, filename: str, type: int, interval: int = 0):
        super().__init__(_DELAY_CUSTOMIZED_FILE_MODE, _CUSTOMIZED_FILE_TAG)
        self.impairment_params[_CUSTOMIZED_FILENAME] = filename
        self.impairment_params[_CUSTOMIZED_TYPE] = type
        self.impairment_params[_CUSTOMIZED_INTERVAL] = interval
        self._param_tag2str = {
            _CUSTOMIZED_FILENAME: "filename",
            _CUSTOMIZED_TYPE: "type",
            _CUSTOMIZED_INTERVAL: "interval",
        }
        self._update_param()

    @property
    def filename(self) -> str:
        return self.impairment_params[_CUSTOMIZED_FILENAME]

    @filename.setter
    @check_parameter
    def filename(self, value: str) -> None:
        self.impairment_params[_CUSTOMIZED_FILENAME] = value
        self._update_param()

    @property
    def type(self) -> int:
        return self.impairment_params[_CUSTOMIZED_TYPE]

    @type.setter
    @check_parameter
    def type(self, value: int) -> None:
        self.impairment_params[_CUSTOMIZED_TYPE] = value
        self._update_param()

    @property
    def interval(self) -> int:
        return self.impairment_params[_CUSTOMIZED_INTERVAL]

    @interval.setter
    @check_parameter
    def interval(self, value: int) -> None:
        self.impairment_params[_CUSTOMIZED_INTERVAL] = value
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'DelayCustomizedFile':
        details = node.find(_CUSTOMIZED_FILE_TAG)
        return DelayCustomizedFile(
            filename=details.findtext(_CUSTOMIZED_FILENAME),
            type=int(details.findtext(_CUSTOMIZED_TYPE)),
            interval=int(details.findtext(_CUSTOMIZED_INTERVAL))
        )


def delay_factory(node: ET.Element) -> Delay:
    mode = int(node.findtext(_IMPAIRMENT_TYPE))
    if mode == _DELAY_CONSTANT_MODE:
        return DelayConstant.construct_from_node(node)
    elif mode == _DELAY_UNIFORM_MODE:
        return DelayUniform.construct_from_node(node)
    elif mode == _DELAY_NORMAL_MODE:
        return DelayNormal.construct_from_node(node)
    elif mode == _DELAY_CUSTOM_MODE:
        return DelayCustom.construct_from_node(node)
    elif mode == _DELAY_JITTER_MODE:
        return DelayJitter.construct_from_node(node)
    elif mode == _DELAY_GAMMA_MODE:
        return DelayGamma.construct_from_node(node)
    elif mode == _DELAY_ACCUMULATE_MODE:
        return DelayAccumulateBurst.construct_from_node(node)
    elif mode == _DELAY_CUSTOMIZED_FILE_MODE:
        return DelayCustomizedFile.construct_from_node(node)
    else:
        raise ValueError(r"Unknown delay type.")

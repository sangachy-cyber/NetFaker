"""
HoloWAN Network Emulator
Python API
"""

import xml.etree.cElementTree as ET

from holowan.v2._holowan_types import check_parameter
from holowan.v2.engine.path._change_mode import ChangeMode, _CHANGE_MODE_TAG
from holowan.v2.engine.path._impairment_base import Loss, _IMPAIRMENT_TYPE

_LOSS_RANDOM_MODE: int = 1
_LOSS_CYCLE_MODE: int = 2
_LOSS_BURST_MODE: int = 3
_LOSS_GB_MODE: int = 4
_LOSS_JITTER_MODE: int = 5
_LOSS_MARKOV_MODE: int = 6
_LOSS_COUNT_MODE: int = 7

import os
from holowan.v2.utils import my_util as mt
from holowan.v2._holowan_types import check_param_decimal, check_param_in_choices, check_param_in_range


class _LossLimits(object):
    _CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
    _INI_PATH: str = os.path.join(_CURRENT_DIR, r"../../../resources/HoloWAN.ini")
    _INI = mt.open_ini(_INI_PATH)

    random_rate_decimal = _INI.get("loss", "randomRateDecimal")

    burst_probability_decimal = _INI.get("loss", "burstProbabilityDecimal")

    dual_good_state_loss_decimal = _INI.get("loss", "dualGoodStateLossDecimal")
    dual_good_to_bad_probability_decimal = _INI.get("loss", "dualGoodToBadProbabilityDecimal")
    dual_bad_state_loss_decimal = _INI.get("loss", "dualBadStateLossDecimal")
    dual_bad_to_good_probability_decimal = _INI.get("loss", "dualBadToGoodProbabilityDecimal")

    markov_p13_decimal = _INI.get("loss", "markovP13Decimal")
    markov_p31_decimal = _INI.get("loss", "markovP31Decimal")
    markov_p32_decimal = _INI.get("loss", "markovP32Decimal")
    markov_p23_decimal = _INI.get("loss", "markovP23Decimal")
    markov_p14_decimal = _INI.get("loss", "markovP14Decimal")
    markov_p13_max = _INI.get("loss", "markovP13Max")
    markov_p13_min = _INI.get("loss", "markovP13Min")
    markov_p31_max = _INI.get("loss", "markovP31Max")
    markov_p31_min = _INI.get("loss", "markovP31Min")
    markov_p32_max = _INI.get("loss", "markovP32Max")
    markov_p32_min = _INI.get("loss", "markovP32Min")
    markov_p23_max = _INI.get("loss", "markovP23Max")
    markov_p23_min = _INI.get("loss", "markovP23Min")
    markov_p14_max = _INI.get("loss", "markovP14Max")
    markov_p14_min = _INI.get("loss", "markovP14Min")

    _msg = 'Argument {argument} must be {expected!r}, but got {got!r}, value {value!r}'

    def check_loss_random(self, kwargs: dict):
        for k, v in kwargs.items():
            if k == _RANDOM_RATE:
                check_param_decimal("loss_rate", v, self.random_rate_decimal)

    def check_delay_burst(self, kwargs: dict):
        for k, v in kwargs.items():
            if k == _BURST_PRO:
                check_param_decimal("probability", v, self.burst_probability_decimal)

    def check_delay_gilbert_elliott(self, kwargs: dict):
        for k, v in kwargs.items():
            if k == _GB_GOOD_LOSS:
                check_param_decimal("good_state_loss", v, self.dual_good_state_loss_decimal)
            elif k == _GB_GOOD_CHANGE:
                check_param_decimal("good_state_change", v, self.dual_good_to_bad_probability_decimal)
            elif k == _GB_BAD_LOSS:
                check_param_decimal("bad_state_loss", v, self.dual_bad_state_loss_decimal)
            elif k == _GB_BAD_CHANGE:
                check_param_decimal("bad_state_loss", v, self.dual_bad_to_good_probability_decimal)

    def check_loss_markov(self, kwargs: dict):
        for k, v in kwargs.items():
            if k == _MKV_P13:
                check_param_decimal("p13", v, self.markov_p13_decimal)
                check_param_in_range("p13", v, self.markov_p13_min, self.markov_p13_max, True)
            elif k == _MKV_P31:
                check_param_decimal("p31", v, self.markov_p31_decimal)
                check_param_in_range("p31", v, self.markov_p31_min, self.markov_p31_max, True)
            elif k == _MKV_P32:
                check_param_in_choices("p31", v, self.markov_p32_decimal)
                check_param_in_range("p32", v, self.markov_p32_min, self.markov_p32_max, True)
            elif k == _MKV_P23:
                check_param_in_choices("p23", v, self.markov_p23_decimal)
                check_param_in_range("p23", v, self.markov_p23_min, self.markov_p23_max, True)
            elif k == _MKV_P14:
                check_param_in_choices("p14", v, self.markov_p14_decimal)
                check_param_in_range("p14", v, self.markov_p14_min, self.markov_p14_max, True)


_loss_checker = _LossLimits()

_RANDOM_TAG: str = r"ra"
_RANDOM_RATE: str = r"r"


class LossRandom(Loss):
    """随机丢包

    Args:
        loss_rate(float): 丢包概率

    Notes:
        强制关键字传参

    Examples:
        >>> rand = LossRandom(loss_rate=0.001)

    """

    def __init__(self, *, loss_rate: float):
        super().__init__(_LOSS_RANDOM_MODE, _RANDOM_TAG)
        self.impairment_params[_RANDOM_RATE] = loss_rate
        self._param_tag2str = {
            _RANDOM_RATE: "loss_rate"
        }
        self._update_param()

    @property
    def loss_rate(self) -> float:
        return self.impairment_params[_RANDOM_RATE]

    @loss_rate.setter
    @check_parameter
    def loss_rate(self, value: float) -> None:
        self.impairment_params[_RANDOM_RATE] = value
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'LossRandom':
        """由随机丢包 xml 节点构造 LossRandom 对象

        Args:
            node: 随机丢包 xml 节点

        Returns:
            LossRandom 对象
        """
        details = node.find(_RANDOM_TAG)
        return LossRandom(loss_rate=float(details.findtext(_RANDOM_RATE)))


_CYCLE_TAG: str = r"cy"
_CYCLE_PERIOD: str = r"a"
_CYCLE_BURST: str = r"l"


class LossCycle(Loss):
    """周期丢包

    Args:
        period(int): 周期
        burst(int): 丢包数量

    Notes:
        强制关键字传参

    Examples:
        >>> cyc = LossCycle(period=900, burst=10)

    """

    def __init__(self, *, period: int, burst: int):
        super().__init__(_LOSS_CYCLE_MODE, _CYCLE_TAG)
        self.impairment_params[_CYCLE_PERIOD] = period
        self.impairment_params[_CYCLE_BURST] = burst
        self._param_tag2str = {
            _CYCLE_PERIOD: "period",
            _CYCLE_BURST: "burst"
        }
        self._update_param()

    @property
    def period(self) -> int:
        return self.impairment_params[_CYCLE_PERIOD]

    @period.setter
    @check_parameter
    def period(self, value: int) -> None:
        self.impairment_params[_CYCLE_PERIOD] = value
        self._update_param()

    @property
    def burst(self) -> int:
        return self.impairment_params[_CYCLE_BURST]

    @burst.setter
    @check_parameter
    def burst(self, value: int) -> None:
        self.impairment_params[_CYCLE_BURST] = value
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'LossCycle':
        """由周期丢包 xml 节点构造 LossCycle 对象

        Args:
            node: 周期丢包 xml 节点

        Returns:
            LossCycle 对象
        """
        details = node.find(_CYCLE_TAG)
        return LossCycle(
            period=int(details.findtext(_CYCLE_PERIOD)),
            burst=int(details.findtext(_CYCLE_BURST))
        )


_BURST_TAG: str = r"bu"
_BURST_PRO: str = r"l"
_BURST_MIN: str = r"mi"
_BURST_MAX: str = r"ma"


class LossBurst(Loss):
    """突发丢包

    Notes:
        强制关键字传参

    Args:
        probability(float): 概率
        min(int): 当突发丢包发生时，最少丢弃的包个数
        max(int): 当突发丢包发生时，最大丢弃的包个数

    Examples:
        >>> bur = LossBurst(probability=0.01, min=10, max=50)

    """

    def __init__(self, *, probability: float, min: int, max: int):
        super().__init__(_LOSS_BURST_MODE, _BURST_TAG)
        self.impairment_params[_BURST_PRO] = probability
        self.impairment_params[_BURST_MIN] = min
        self.impairment_params[_BURST_MAX] = max
        self._param_tag2str = {
            _BURST_PRO: "probability",
            _BURST_MIN: "min",
            _BURST_MAX: "max"
        }
        self._update_param()

    @property
    def probability(self) -> float:
        return self.impairment_params[_BURST_PRO]

    @probability.setter
    @check_parameter
    def probability(self, value: float) -> None:
        self.impairment_params[_BURST_PRO] = value
        self._update_param()

    @property
    def min(self) -> int:
        return self.impairment_params[_BURST_MIN]

    @min.setter
    @check_parameter
    def min(self, value: int) -> None:
        self.impairment_params[_BURST_MIN] = value
        self._update_param()

    @property
    def max(self) -> int:
        return self.impairment_params[_BURST_MAX]

    @max.setter
    @check_parameter
    def max(self, value: int) -> None:
        self.impairment_params[_BURST_MAX] = value
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'LossBurst':
        details = node.find(_BURST_TAG)
        return LossBurst(
            probability=float(details.findtext(_BURST_PRO)),
            min=int(details.findtext(_BURST_MIN)), max=int(details.findtext(_BURST_MAX))
        )


_GB_TAG: str = r"du"
_GB_GOOD_LOSS: str = r"g"
_GB_GOOD_CHANGE: str = r"gtb"
_GB_BAD_LOSS: str = r"b"
_GB_BAD_CHANGE: str = r"btg"


class LossGilbertElliott(Loss):
    """双通道模式丢包

    Notes:
        强制关键字传参

    Args:
        good_state_loss(float): 良好通道丢包概率
        good_state_change(float): 转换概率
        bad_state_loss(float): 较差通道丢包概率
        bad_state_change(float): 转换概率

    Examples:
        >>> gb = LossGilbertElliott(good_state_loss=0.1, good_state_change=0.2, bad_state_loss=0.3, bad_state_change=0.4)

    """

    def __init__(self, *, good_state_loss: float, good_state_change: float, bad_state_loss: float,
                 bad_state_change: float):
        super().__init__(_LOSS_GB_MODE, _GB_TAG)
        self.impairment_params[_GB_GOOD_LOSS] = good_state_loss
        self.impairment_params[_GB_GOOD_CHANGE] = good_state_change
        self.impairment_params[_GB_BAD_LOSS] = bad_state_loss
        self.impairment_params[_GB_BAD_CHANGE] = bad_state_change
        self._param_tag2str = {
            _GB_GOOD_LOSS: "good_state_loss",
            _GB_GOOD_CHANGE: "good_state_change",
            _GB_BAD_LOSS: "bad_state_loss",
            _GB_BAD_CHANGE: "bad_state_change"
        }
        self._update_param()

    @property
    def good_state_loss(self) -> float:
        return self.impairment_params[_GB_GOOD_LOSS]

    @good_state_loss.setter
    @check_parameter
    def good_state_loss(self, value: float) -> None:
        self.impairment_params[_GB_GOOD_LOSS] = value
        self._update_param()

    @property
    def good_state_change(self) -> float:
        return self.impairment_params[_GB_GOOD_CHANGE]

    @good_state_change.setter
    @check_parameter
    def good_state_change(self, value: float) -> None:
        self.impairment_params[_GB_GOOD_CHANGE] = value
        self._update_param()

    @property
    def bad_state_loss(self) -> float:
        return self.impairment_params[_GB_BAD_LOSS]

    @bad_state_loss.setter
    @check_parameter
    def bad_state_loss(self, value: float) -> None:
        self.impairment_params[_GB_BAD_LOSS] = value
        self._update_param()

    @property
    def bad_state_change(self) -> float:
        return self.impairment_params[_GB_BAD_CHANGE]

    @bad_state_change.setter
    @check_parameter
    def bad_state_change(self, value: float) -> None:
        self.impairment_params[_GB_BAD_CHANGE] = value
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'LossGilbertElliott':
        """由双通道丢包 xml 节点构造 LossGilbertElliott 对象

        Args:
            node: 双通道丢包 xml 节点

        Returns:
            LossGilbertElliott 对象
        """
        details = node.find(_GB_TAG)
        return LossGilbertElliott(
            good_state_loss=float(details.findtext(_GB_GOOD_LOSS)),
            good_state_change=float(details.findtext(_GB_GOOD_CHANGE)),
            bad_state_loss=float(details.findtext(_GB_BAD_LOSS)),
            bad_state_change=float(details.findtext(_GB_BAD_CHANGE))
        )


class LossJitter(Loss):
    """抖动丢包

    Notes:
        强制关键字传参

    Args:
        change_mode(ChangeMode): 变化模式

    Examples:
        >>> jitter = LossJitter(change_mode=ChangeMode(mode=6, max=99.9, min=9.9, section=50, period=100))
    """

    def __init__(self, *, change_mode: ChangeMode) -> None:
        super().__init__(_LOSS_JITTER_MODE, None)
        self.impairment_params = {}
        self._change_mode = change_mode
        self._update_param()

    def _update_param(self) -> None:
        self.clear_children()
        self.add_child(self._change_mode.node)

    @property
    def change_mode(self) -> ChangeMode:
        return self._change_mode

    @change_mode.setter
    @check_parameter
    def change_mode(self, value: ChangeMode) -> None:
        if value.mode == 0:
            raise ValueError("The change mode in bandwidth limitation jitter mode can not be mode 0.")
        else:
            self._change_mode = value
        self._update_param()

    def __str__(self):
        res = self.__class__.__name__ + ":{ "
        res += self._change_mode.__str__()
        res += "}"
        return res

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'LossJitter':
        """由抖动丢包 xml 节点构造 LossJitter 对象

        Args:
            node: 抖动丢包 xml 节点

        Returns:
            LossJitter 对象
        """
        change_mode = ChangeMode.construct_from_node(node.find(_CHANGE_MODE_TAG))
        return LossJitter(change_mode=change_mode)


_MKV_TAG: str = r"mkv"
_MKV_P13: str = r"p13"
_MKV_P31: str = r"p31"
_MKV_P32: str = r"p32"
_MKV_P23: str = r"p23"
_MKV_P14: str = r"p14"


class LossMarkov(Loss):
    """四状态马尔可夫丢包

    Notes:
        强制关键字传参

    Args:
        p13: 状态 1 转换到 3 的概率
        p31: 状态 3 转换到 1 的概率
        p32: 状态 3 转换到 2 的概率
        p23: 状态 2 转换到 3 的概率
        p14: 状态 1 转换到 4 的概率

    """

    def __init__(self, *, p13: float, p31: float, p32: float, p23: float, p14: float):
        super().__init__(_LOSS_MARKOV_MODE, _MKV_TAG)
        self.impairment_params[_MKV_P13] = p13
        self.impairment_params[_MKV_P31] = p31
        self.impairment_params[_MKV_P32] = p32
        self.impairment_params[_MKV_P23] = p23
        self.impairment_params[_MKV_P14] = p14
        self._param_tag2str = {
            _MKV_P13: "p13",
            _MKV_P31: "p31",
            _MKV_P32: "p32",
            _MKV_P23: "p23",
            _MKV_P14: "p14"
        }
        self._update_param()

    # ===========================================================
    @property
    def p13(self) -> float:
        return self.impairment_params[_MKV_P13]

    @p13.setter
    @check_parameter
    def p13(self, value: float) -> None:
        self.impairment_params[_MKV_P13] = value
        self._update_param()

    # ===========================================================
    @property
    def p31(self) -> float:
        return self.impairment_params[_MKV_P31]

    @p31.setter
    @check_parameter
    def p31(self, value: float) -> None:
        self.impairment_params[_MKV_P31] = value
        self._update_param()

    # ===========================================================

    @property
    def p32(self) -> float:
        return self.impairment_params[_MKV_P32]

    @p32.setter
    @check_parameter
    def p32(self, value: float) -> None:
        self.impairment_params[_MKV_P32] = value
        self._update_param()

    # ===========================================================
    @property
    def p23(self) -> float:
        return self.impairment_params[_MKV_P23]

    @p23.setter
    @check_parameter
    def p23(self, value: float) -> None:
        self.impairment_params[_MKV_P23] = value
        self._update_param()

    # ===========================================================
    @property
    def p14(self) -> float:
        return self.impairment_params[_MKV_P14]

    @p14.setter
    @check_parameter
    def p14(self, value: float) -> None:
        self.impairment_params[_MKV_P14] = value
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'LossMarkov':
        """由四状态马尔可夫丢包 xml 节点构造 LossMarkov 对象

        Args:
            node: 四状态马尔可夫丢包 xml 节点

        Returns:
            LossMarkov 对象
        """
        details = node.find(_MKV_TAG)
        return LossMarkov(
            p13=float(details.findtext(_MKV_P13)),
            p31=float(details.findtext(_MKV_P31)),
            p32=float(details.findtext(_MKV_P32)),
            p23=float(details.findtext(_MKV_P23)),
            p14=float(details.findtext(_MKV_P14))
        )

_COUNT_TAG: str = r"count"
_COUNT_START_OFFSET: str = r"start_offset"
_COUNT_LOSS_COUNT: str = r"loss_count"

class LossCount(Loss):
    """计数丢包

    Args:
        start_offset(int): 起始偏移量
        loss_count(int): 丢包次数

    Notes:
        强制关键字传参

    Examples:
        >>> cnt = LossCount(start_offset=100, loss_count=10)

    """

    def __init__(self, *, start_offset: int, loss_count: int):
        super().__init__(_LOSS_COUNT_MODE, _COUNT_TAG)
        self.impairment_params[_COUNT_START_OFFSET] = start_offset
        self.impairment_params[_COUNT_LOSS_COUNT] = loss_count
        self._param_tag2str = {
            _COUNT_START_OFFSET: "start_offset",
            _COUNT_LOSS_COUNT: "loss_count"
        }
        self._update_param()

    @property
    def start_offset(self) -> int:
        return self.impairment_params[_COUNT_START_OFFSET]

    @start_offset.setter
    @check_parameter
    def start_offset(self, value: int) -> None:
        self.impairment_params[_COUNT_START_OFFSET] = value
        self._update_param()

    @property
    def loss_count(self) -> int:
        return self.impairment_params[_COUNT_LOSS_COUNT]

    @loss_count.setter
    @check_parameter
    def loss_count(self, value: int) -> None:
        self.impairment_params[_COUNT_LOSS_COUNT] = value
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'LossCount':
        details = node.find(_COUNT_TAG)
        return LossCount(
            start_offset=int(details.findtext(_COUNT_START_OFFSET)),
            loss_count=int(details.findtext(_COUNT_LOSS_COUNT))
        )



def loss_factory(node: ET.Element) -> Loss:
    mode = int(node.findtext(_IMPAIRMENT_TYPE))
    if mode == _LOSS_RANDOM_MODE:
        return LossRandom.construct_from_node(node)
    elif mode == _LOSS_CYCLE_MODE:
        return LossCycle.construct_from_node(node)
    elif mode == _LOSS_BURST_MODE:
        return LossBurst.construct_from_node(node)
    elif mode == _LOSS_GB_MODE:
        return LossGilbertElliott.construct_from_node(node)
    elif mode == _LOSS_JITTER_MODE:
        return LossJitter.construct_from_node(node)
    elif mode == _LOSS_MARKOV_MODE:
        return LossMarkov.construct_from_node(node)
    elif mode == _LOSS_COUNT_MODE:
        return LossCount.construct_from_node(node)
    else:
        raise ValueError(r"Unknown loss type.")

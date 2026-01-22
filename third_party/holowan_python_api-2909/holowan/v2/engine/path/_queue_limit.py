"""
HoloWAN Network Emulator
Python API
"""

import xml.etree.cElementTree as ET

from holowan.v2._holowan_types import check_parameter
from holowan.v2.engine.path._impairment_base import QueueLimit, _IMPAIRMENT_TYPE

_QL_SIMPLE_MODE: int = 1
_QL_DT_MODE: int = 1
_QL_RED_MODE: int = 2

_QL_SIMPLE_DT_TAG = r"dt"
_QL_SIMPLE_DT_DEPTH = r"qd"
_QL_SIMPLE_DUMMY1 = r"qdt"
_QL_SIMPLE_DUMMY2 = r"qdm"

_QL_DT_UNIT = r"qdt"
_QL_DT_DUMMY = r"qdm"

_QL_RED_TAG: str = r"red"
_QL_RED_WEIGHT: str = r"w"
_QL_RED_MIN: str = r"mith"
_QL_RED_MAX: str = r"math"
_QL_RED_PRO: str = r"madp"

_QUEUE_LIMIT_SIMPLE_PARAM = {
    1: {
        _QL_SIMPLE_DT_DEPTH: 16,
        _QL_DT_UNIT: 2,
        _QL_DT_DUMMY: 0
    },
    2: {
        _QL_SIMPLE_DT_DEPTH: 256,
        _QL_DT_UNIT: 2,
        _QL_DT_DUMMY: 0
    },
    3: {
        _QL_SIMPLE_DT_DEPTH: 250,
        _QL_DT_UNIT: 3,
        _QL_DT_DUMMY: 0
    },
    4: {
        _QL_SIMPLE_DT_DEPTH: 262144,
        _QL_DT_UNIT: 2,
        _QL_DT_DUMMY: 0
    }
}


class QueueLimitSimple(QueueLimit):
    """队列限制

    Notes:
        强制关键字传参

    Args:
        option(int): 简单模式的队列深度选择
        1: 16KB
        2: 256KB
        3: 250ms
        4: 256MB

    Examples:
        >>> qls = QueueLimitSimple(option=1)

    """

    def __init__(self, *, option: int):
        super().__init__(_QL_SIMPLE_MODE, _QL_SIMPLE_DT_TAG)
        self.impairment_params = _QUEUE_LIMIT_SIMPLE_PARAM[option]
        self._option = option
        self._update_param()

    @property
    def option(self) -> int:
        """队列深度

        Notes:
            可读可写

        """
        return self._option

    @option.setter
    @check_parameter
    def option(self, value: int) -> None:
        self._option = value
        self.impairment_params = _QUEUE_LIMIT_SIMPLE_PARAM[value]
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'QueueLimitSimple':
        """由队列深度简单模式 xml 节点构造 QueueLimitSimple 对象

        Args:
            node: 队列深度简单模式 xml 节点

        Returns:
            QueueLimitSimple 对象
        """
        details = node.find(_QL_SIMPLE_DT_TAG)
        depth = int(details.findtext(_QL_SIMPLE_DT_DEPTH))
        unit = int(details.findtext(_QL_DT_UNIT))
        option = 3
        for k, v in _QUEUE_LIMIT_SIMPLE_PARAM.items():
            if v[_QL_SIMPLE_DT_DEPTH] == depth and v[_QL_DT_UNIT] == unit:
                option = k
        return QueueLimitSimple(option=option)

    def __str__(self):
        res = self.__class__.__name__ + r":{ "
        res += "{0}:{1} ".format("option", self._option)
        res += "}"
        return res


class QueueLimitDropTail(QueueLimit):
    """队列深度尾部丢弃模式

    Notes:
        强制关键字传参

    Args:
        depth(int): 队列深度
        unit(int): 队列深度单位

    Examples:
        >>> qldt = QueueLimitDropTail(depth=256, unit=2)

    """

    def __init__(self, *, depth: int, unit: int):
        super().__init__(_QL_DT_MODE, _QL_SIMPLE_DT_TAG)
        self.impairment_params[_QL_SIMPLE_DT_DEPTH] = depth
        self.impairment_params[_QL_DT_UNIT] = unit
        self.impairment_params[_QL_SIMPLE_DUMMY2] = 1
        self._param_tag2str = {
            _QL_SIMPLE_DT_DEPTH: "depth",
            _QL_DT_UNIT: "unit"
        }
        self._update_param()

    @property
    def depth(self) -> int:
        """队列深度

        Notes:
            可读可写

        """
        return self.impairment_params[_QL_SIMPLE_DT_DEPTH]

    @depth.setter
    @check_parameter
    def depth(self, value: int) -> None:
        self.impairment_params[_QL_SIMPLE_DT_DEPTH] = value
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'QueueLimitDropTail':
        """由队列深度尾部丢弃模式 xml 节点构造 QueueLimitDropTail 对象

        Args:
            node: 队列深度尾部丢弃模式 xml 节点

        Returns:
            QueueLimitDropTail 对象
        """
        details = node.find(_QL_SIMPLE_DT_TAG)
        return QueueLimitDropTail(
            depth=int(details.findtext(_QL_SIMPLE_DT_DEPTH)),
            unit=int(details.findtext(_QL_DT_UNIT))
        )

    def __str__(self):
        res = self.__class__.__name__ + ":{ "
        for k, v in self.impairment_params.items():
            if k == _QL_SIMPLE_DUMMY2:
                continue
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)

        res += "}"
        return res


class QueueLimitRED(QueueLimit):
    """队列深度随机早期检测

    Notes:
        强制关键字传参

    Args:
        weight(float): 权重
        min_threshold(int): 最小阈值
        max_threshold(int): 最大阈值
        probability(float): 概率

    Examples:
        >>> red = QueueLimitRED(weight=0.001, min_threshold=8, max_threshold=21, probability=0.3)

    """

    def __init__(self, *, weight: float, min_threshold: int, max_threshold: int, probability: float):
        super().__init__(_QL_RED_MODE, _QL_RED_TAG)
        self.impairment_params[_QL_RED_WEIGHT] = weight
        self.impairment_params[_QL_RED_MIN] = min_threshold
        self.impairment_params[_QL_RED_MAX] = max_threshold
        self.impairment_params[_QL_RED_PRO] = probability
        self._param_tag2str = {
            _QL_RED_WEIGHT: "weight",
            _QL_RED_MIN: "min_threshold",
            _QL_RED_MAX: "max_threshold",
            _QL_RED_PRO: "probability"
        }
        self._update_param()

    @property
    def weight(self) -> float:
        """权重

        Notes:
            可读可写

        """
        return self.impairment_params[_QL_RED_WEIGHT]

    @weight.setter
    @check_parameter
    def weight(self, value: float) -> None:
        self.impairment_params[_QL_RED_WEIGHT] = value
        self._update_param()

    @property
    def min_threshold(self) -> int:
        """最小阈值

        Notes:
            可读可写

        """
        return self.impairment_params[_QL_RED_MIN]

    @min_threshold.setter
    @check_parameter
    def min_threshold(self, value: int) -> None:
        self.impairment_params[_QL_RED_MIN] = value
        self._update_param()

    @property
    def max_threshold(self) -> int:
        """最大阈值

        Notes:
            可读可写

        """
        return self.impairment_params[_QL_RED_MAX]

    @max_threshold.setter
    @check_parameter
    def max_threshold(self, value: int) -> None:
        self.impairment_params[_QL_RED_MAX] = value
        self._update_param()

    @property
    def max_drop_probability(self) -> float:
        """最大丢弃概率

        Notes:
            可读可写

        """
        return self.impairment_params[_QL_RED_PRO]

    @max_drop_probability.setter
    @check_parameter
    def max_drop_probability(self, value: float) -> None:
        self.impairment_params[_QL_RED_PRO] = value
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'QueueLimitRED':
        """由队列深度随机早期检测 xml 节点构造 QueueLimitRED 对象

        Args:
            node: 队列深度随机早期检测 xml 节点

        Returns:
            QueueLimitRED 对象
        """
        details = node.find(_QL_RED_TAG)
        return QueueLimitRED(
            weight=float(details.findtext(_QL_RED_WEIGHT)),
            min_threshold=int(details.findtext(_QL_RED_MIN)),
            max_threshold=int(details.findtext(_QL_RED_MAX)),
            probability=float(details.findtext(_QL_RED_PRO))
        )


def queue_limit_factory(node: ET.Element):
    mode = int(node.findtext(_IMPAIRMENT_TYPE))
    if mode == _QL_RED_MODE:
        return QueueLimitRED.construct_from_node(node)
    elif mode == _QL_DT_MODE:
        return QueueLimitDropTail.construct_from_node(node)
    elif mode == _QL_SIMPLE_MODE:
        return QueueLimitSimple.construct_from_node(node)
    else:
        raise ValueError(r"Unknown queue limit type.")

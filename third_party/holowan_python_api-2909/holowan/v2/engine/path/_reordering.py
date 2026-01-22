"""
HoloWAN Network Emulator
Python API
"""

import xml.etree.cElementTree as ET

from holowan.v2._holowan_types import check_parameter
from holowan.v2.engine.path._change_mode import ChangeMode, _CHANGE_MODE_TAG
from holowan.v2.engine.path._impairment_base import Reordering, _IMPAIRMENT_TYPE

_REORDER_NORMAL_MODE: int = 1
_REORDER_JITTER_MODE: int = 2
_REORDER_CYCLE_MODE: int = 3

_NORMAL_TAG: str = r"no"
_NORMAL_PRO: str = r"p"
_NORMAL_MIN: str = r"dmi"
_NORMAL_MAX: str = r"dma"


class ReorderingNormal(Reordering):
    """报文乱序标准模式

    Notes:
        强制关键字传参

    Args:
        probability(float): 概率
        min_delay(float): 最小时延
        max_delay(float): 最大时延

    Examples:
        >>> normal = ReorderingNormal(probability=5, min_delay=11.1, max_delay=99.9)

    """

    def __init__(self, *, probability: float, min_delay: float, max_delay: float) -> None:
        super().__init__(_REORDER_NORMAL_MODE, _NORMAL_TAG)
        self.impairment_params[_NORMAL_PRO] = probability
        self.impairment_params[_NORMAL_MIN] = min_delay
        self.impairment_params[_NORMAL_MAX] = max_delay
        self._param_tag2str = {
            _NORMAL_PRO: "probability",
            _NORMAL_MIN: "min_delay",
            _NORMAL_MAX: "max_delay",
        }
        self._update_param()

    @property
    def probability(self) -> float:
        """概率

        Notes:
            可读可写

        """
        return self.impairment_params[_NORMAL_PRO]

    @probability.setter
    @check_parameter
    def probability(self, value: float) -> None:
        self.impairment_params[_NORMAL_PRO] = value
        self._update_param()

    @property
    def min_delay(self) -> float:
        """最小时延

        Notes:
            可读可写

        """
        return self.impairment_params[_NORMAL_MIN]

    @min_delay.setter
    @check_parameter
    def min_delay(self, value: float) -> None:
        self.impairment_params[_NORMAL_MIN] = value
        self._update_param()

    @property
    def max_delay(self) -> float:
        """最大时延

        Notes:
            可读可写

        """
        return self.impairment_params[_NORMAL_MAX]

    @max_delay.setter
    @check_parameter
    def max_delay(self, value: float) -> None:
        self.impairment_params[_NORMAL_MAX] = value
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'ReorderingNormal':
        """由报文乱序标准模式 xml 节点构造 ReorderingNormal 对象

        Args:
            node: 报文乱序标准模式 xml 节点

        Returns:
            ReorderingNormal 对象
        """
        details = node.find(_NORMAL_TAG)
        return ReorderingNormal(
            probability=float(details.findtext(_NORMAL_PRO)),
            min_delay=float(details.findtext(_NORMAL_MIN)),
            max_delay=float(details.findtext(_NORMAL_MAX))
        )


_JITTER_TAG: str = r"ji"
_JITTER_MIN: str = r"dmi"
_JITTER_MAX: str = r"dma"


class ReorderingJitter(Reordering):
    """报文乱序抖动模式

    Notes:
        强制关键字传参

    Args:
        min_delay(float): 最小时延
        max_delay(float): 最大时延
        change_mode(ChangeMode): 变化模式

    Examples:
        >>> jitter = ReorderingJitter(min_delay=11.1, max_delay=99.9,
        >>>                          change_mode=ChangeMode(mode=1, max=99.9, min=11.1, phase=60, period=100))

    """

    def __init__(self, *, min_delay: float, max_delay: float, change_mode: ChangeMode) -> None:
        super().__init__(_REORDER_JITTER_MODE, _JITTER_TAG)
        self.impairment_params[_JITTER_MIN] = min_delay
        self.impairment_params[_JITTER_MAX] = max_delay
        self._change_mode = change_mode
        self._param_tag2str = {
            _JITTER_MIN: "min_delay",
            _JITTER_MAX: "max_delay",
        }
        self._update_param()

    def _construct_details(self) -> ET.Element:
        root = ET.Element(self._detail_tag)
        for k, v in self.impairment_params.items():
            node = ET.Element(k)
            node.text = str(v)
            root.append(node)

        root.append(self._change_mode.node)
        return root

    def _update_param(self) -> None:
        self.clear_children()
        self.add_child(self._construct_details())

    @property
    def change_mode(self) -> ChangeMode:
        """变化模式

        Notes:
            可读可写
        """
        return self._change_mode

    @change_mode.setter
    @check_parameter
    def change_mode(self, value: ChangeMode) -> None:
        if value.mode == 0:
            raise ValueError("The change mode in bandwidth limitation jitter mode can not be mode 0.")
        else:
            self._change_mode = value
        self._update_param()

    @property
    def min_delay(self) -> float:
        """最小时延

        Notes:
            可读可写
        """
        return self.impairment_params[_JITTER_MIN]

    @min_delay.setter
    @check_parameter
    def min_delay(self, value: float) -> None:
        self.impairment_params[_JITTER_MIN] = value
        self._update_param()

    @property
    def max_delay(self) -> float:
        """最大时延

        Notes:
            可读可写
        """
        return self.impairment_params[_JITTER_MAX]

    @max_delay.setter
    @check_parameter
    def max_delay(self, value: float) -> None:
        self.impairment_params[_JITTER_MAX] = value
        self._update_param()

    def __str__(self):
        res = self.__class__.__name__ + ":{ "
        for k, v in self.impairment_params.items():
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)
        res += self._change_mode.__str__()
        res += "}"
        return res

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'ReorderingJitter':
        """由报文乱序抖动模式 xml 节点构造 ReorderingJitter 对象

        Args:
            node: 报文乱序抖动模式 xml 节点

        Returns:
            ReorderingJitter 对象
        """
        details = node.find(_JITTER_TAG)
        change_mode = ChangeMode.construct_from_node(details.find(_CHANGE_MODE_TAG))
        return ReorderingJitter(
            min_delay=float(details.findtext(_JITTER_MIN)),
            max_delay=float(details.findtext(_JITTER_MAX)),
            change_mode=change_mode
        )


_CYCLE_TAG: str = r"cy"
_CYCLE_TYPE: str = r"type"
_CYCLE_PERIOD: str = r"period"
_CYCLE_COUNT: str = r"count"
_CYCLE_MIN: str = r"dmi"
_CYCLE_MAX: str = r"dma"


class ReorderingCycle(Reordering):
    """报文乱序周期模式

    Notes:
        强制关键字传参

    Args:
        type(int): 类型
        period(int): 周期
        count(int): 个数
        min_delay(float): 最小时延
        max_delay(float): 最大时延
    """

    def __init__(self, *, type: int, period: int, count: int, min_delay: float, max_delay: float):
        super().__init__(_REORDER_CYCLE_MODE, _CYCLE_TAG)
        self.impairment_params[_CYCLE_TYPE] = type
        self.impairment_params[_CYCLE_PERIOD] = period
        self.impairment_params[_CYCLE_COUNT] = count
        self.impairment_params[_CYCLE_MIN] = min_delay
        self.impairment_params[_CYCLE_MAX] = max_delay
        self._param_tag2str = {
            _CYCLE_TYPE: "type",
            _CYCLE_PERIOD: "period",
            _CYCLE_COUNT: "count",
            _CYCLE_MIN: "min_delay",
            _CYCLE_MAX: "max_delay",
        }
        self._update_param()

    @property
    def type(self) -> int:
        """类型

        Notes:
            可读可写
        """
        return self.impairment_params[_CYCLE_TYPE]

    @type.setter
    @check_parameter
    def type(self, value: int) -> None:
        self.impairment_params[_CYCLE_TYPE] = value
        self._update_param()

    @property
    def period(self) -> int:
        """周期

        Notes:
            可读可写
        """
        return self.impairment_params[_CYCLE_PERIOD]

    @period.setter
    @check_parameter
    def period(self, value: int) -> None:
        self.impairment_params[_CYCLE_PERIOD] = value
        self._update_param()

    @property
    def count_packets(self) -> int:
        """报文个数

        Notes:
            可读可写
        """
        return self.impairment_params[_CYCLE_COUNT]

    @count_packets.setter
    @check_parameter
    def count_packets(self, value: int) -> None:
        self.impairment_params[_CYCLE_COUNT] = value
        self._update_param()

    @property
    def min_delay(self) -> float:
        """最小时延

        Notes:
            可读可写
        """
        return self.impairment_params[_CYCLE_MIN]

    @min_delay.setter
    @check_parameter
    def min_delay(self, value: float) -> None:
        self.impairment_params[_CYCLE_MIN] = value
        self._update_param()

    @property
    def max_delay(self) -> float:
        """最大时延

        Notes:
            可读可写
        """
        return self.impairment_params[_CYCLE_MAX]

    @max_delay.setter
    @check_parameter
    def max_delay(self, value: float) -> None:
        self.impairment_params[_CYCLE_MAX] = value
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'ReorderingCycle':
        """由报文乱序周期模式 xml 节点构造 ReorderingCycle 对象

        Args:
            node: 报文乱序周期模式 xml 节点

        Returns:
            ReorderingCycle 对象
        """
        details = node.find(_CYCLE_TAG)
        return ReorderingCycle(
            type=int(details.findtext(_CYCLE_TYPE)),
            period=int(details.findtext(_CYCLE_PERIOD)),
            count=int(details.findtext(_CYCLE_COUNT)),
            min_delay=float(details.findtext(_CYCLE_MIN)),
            max_delay=float(details.findtext(_CYCLE_MAX))
        )


def reordering_factory(node: ET.Element) -> Reordering:
    mode = int(node.findtext(_IMPAIRMENT_TYPE))
    if mode == _REORDER_NORMAL_MODE:
        return ReorderingNormal.construct_from_node(node)
    elif mode == _REORDER_JITTER_MODE:
        return ReorderingJitter.construct_from_node(node)
    elif mode == _REORDER_CYCLE_MODE:
        return ReorderingCycle.construct_from_node(node)
    else:
        raise ValueError("Unkown reordering type.")

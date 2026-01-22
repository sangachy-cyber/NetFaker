"""
HoloWAN Network Emulator
Python API
"""

import xml.etree.cElementTree as ET

from holowan.v2._holowan_types import check_parameter
from holowan.v2.engine.path._change_mode import ChangeMode, _CHANGE_MODE_TAG
from holowan.v2.engine.path._impairment_base import Duplication, _IMPAIRMENT_TYPE

_DUP_NORMAL_MODE: int = 1
_DUP_JITTER_MODE: int = 2

_NORMAL_PRO: str = r"p"


class DuplicationNormal(Duplication):
    """普通报文重复

    Notes:
        强制关键字传参

    Examples:
        >>> normal = DuplicationNormal(probability=5)
    """

    def __init__(self, *, probability: float):
        super().__init__(_DUP_NORMAL_MODE)
        self.impairment_params[_NORMAL_PRO] = probability
        self._param_tag2str = {
            _NORMAL_PRO: "probability"
        }
        self._update_param()

    @property
    def probability(self) -> float:
        return self.impairment_params[_NORMAL_PRO]

    @probability.setter
    @check_parameter
    def probability(self, value: float) -> None:
        self.impairment_params[_NORMAL_PRO] = value
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'DuplicationNormal':
        """由普通报文重复模式 xml 节点构造 DuplicationNormal 对象

        Args:
            node: 普通报文重复模式 xml 节点

        Returns:
            DuplicationNormal 对象
        """
        return DuplicationNormal(probability=float(node.findtext(_NORMAL_PRO)))


class DuplicationJitter(Duplication):
    """抖动报文重复模式

    Notes:
        强制关键字传参

    Examples:
        >>> jitter = DuplicationJitter(change_mode=ChangeMode(mode=6, max=99.9, min=9.9, section=50, period=100))

    """

    def __init__(self, *, change_mode: ChangeMode):
        super().__init__(_DUP_JITTER_MODE)
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
            raise ValueError("The change mode in duplication jitter mode can not be mode 0.")
        else:
            self._change_mode = value
        self._update_param()

    def __str__(self):
        res = self.__class__.__name__ + ":{ "
        res += self._change_mode.__str__()

        res += "}"
        return res

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'DuplicationJitter':
        """由抖动报文重复模式 xml 节点构造 DuplicationJitter 对象

        Args:
            node: 抖动报文重复模式 xml 节点

        Returns:
            DuplicationJitter 对象

        """
        change_mode = ChangeMode.construct_from_node(node.find(_CHANGE_MODE_TAG))
        return DuplicationJitter(change_mode=change_mode)


def duplication_factory(node: ET.Element) -> Duplication:
    mode = int(node.findtext(_IMPAIRMENT_TYPE))
    _DUP_NORMAL_MODE: int = 1
    _DUP_JITTER_MODE: int = 2
    if mode == _DUP_NORMAL_MODE:
        return DuplicationNormal.construct_from_node(node)
    elif mode == _DUP_JITTER_MODE:
        return DuplicationJitter.construct_from_node(node)
    else:
        raise ValueError("Unkown duplication type.")

"""
HoloWAN Network Emulator
Python API
"""

import xml.etree.cElementTree as ET

from holowan.v2._holowan_types import check_parameter
from holowan.v2.engine.path._impairment_base import FrameOverhead

_FO_24ETHER_MODE: int = 1
_FO_4ETHER_MODE: int = 2
_FO_CUSTOM_MODE: int = 3

_24ETHERNET: int = 24
_4ETHERNET: int = 4

_FO_TYPE: str = r"t"
_FO_RATE: str = r"r"


class FrameOverhead24Ethernet(FrameOverhead):
    """24Ethernet 帧开销

    Examples:
        >>> fo24 = FrameOverhead24Ethernet()
    """

    def __init__(self):
        super().__init__(
            _FO_24ETHER_MODE,
            {
                _FO_TYPE: _FO_24ETHER_MODE,
                _FO_RATE: _24ETHERNET
            }
        )

    @property
    def rate(self) -> int:
        return self.impairment_params[_FO_RATE]

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'FrameOverhead24Ethernet':
        """由 24Ethernet 帧开销 xml 节点构造 FrameOverhead24Ethernet 对象

        Args:
            node: 24Ethernet 帧开销 xml 节点

        Returns:
            FrameOverhead24Ethernet 对象
        """
        return FrameOverhead24Ethernet()


class FrameOverhead4Ethernet(FrameOverhead):
    """4Ethernet 帧开销

    Examples:
        >>> fo4 = FrameOverhead4Ethernet()
    """

    def __init__(self):
        super().__init__(
            _FO_4ETHER_MODE,
            {
                _FO_TYPE: _FO_4ETHER_MODE,
                _FO_RATE: _4ETHERNET
            }
        )

    @property
    def rate(self) -> int:
        return self.impairment_params[_FO_RATE]

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'FrameOverhead4Ethernet':
        """由 4Ethernet 帧开销 xml 节点构造 FrameOverhead4Ethernet 对象

        Args:
            node: 4Ethernet 帧开销 xml 节点

        Returns:
            FrameOverhead4Ethernet 对象
        """
        return FrameOverhead4Ethernet()


class FrameOverheadCustom(FrameOverhead):
    """自定义帧开销

    Notes:
        强制关键字传参

    Examples:
        >>> foc = FrameOverheadCustom(size=64)

    """

    def __init__(self, *, size: int):
        super().__init__(
            _FO_CUSTOM_MODE,
            {
                _FO_TYPE: _FO_CUSTOM_MODE,
                _FO_RATE: size
            }
        )

    @property
    def size(self) -> int:
        return self.impairment_params[_FO_RATE]

    @size.setter
    @check_parameter
    def size(self, value: int) -> None:
        self.impairment_params[_FO_RATE] = value
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'FrameOverheadCustom':
        """由自定义帧开销 xml 节点构造 FrameOverheadCustom 对象

        Args:
            node: 自定义帧开销 xml 节点

        Returns:
            FrameOverheadCustom 对象
        """
        return FrameOverheadCustom(size=int(node.findtext(_FO_RATE)))


def frame_overhead_factory(node: ET.Element) -> FrameOverhead:
    mode = int(node.findtext(_FO_TYPE))
    if mode == _FO_24ETHER_MODE:
        return FrameOverhead24Ethernet.construct_from_node(node)
    elif mode == _FO_4ETHER_MODE:
        return FrameOverhead4Ethernet.construct_from_node(node)
    elif mode == _FO_CUSTOM_MODE:
        return FrameOverheadCustom.construct_from_node(node)
    else:
        raise ValueError(r"Unknown frame overhead type.")

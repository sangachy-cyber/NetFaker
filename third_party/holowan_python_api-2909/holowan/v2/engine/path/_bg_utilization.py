"""
HoloWAN Network Emulator
Python API
"""

import xml.etree.cElementTree as ET

from holowan.v2._holowan_types import check_parameter
from holowan.v2.engine.path._impairment_base import BackgroundUtilization, _IMPAIRMENT_TYPE

_BG_UTILIZATION_DISABLE_MODE: int = 1
_BG_UTILIZATION_RANDOM_MODE: int = 2
_BG_UTILIZATION_PCAP_MODE: int = 3
_BG_RATE_TAG: str = "lu"
_BG_BURST_TAG: str = "bs"
_BG_PCAP_NAME: str = r"pcap_name"


class BackgroundUtilizationDisable(BackgroundUtilization):
    """默认的背景流量模式

    """

    def __init__(self):
        super().__init__(_BG_UTILIZATION_DISABLE_MODE)
        self.impairment_params[_BG_RATE_TAG] = 0
        self.impairment_params[_BG_BURST_TAG] = 60
        self.disable_impair()
        self._update_param()

    def __str__(self):
        return self.__class__.__name__ + r":{}"


class BackgroundUtilizationRandom(BackgroundUtilization):
    """背景流量的随机模式

    Notes:
        强制关键字传参

    Args:
        rate(float): 速率
        burst(int): 突发大小

    """

    @check_parameter
    def __init__(self, *, rate: float, burst: int) -> None:
        super().__init__(_BG_UTILIZATION_RANDOM_MODE)
        self.impairment_params[_BG_RATE_TAG] = rate
        self.impairment_params[_BG_BURST_TAG] = burst
        self._param_tag2str = {
            _BG_RATE_TAG: "rate",
            _BG_BURST_TAG: "burst"
        }
        self._update_param()

    def construct_from_node(node: ET.Element) -> 'BackgroundUtilizationRandom':
        """由一个背景流量的随机模式的 xml 节点构造 BackgroundUtilizationRandom 对象

        Args:
            背景流量的随机模式的 xml 节点

        Returns:
            BackgroundUtilizationRandom 对象

        """
        return BackgroundUtilizationRandom(
            rate=float(node.findtext(_BG_RATE_TAG)),
            burst=int(node.findtext(_BG_BURST_TAG))
        )

    @property
    def rate(self) -> float:
        return self.impairment_params[_BG_RATE_TAG]

    @rate.setter
    def rate(self, value: float):
        self.impairment_params[_BG_RATE_TAG] = value
        self._update_param()

    @property
    def burst(self) -> int:
        return self.impairment_params[_BG_BURST_TAG]

    @burst.setter
    def burst(self, value: int):
        self.impairment_params[_BG_BURST_TAG] = value
        self._update_param()


class BackgroundUtilizationPCAP(BackgroundUtilization):
    """背景流量的PCAP模式

    Notes:
        强制关键字传参

    Args:
        rate(float): 速率
        pcap_name(str): pcap文件名
    """

    @check_parameter
    def __init__(self, *, rate: float, pcap_name: str) -> None:
        super().__init__(_BG_UTILIZATION_PCAP_MODE)
        self.impairment_params[_BG_RATE_TAG] = rate
        self.impairment_params[_BG_PCAP_NAME] = pcap_name
        self._param_tag2str = {
            _BG_RATE_TAG: "rate",
            _BG_PCAP_NAME: "pcap_name"
        }
        self._update_param()

    @property
    def rate(self) -> float:
        return self.impairment_params[_BG_RATE_TAG]

    @rate.setter
    @check_parameter
    def rate(self, value: float) -> None:
        self.impairment_params[_BG_RATE_TAG] = value
        self._update_param()

    @property
    def pcap_name(self) -> str:
        return self.impairment_params[_BG_PCAP_NAME]

    @pcap_name.setter
    @check_parameter
    def pcap_name(self, value: str) -> None:
        self.impairment_params[_BG_PCAP_NAME] = value
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'BackgroundUtilizationPCAP':
        """由一个背景流量PCAP模式的 xml 节点构造 BackgroundUtilizationPCAP 对象

        Args:
            node: 背景流量PCAP模式的 xml 节点

        Returns:
            BackgroundUtilizationPCAP 对象
        """
        return BackgroundUtilizationPCAP(
            rate=float(node.findtext(_BG_RATE_TAG)),
            pcap_name=node.findtext(_BG_PCAP_NAME)
        )


def bg_utilization_factory(node: ET.Element) -> BackgroundUtilization:
    mode = int(node.findtext(_IMPAIRMENT_TYPE))
    if mode == _BG_UTILIZATION_DISABLE_MODE:
        return BackgroundUtilizationDisable()
    elif mode == _BG_UTILIZATION_RANDOM_MODE:
        return BackgroundUtilizationRandom.construct_from_node(node)
    elif mode == _BG_UTILIZATION_PCAP_MODE:
        return BackgroundUtilizationPCAP.construct_from_node(node)
    else:
        raise ValueError(r"Unknown background utilization type.")

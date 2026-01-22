"""
HoloWAN Network Emulator
Python API
"""

import xml.etree.cElementTree as ET

from holowan.v2._holowan_types import check_parameter
from holowan.v2.engine.path._impairment_base import MTU, _IMPAIRMENT_TYPE

_MTU_TYPE: str = r"s"
_MTU_LIMIT: str = r"n"
_MTU_OVERSIZE: str = r"ov"
_MTU_DF: str = r"df"

_MTU_DISABLE_MODE: int = 1
_MTU_ON_MODE: int = 2


class MTUDisable(MTU):
    def __init__(self):
        super().__init__(_MTU_DISABLE_MODE)
        self.impairment_params[_MTU_TYPE] = _MTU_DISABLE_MODE
        self.impairment_params[_MTU_LIMIT] = 1500
        self.disable_impair()
        self._update_param()

    def __str__(self):
        res = self.__class__.__name__ + ":{}"
        return res


class MTULimit(MTU):
    """MTU限制

    Notes:
        强制关键字传参

    Args:
        limit(int): 限制大小
        drop_oversize_packets(int): 是否丢弃超限数据包
        drop_df_packets(int): 是否丢弃DF数据包

    Examples:
        >>> mtu = MTULimit(limit=1499, drop_df_packets=1, drop_oversize_packets=0)

    """

    def __init__(self, *, limit: int, drop_oversize_packets: int, drop_df_packets: int):
        super().__init__(_MTU_ON_MODE)
        if drop_oversize_packets + drop_df_packets > 1:
            raise ValueError(
                "Argument 'drop_oversize_packets' and 'drop_df_packets' can not be value 1 at the same time.")
        self.impairment_params[_MTU_TYPE] = _MTU_ON_MODE
        self.impairment_params[_MTU_LIMIT] = limit
        self.impairment_params[_MTU_OVERSIZE] = drop_oversize_packets
        self.impairment_params[_MTU_DF] = drop_df_packets
        self._param_tag2str = {
            _MTU_LIMIT: "limit",
            _MTU_OVERSIZE: "drop_oversize_packets",
            _MTU_DF: "drop_df_packets"
        }
        self._update_param()

    @property
    def mtu_limit(self) -> int:
        return self.impairment_params[_MTU_LIMIT]

    @mtu_limit.setter
    @check_parameter
    def mtu_limit(self, value: int) -> None:
        self.impairment_params[_MTU_LIMIT] = value
        self._update_param()

    @property
    def drop_oversize_packets(self) -> int:
        return self.impairment_params[_MTU_OVERSIZE]

    @drop_oversize_packets.setter
    @check_parameter
    def drop_oversize_packets(self, value: int) -> None:
        self.impairment_params[_MTU_OVERSIZE] = value
        self._update_param()

    @property
    def drop_df_packets(self) -> int:
        return self.impairment_params[_MTU_OVERSIZE]

    @drop_df_packets.setter
    @check_parameter
    def drop_df_packets(self, value: int) -> None:
        self.impairment_params[_MTU_DF] = value
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'MTULimit':
        """由 MTU 限制 xml 节点构造 MTULimit 对象

        Args:
            node: MTU 限制 xml 节点

        Returns:
            MTULimit 对象
        """
        return MTULimit(
            limit=int(node.findtext(_MTU_LIMIT)),
            drop_oversize_packets=int(node.findtext(_MTU_OVERSIZE)),
            drop_df_packets=int(node.findtext(_MTU_DF))
        )


def mtu_factory(node: ET.Element):
    mode = int(node.findtext(_IMPAIRMENT_TYPE))
    if mode == _MTU_DISABLE_MODE:
        return MTUDisable()
    elif mode == _MTU_ON_MODE:
        return MTULimit.construct_from_node(node)
    else:
        raise ValueError(r"Unknown MTU type.")

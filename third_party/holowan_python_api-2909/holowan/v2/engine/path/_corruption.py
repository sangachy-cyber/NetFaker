"""
HoloWAN Network Emulator
Python API
"""

import xml.etree.cElementTree as ET

from holowan.v2._holowan_types import check_parameter
from holowan.v2.engine.path._impairment_base import Corruption, _IMPAIRMENT_TYPE

_BER_COMMON_MODE: int = 1
_BER_RANGE_MODE: int = 2
_BER_PACKET_MODE: int = 3
_BER_COUNT_MODE: int = 4

_COMMON_RATE: str = r"ber"
_COMMON_INDEX: str = r"beri"
_CRC: str = r"crc"


class BER(Corruption):
    """误码

    Notes:
        强制关键字传参

    Args:
        error_rate(int): 误码率底数
        error_rate_index(int): 误码率指数
        crc(int): 是否重算 CRC

    Examples:
        >>> ber = BER(error_rate=1, error_rate_index=14, crc=1)

    """

    def __init__(self, *, error_rate: int, error_rate_index: int, crc: int = 0):
        # 1u 没有 crc 给个 0 的默认值
        super().__init__(_BER_COMMON_MODE)
        self.impairment_params[_COMMON_RATE] = error_rate
        self.impairment_params[_COMMON_INDEX] = error_rate_index
        self.impairment_params[_CRC] = crc
        self._param_tag2str = {
            _COMMON_RATE: "error_rate",
            _COMMON_INDEX: "error_rate_index",
            _CRC: "crc"
        }
        self._update_param()

    @property
    def bit_error_rate(self) -> int:
        return self.impairment_params[_COMMON_RATE]

    @bit_error_rate.setter
    @check_parameter
    def bit_error_rate(self, value: int) -> None:
        self.impairment_params[_COMMON_RATE] = value
        self._update_param()

    @property
    def bit_error_rate_index(self) -> int:
        return self.impairment_params[_COMMON_INDEX]

    @bit_error_rate_index.setter
    @check_parameter
    def bit_error_rate_index(self, value: int) -> None:
        self.impairment_params[_COMMON_INDEX] = value
        self._update_param()

    @property
    def crc(self) -> int:
        return self.impairment_params[_CRC]

    @crc.setter
    @check_parameter
    def crc(self, value: float) -> None:
        self.impairment_params[_CRC] = value
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'BER':
        """由误码 xml 节点构造 BER 对象

        Args:
            node: 误码 xml 节点

        Returns:
            BER 对象
        """
        crc = node.findtext(_CRC)
        # 1u 是没有 crc 的, 特殊处理一下
        if crc is None:
            crc = 0
        return BER(
            error_rate=int(node.findtext(_COMMON_RATE)),
            error_rate_index=int(node.findtext(_COMMON_INDEX)),
            crc=int(crc)
        )


_RANGE_RATE: str = r"ber"
_RANGE_INDEX: str = r"beri"
_RANGE_RANGES_TAG = r"range"
_RANGE_RANGES_ITEM_TAG: str = r"r"
_RANGE_PRO_START: str = r"start"
_RANGE_PRO_END: str = r"end"


class BERRange(Corruption):
    """范围误码

    Notes:
        强制关键字传参

    Args:
        error_rate(int): 误码率底数
        error_rate_index(int): 误码率指数
        crc(int): 是否重算 CRC
        range_list(list): 范围列表

    Examples:
        >>> range = BERRange(bit_error_rate=1, bit_error_rate_index=14, crc=1, range_list=["1-10", "11-20", "21-30"])

    """

    def __init__(self, *, bit_error_rate: int, bit_error_rate_index: int, crc: int, range_list: list):
        super().__init__(_BER_RANGE_MODE)
        self.impairment_params[_RANGE_RATE] = bit_error_rate
        self.impairment_params[_RANGE_INDEX] = bit_error_rate_index
        if not isinstance(range_list, list) or len(range_list) == 0:
            raise ValueError("Argumment range_list is not valid, got {got!r}, value {value!r}".format(
                got=type(range_list), value=range_list
            ))
        self._range_list = range_list
        self.impairment_params[_CRC] = crc
        self._param_tag2str = {
            _RANGE_RATE: "bit_error_rate",
            _RANGE_INDEX: "bit_error_rate_index",
            _CRC: "crc"
        }
        self._update_param()

    def _construct_ranges(self):
        rg_node = ET.Element(_RANGE_RANGES_TAG)
        for idx, rg in enumerate(self._range_list):
            r = rg.split('-')
            start, end = r[0], r[1]
            node = ET.Element(_RANGE_RANGES_ITEM_TAG + str(idx + 1))
            node.set(_RANGE_PRO_START, str(start))
            node.set(_RANGE_PRO_END, str(end))
            rg_node.append(node)
        return rg_node

    def _update_param(self):
        self.clear_children()
        self.add_children(self.impairment_params)
        self.add_child(self._construct_ranges())

    @property
    def bit_error_rate(self) -> int:
        return self.impairment_params[_RANGE_RATE]

    @bit_error_rate.setter
    @check_parameter
    def bit_error_rate(self, value: int) -> None:
        self.impairment_params[_RANGE_RATE] = value
        self._update_param()

    @property
    def bit_error_rate_index(self) -> int:
        return self.impairment_params[_RANGE_INDEX]

    @bit_error_rate_index.setter
    @check_parameter
    def bit_error_rate_index(self, value: int) -> None:
        self.impairment_params[_RANGE_INDEX] = value
        self._update_param()

    @property
    def crc(self) -> int:
        return self.impairment_params[_CRC]

    @crc.setter
    @check_parameter
    def crc(self, value: float) -> None:
        self.impairment_params[_CRC] = value
        self._update_param()

    def add_range(self, range: str) -> None:
        self._range_list.append(range)
        self._update_param()

    def remove_range(self, index: int) -> None:
        del self._range_list[index]
        self._update_param()

    def insert_range(self, index: int, range: str) -> None:
        self._range_list.insert(index, range)
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'BERRange':
        """由范围误码 xml 节点构造 BERRange 对象

        Args:
            node: 范围误码 xml 节点

        Returns:
            BERRange 对象
        """
        range_list = []
        for r in list(node.find(_RANGE_RANGES_TAG)):
            start = r.get(_RANGE_PRO_START)
            end = r.get(_RANGE_PRO_END)
            range_list.append(str(start) + "-" + str(end))

        return BERRange(
            bit_error_rate=int(node.findtext(_RANGE_RATE)),
            bit_error_rate_index=int(node.findtext(_RANGE_INDEX)),
            crc=int(node.findtext(_CRC)),
            range_list=range_list
        )

    def __str__(self):
        res = self.__class__.__name__ + ":{ "
        for k, v in self.impairment_params.items():
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)

        res += r"range_list: " + str(self._range_list)
        res += "}"
        return res


_PACKET_RATE: str = r"nb"
_PACKET_PRO: str = r"prob"


class BERPacket(Corruption):
    """包误码

    Notes:
        强制关键字传参

    Args:
        ber_per_packet(int): 误码数
        probability(int): 误码概率
        crc(int): 是否重算 CRC

    Examples:
        >>> pack = BERPacket(ber_per_packet=10, probability=5, crc=1)

    """

    def __init__(self, *, ber_per_packet: int, probability: float, crc: int):
        super().__init__(_BER_PACKET_MODE)
        self.impairment_params[_PACKET_RATE] = ber_per_packet
        self.impairment_params[_PACKET_PRO] = probability
        self.impairment_params[_CRC] = crc
        self._param_tag2str = {
            _PACKET_RATE: "ber_per_packet",
            _PACKET_PRO: "probability",
            _CRC: "crc"
        }
        self._update_param()

    @property
    def ber_per_packet(self) -> int:
        return self.impairment_params[_PACKET_RATE]

    @ber_per_packet.setter
    @check_parameter
    def ber_per_packet(self, value: int) -> None:
        self.impairment_params[_PACKET_RATE] = value
        self._update_param()

    @property
    def probability(self) -> int:
        return self.impairment_params[_PACKET_PRO]

    @probability.setter
    @check_parameter
    def probability(self, value: int) -> None:
        self.impairment_params[_PACKET_PRO] = value
        self._update_param()

    @property
    def crc(self) -> int:
        return self.impairment_params[_CRC]

    @crc.setter
    @check_parameter
    def crc(self, value: float) -> None:
        self.impairment_params[_CRC] = value
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'BERPacket':
        """由包误码 xml 节点构造 BERPacket 对象

        Args:
            node: 包误码 xml 节点

        Returns:
            BERPacket 对象
        """
        return BERPacket(
            ber_per_packet=int(node.findtext(_PACKET_RATE)),
            probability=float(node.findtext(_PACKET_PRO)),
            crc=int(node.findtext(_CRC))
        )

_COUNT_BER_PER_PACKET:str = r"nb"
_COUNT_START_OFFSET:str = r"start_offset"
_COUNT_BER_COUNT:str = r"ber_count"


class BERCount(Corruption):
    """计数误码

    Notes:
        强制关键字传参

    Args:
        ber_per_packet(int): 误码数
        start_offset(int): 起始偏移量
        ber_count(int): 误码次数

    Examples:
        >>> cnt = BERCount(ber_per_packet=10, start_offset=100, ber_count=5, crc=1)

    """

    def __init__(self, *, ber_per_packet: int, start_offset: int, ber_count: int, crc: int):
        super().__init__(_BER_COUNT_MODE)
        self.impairment_params[_COUNT_BER_PER_PACKET] = ber_per_packet
        self.impairment_params[_COUNT_START_OFFSET] = start_offset
        self.impairment_params[_COUNT_BER_COUNT] = ber_count
        self.impairment_params[_CRC] = crc
        self._param_tag2str = {
            _PACKET_RATE: "ber_per_packet",
            _COUNT_START_OFFSET: "start_offset",
            _COUNT_BER_COUNT: "ber_count",
            _CRC: "crc"
        }
        self._update_param()

    @property
    def ber_per_packet(self) -> int:
        return self.impairment_params[_COUNT_BER_PER_PACKET]

    @ber_per_packet.setter
    @check_parameter
    def ber_per_packet(self, value: int) -> None:
        self.impairment_params[_COUNT_BER_PER_PACKET] = value
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
    def ber_count(self) -> int:
        return self.impairment_params[_COUNT_BER_COUNT]

    @ber_count.setter
    @check_parameter
    def ber_count(self, value: int) -> None:
        self.impairment_params[_COUNT_BER_COUNT] = value
        self._update_param()

    @property
    def crc(self) -> int:
        return self.impairment_params[_CRC]

    @crc.setter
    @check_parameter
    def crc(self, value: float) -> None:
        self.impairment_params[_CRC] = value
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'BERCount':
        """由计数误码 xml 节点构造 BERCount 对象

        Args:
            node: 计数误码 xml 节点

        Returns:
            BERCount 对象
        """
        return BERCount(
            ber_per_packet=int(node.findtext(_COUNT_BER_PER_PACKET)),
            start_offset=int(node.findtext(_COUNT_START_OFFSET)),
            ber_count=int(node.findtext(_COUNT_BER_COUNT)),
            crc=int(node.findtext(_CRC))
        )


def corruption_factory(node: ET.Element) -> Corruption:
    mode = int(node.findtext(_IMPAIRMENT_TYPE))
    if mode == _BER_COMMON_MODE:
        return BER.construct_from_node(node)
    elif mode == _BER_RANGE_MODE:
        return BERRange.construct_from_node(node)
    elif mode == _BER_PACKET_MODE:
        return BERPacket.construct_from_node(node)
    elif mode == _BER_COUNT_MODE:
        return BERCount.construct_from_node(node)
    else:
        raise ValueError(r"Unknown corruption type.")



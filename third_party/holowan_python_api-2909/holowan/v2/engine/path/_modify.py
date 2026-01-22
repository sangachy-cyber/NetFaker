"""
HoloWAN Network Emulator
Python API
"""

import xml.etree.cElementTree as ET
from typing import overload

from holowan.v2._holowan_types import check_parameter
from holowan.v2.engine.path._impairment_base import Modify

_MODIFY_TYPE_TAG: str = r"cs"
_MODIFY_SWITCH_TAG: str = r"fs"

_MODIFY_DISABLE_MODE: int = 0
_MODIFY_NORMAL_MODE: int = 1
_MODIFY_CYCLE_MODE: int = 2
_MODIFY_RANDOM_MODE: int = 3
_MODIFY_RANGES_MODE: int = 4
_MODIFY_INSERT_MODE: int = 5
_MODIFY_EXCHANGE_MODE: int = 6
_MODIFY_COUNT_MODE: int = 7
_MODIFY_DELETE_MODE: int = 8

_MODIFY_CRC: str = r"crc"
_MODIFY_CHECKSUM: str = r"checksum"

_MATCH_HEADER: str = r"pt"
_MATCH_OFFSET: str = r"po"
_MATCH_SIZE: str = r"ps"
_MATCH_VALUE: str = r"pv"

_MODIFY_HEADER: str = r"mt"
_MODIFY_OFFSET: str = r"mo"
_MODIFY_SIZE: str = r"ms"
_MODIFY_VALUE: str = r"mv"

_MODIFY_CYCLE_PERIOD: str = r"mcp"
_MODIFY_CYCLE_BURST: str = r"mcb"

_MODIFY_RANDOM_RATE: str = r"mrr"

"""
<ra>
    <nb>2</nb>
    <range0>
      <offset>0</offset>
      <size>1</size>
      <value>ff</value>
    </range0>
    <range1>
      <offset>1</offset>
      <size>1</size>
      <value>ff</value>
    </range1>
</ra>
"""
_MODIFY_RANGES: str = r"ra"
_MODIFY_RANGES_COUNT: str = r"nb"
_MODIFY_RANGES_RANGE: str = r"range"
_MODIFY_RANGES_RANGE_OFFSET: str = "offset"
_MODIFY_RANGES_RANGE_SIZE: str = "size"
_MODIFY_RANGES_RANGE_SIZE_CONSTANT: int = 1
_MODIFY_RANGES_RANGE_VALUE: str = "value"

_MODIFY_INSERT_OFFSET: str = "mo"
_MODIFY_INSERT_VALUE: str = r"ins"

_MODIFY_DELETE: str = "de"
_MODIFY_DELETE_OFFSET: str = "offset"
_MODIFY_DELETE_SIZE: str = "size"

_MODIFY_EXCHANGE: str = "ex"
_MODIFY_EXCHANGE_PREVIOUS: str = "prev"
_MODIFY_EXCHANGE_NEXT: str = "next"

_MODIFY_COUNT: str = "ra"
_MODIFY_COUNT_OFFSET: str = "offset"
_MODIFY_COUNT_SIZE_CONSTANT: int = 1
_MODIFY_COUNT_VALUE: str = "value"
_MODIFY_COUNT_START_OFFSET: str = "start_offset"
_MODIFY_COUNT_MODIFY_COUNT: str = "modify_count"

_modify_tag2str_dict = {
    _MATCH_HEADER: "match_header",
    _MATCH_OFFSET: "match_offset",
    _MATCH_SIZE: "match_size",
    _MATCH_VALUE: "match_value",
    _MODIFY_HEADER: "modify_header",
    _MODIFY_OFFSET: "modify_offset",
    _MODIFY_SIZE: "modify_size",
    _MODIFY_VALUE: "modify_value",
    _MODIFY_CRC: "crc",
    _MODIFY_CHECKSUM: "checksum",
    _MODIFY_CYCLE_PERIOD: "cycle_period",
    _MODIFY_CYCLE_BURST: "cycle_burst",
    _MODIFY_RANDOM_RATE: "random_rate",
    _MODIFY_SWITCH_TAG: "modify_switch",
    _MODIFY_RANGES: "modify_ranges",
    _MODIFY_INSERT_OFFSET: "insert_offset",
    _MODIFY_INSERT_VALUE: "insert_value",
    _MODIFY_DELETE: "modify_delete",
}

import os
from holowan.v2.utils import my_util as mt
from holowan.v2._holowan_types import check_param_decimal, check_param_in_choices


class _ModifyLimits(object):
    _CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
    _INI_PATH: str = os.path.join(_CURRENT_DIR, r"../../../resources/HoloWAN.ini")
    _INI = mt.open_ini(_INI_PATH)

    match_header_list = _INI.get("modify", "matchType").split(", ")
    modify_header_list = list(_INI.get("modify", "modifyType").split(", "))
    modify_random_rate_decimal = _INI.get("modify", "modifyRandomRateDecimal")
    match_size_list = [1, 2, 4, 6, 8]
    modify_size_list = [1, 2, 4, 6, 8]
    crc_list = [0, 1]
    check_sum_list = [0, 1]

    _msg = 'Argument {argument} must be {expected!r}, but got {got!r}, value {value!r}'

    def valid_match_header(self, value):
        check_param_in_choices("match_header", str(value), self.match_header_list)

    def valid_modify_header(self, value):
        check_param_in_choices("modify_header", str(value), self.modify_header_list)

    def valid_random_rate(self, value):
        check_param_decimal("random_rate", value, self.modify_random_rate_decimal)

    def valid_match_size(self, value):
        check_param_in_choices("match_size", value, self.match_size_list)

    def valid_modify_size(self, value):
        check_param_in_choices("modify_size", value, self.modify_size_list)

    def valid_hex_value(self, value: str, size, name):
        if isinstance(value, str):
            if value.find("0x") != -1:
                hvalue = value.split("0x")[-1]
                st = "0x" + "0" * size * 2
                en = "0x" + "F" * size * 2
                if not int(st, 16) <= int(hvalue, 16) <= int(en, 16):
                    raise ValueError(
                        r"Argument {argument!r} is not valid {expect!r}, it must be in range {start!r}-{end!r}, got {got!r}".format(
                            argument=name, expect="hex value", start=st, end=en,
                            got=value
                        )
                    )
        else:
            raise ValueError(
                self._msg.format(
                    argument=name, expect=str,
                    got=type(value), value=value
                )
            )

    def valid_crc(self, value):
        check_param_in_choices("crc", value, self.crc_list)

    def valid_check_sum(self, value):
        check_param_in_choices("checksum", value, self.check_sum_list)

    def check(self, kwargs: dict):
        for k, v in kwargs.items():
            if k == _MODIFY_TYPE_TAG:
                continue
            key = _modify_tag2str_dict[k]
            if key == "match_header":
                self.valid_match_header(v)
            elif key == "modify_header":
                self.valid_modify_header(v)
            elif key == "random_rate":
                self.valid_random_rate(v)
            elif key == "match_size":
                self.valid_match_size(v)
            elif key == "modify_size":
                self.valid_modify_size(v)
            elif key == "match_value":
                self.valid_match_size(kwargs[_MATCH_SIZE])
                self.valid_hex_value(v, kwargs[_MATCH_SIZE], key)
            elif key == "modify_value":
                self.valid_modify_size(kwargs[_MODIFY_SIZE])
                self.valid_hex_value(v, kwargs[_MODIFY_SIZE], key)
            elif key == "crc":
                self.valid_crc(v)
            elif key == "checksum":
                self.valid_check_sum(v)


_modify_checker = _ModifyLimits()


class ModifyDisable(Modify):
    def __init__(self):
        super().__init__(_MODIFY_DISABLE_MODE)
        self.impairment_params[_MODIFY_TYPE_TAG] = _MODIFY_DISABLE_MODE
        self.disable_impair()
        self._update_params()

    def __str__(self):
        res = self.__class__.__name__ + ":{}"
        return res


class ModifyNormal(Modify):
    """普通报文修改

    Notes:
        强制关键字传参



    """

    @overload
    def __init__(self, *,
                 match_header: int, match_offset: int, match_size: int, match_value: str,
                 modify_header: int, modify_offset: int, modify_size: int, modify_value: str,
                 checksum: int, crc: int = 0, match_switch: int = 1):
        """

        Args:
            match_header: 匹配报文头
            match_offset: 匹配偏移量
            match_size: 匹配大小
            match_value: 匹配值
            modify_header: 修改头
            modify_offset: 修改偏移量
            modify_size: 修改大小
            modify_value: 修改值
            crc: 是否重算 CRC
            checksum: 是否重算 checksum

        Examples:
            Examples:
        >>> normal = ModifyNormal(match_header=1, match_offset=0, match_size=1, match_value="0x00", modify_header=1,
        >>>                      modify_offset=0, modify_size=1, modify_value="0xFF", crc=1, checksum=1,match_switch = 1)
        """
        ...

    @overload
    def __init__(self, match_params: dict, modify_params: dict, checksum: int, crc: int = 0, match_switch: int = 1):
        """

        Args:
            match_params: 匹配参数集
            modify_params: 修改参数集
            crc: 是否重算 CRC
            checksum: 是否重算 checksum

        Examples:
            >>> match_params = {
            >>>    "match_header":1, "match_offset":0, "match_size":1, "match_value":"0x0",
            >>> }
            >>> modify_params = {
            >>>    "modify_header":1, "modify_offset":0, "modify_size":1, "modify_value":"0xF"
            >>> }
            >>> normal = ModifyNormal(match_params = match_params,crc=1, modify_params=modify_params,checksum=1)
        """
        ...

    def __init__(self, **kwargs):
        super().__init__(_MODIFY_NORMAL_MODE)
        self.impairment_params[_MODIFY_TYPE_TAG] = _MODIFY_NORMAL_MODE
        match_switch = kwargs.get("match_switch")
        if match_switch is None:
            match_switch = 1
        self.impairment_params[_MODIFY_SWITCH_TAG] = match_switch
        # using default "match_params" first
        self.impairment_params[_MATCH_HEADER] = 1
        self.impairment_params[_MATCH_OFFSET] = 0
        self.impairment_params[_MATCH_SIZE] = 1
        self.impairment_params[_MATCH_VALUE] = "0x0"

        if kwargs.get("modify_params") is not None:
            # in this branch "modify_params" can not be None
            modify_params = kwargs["modify_params"]

            if match_switch == 1:
                # if "match_switch" is 1 and got an invalid "match_params"
                # there will be a key error
                match_params = kwargs["match_params"]
                self.impairment_params[_MATCH_HEADER] = match_params["match_header"]
                self.impairment_params[_MATCH_OFFSET] = match_params["match_offset"]
                self.impairment_params[_MATCH_SIZE] = match_params["match_size"]
                self.impairment_params[_MATCH_VALUE] = match_params["match_value"]

            self.impairment_params[_MODIFY_HEADER] = modify_params["modify_header"]
            self.impairment_params[_MODIFY_OFFSET] = modify_params["modify_offset"]
            self.impairment_params[_MODIFY_SIZE] = modify_params["modify_size"]
            self.impairment_params[_MODIFY_VALUE] = modify_params["modify_value"]
            crc = kwargs.get("crc")
            if crc is not None:
                self.impairment_params[_MODIFY_CRC] = crc
            self.impairment_params[_MODIFY_CHECKSUM] = kwargs["checksum"]
        else:
            if match_switch == 1:
                self.impairment_params[_MATCH_HEADER] = kwargs["match_header"]
                self.impairment_params[_MATCH_OFFSET] = kwargs["match_offset"]
                self.impairment_params[_MATCH_SIZE] = kwargs["match_size"]
                self.impairment_params[_MATCH_VALUE] = kwargs["match_value"]

            self.impairment_params[_MODIFY_HEADER] = kwargs["modify_header"]
            self.impairment_params[_MODIFY_OFFSET] = kwargs["modify_offset"]
            self.impairment_params[_MODIFY_SIZE] = kwargs["modify_size"]
            self.impairment_params[_MODIFY_VALUE] = kwargs["modify_value"]
            crc = kwargs.get("crc")
            if crc is not None:
                self.impairment_params[_MODIFY_CRC] = crc
            self.impairment_params[_MODIFY_CHECKSUM] = kwargs["checksum"]

        _modify_checker.check(self.impairment_params)
        self._param_tag2str = _modify_tag2str_dict
        self._update_params()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'ModifyNormal':
        """由普通修改 xml 节点构造 ModifyNormal 对象

        Args:
            node: 普通修改 xml 节点

        Returns:
            ModifyNormal 对象
        """
        return ModifyNormal(
            match_header=int(node.findtext(_MATCH_HEADER)),
            match_offset=int(node.findtext(_MATCH_OFFSET)),
            match_size=int(node.findtext(_MATCH_SIZE)),
            match_value=node.findtext(_MATCH_VALUE),
            modify_header=int(node.findtext(_MODIFY_HEADER)),
            modify_offset=int(node.findtext(_MODIFY_OFFSET)),
            modify_size=int(node.findtext(_MODIFY_SIZE)),
            modify_value=node.findtext(_MODIFY_VALUE),
            crc=int(node.findtext(_MODIFY_CRC)),
            checksum=int(node.findtext(_MODIFY_CHECKSUM))
        )


class ModifyCycle(Modify):
    """周期修改

    Notes:
        强制关键字传参

    """

    @overload
    def __init__(self, *,
                 match_header: int, match_offset: int, match_size: int, match_value: str,
                 modify_header: int, modify_offset: int, modify_size: int, modify_value: str,
                 cycle_period: int, cycle_burst: int,
                 checksum: int, crc: int = 0, match_switch: int = 1):
        """

        Args:
            match_header: 匹配报文头
            match_offset: 匹配偏移量
            match_size: 匹配大小
            match_value: 匹配值
            modify_header: 修改头
            modify_offset: 修改偏移量
            modify_size: 修改大小
            modify_value: 修改值
            cycle_period: 周期
            cycle_burst: 周期突发
            crc: 是否重算 CRC
            checksum: 是否重算 checksum

        Examples:
            >>> cycle = ModifyCycle(match_header=1, match_offset=0, match_size=1, match_value="0x0", modify_header=1,
            >>>                modify_offset=0, modify_size=1, modify_value="0xF",
            >>>                cycle_period=1000, cycle_burst=100,
            >>>                crc=1, checksum=1)

        """
        ...

    @overload
    def __init__(self, match_params: dict, modify_params: dict, checksum: int, crc: int = 0, match_switch: int = 1):
        """

        Args:
            match_params: 匹配参数集
            modify_params: 修改参数集
            crc: 是否重算 CRC
            checksum: 是否重算 checksum

        Examples:
            >>> match_params = {
            >>>    "match_header":1, "match_offset":0, "match_size":1, "match_value":"0x0",
            >>> }
            >>> modify_params = {
            >>>    "modify_header":1, "modify_offset":0, "modify_size":1, "modify_value":"0xF",
            >>>    "cycle_period":1000, "cycle_burst":100,
            >>> }
            >>> cycle = ModifyCycle(match_params = match_params,modify_params=modify_params,crc=1, checksum=1)
        """
        ...

    def __init__(self, **kwargs):
        super().__init__(_MODIFY_CYCLE_MODE)
        self.impairment_params[_MODIFY_TYPE_TAG] = _MODIFY_CYCLE_MODE
        match_switch = kwargs.get("match_switch")
        if match_switch is None:
            match_switch = 1
        self.impairment_params[_MODIFY_SWITCH_TAG] = match_switch
        # using default "match_params" first
        self.impairment_params[_MATCH_HEADER] = 1
        self.impairment_params[_MATCH_OFFSET] = 0
        self.impairment_params[_MATCH_SIZE] = 1
        self.impairment_params[_MATCH_VALUE] = "0x0"
        if kwargs.get("modify_params") is not None:
            modify_params = kwargs["modify_params"]
            if match_switch == 1:
                match_params = kwargs["match_params"]
                self.impairment_params[_MATCH_HEADER] = match_params["match_header"]
                self.impairment_params[_MATCH_OFFSET] = match_params["match_offset"]
                self.impairment_params[_MATCH_SIZE] = match_params["match_size"]
                self.impairment_params[_MATCH_VALUE] = match_params["match_value"]

            self.impairment_params[_MODIFY_HEADER] = modify_params["modify_header"]
            self.impairment_params[_MODIFY_OFFSET] = modify_params["modify_offset"]
            self.impairment_params[_MODIFY_SIZE] = modify_params["modify_size"]
            self.impairment_params[_MODIFY_VALUE] = modify_params["modify_value"]
            self.impairment_params[_MODIFY_CYCLE_PERIOD] = modify_params["cycle_period"]
            self.impairment_params[_MODIFY_CYCLE_BURST] = modify_params["cycle_burst"]
            crc = kwargs.get("crc")
            if crc is not None:
                self.impairment_params[_MODIFY_CRC] = crc
            self.impairment_params[_MODIFY_CHECKSUM] = kwargs["checksum"]
        else:
            self.impairment_params[_MATCH_HEADER] = kwargs["match_header"]
            self.impairment_params[_MATCH_OFFSET] = kwargs["match_offset"]
            self.impairment_params[_MATCH_SIZE] = kwargs["match_size"]
            self.impairment_params[_MATCH_VALUE] = kwargs["match_value"]
            self.impairment_params[_MODIFY_HEADER] = kwargs["modify_header"]
            self.impairment_params[_MODIFY_OFFSET] = kwargs["modify_offset"]
            self.impairment_params[_MODIFY_SIZE] = kwargs["modify_size"]
            self.impairment_params[_MODIFY_VALUE] = kwargs["modify_value"]
            self.impairment_params[_MODIFY_CYCLE_PERIOD] = kwargs["cycle_period"]
            self.impairment_params[_MODIFY_CYCLE_BURST] = kwargs["cycle_burst"]
            crc = kwargs.get("crc")
            if crc is not None:
                self.impairment_params[_MODIFY_CRC] = crc
            self.impairment_params[_MODIFY_CHECKSUM] = kwargs["checksum"]

        _modify_checker.check(self.impairment_params)
        self._param_tag2str = _modify_tag2str_dict
        self._update_params()

    # ===========================================================
    @property
    def cycle_period(self) -> int:
        return self.impairment_params[_MODIFY_CYCLE_PERIOD]

    @cycle_period.setter
    @check_parameter
    def cycle_period(self, value: int) -> None:
        self.impairment_params[_MODIFY_CYCLE_PERIOD] = value
        self._update_params()

    # ===========================================================
    @property
    def cycle_burst(self) -> int:
        return self.impairment_params[_MODIFY_CYCLE_BURST]

    @cycle_burst.setter
    @check_parameter
    def cycle_burst(self, value: int) -> None:
        self.impairment_params[_MODIFY_CYCLE_BURST] = value
        self._update_params()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'ModifyCycle':
        """由周期修改 xml 节点构造 ModifyCycle 对象

        Args:
            node: 周期修改 xml 节点

        Returns:
            ModifyCycle 对象
        """
        return ModifyCycle(
            match_header=int(node.findtext(_MATCH_HEADER)),
            match_offset=int(node.findtext(_MATCH_OFFSET)),
            match_size=int(node.findtext(_MATCH_SIZE)),
            match_value=node.findtext(_MATCH_VALUE),
            modify_header=int(node.findtext(_MODIFY_HEADER)),
            modify_offset=int(node.findtext(_MODIFY_OFFSET)),
            modify_size=int(node.findtext(_MODIFY_SIZE)),
            modify_value=node.findtext(_MODIFY_VALUE),
            cycle_period=int(node.findtext(_MODIFY_CYCLE_PERIOD)),
            cycle_burst=int(node.findtext(_MODIFY_CYCLE_BURST)),
            crc=int(node.findtext(_MODIFY_CRC)),
            checksum=int(node.findtext(_MODIFY_CHECKSUM))
        )


class ModifyRandom(Modify):
    """随机修改

    Notes:
        强制关键字传参

    """

    @overload
    def __init__(self, *,
                 match_header: int, match_offset: int, match_size: int, match_value: str,
                 modify_header: int, modify_offset: int, modify_size: int, modify_value: str,
                 random_rate: float,
                 checksum: int, crc: int = 0, match_switch: int = 1):
        """

        Args:
            match_header: 匹配报文头
            match_offset: 匹配偏移量
            match_size: 匹配大小
            match_value: 匹配值
            modify_header: 修改头
            modify_offset: 修改偏移量
            modify_size: 修改大小
            modify_value: 修改值
            random_rate: 概率
            crc: 是否重算 CRC
            checksum: 是否重算 checksum

        Examples:
            >>> rand = ModifyRandom(match_header=1, match_offset=0, match_size=1, match_value="0x0", modify_header=1,
            >>>                modify_offset=0, modify_size=1, modify_value="0xF",
            >>>                random_rate=0.001,
            >>>                crc=1, checksum=1)
        """
        ...

    @overload
    def __init__(self, match_params: dict, modify_params: dict, checksum: int, crc: int = 0, match_switch: int = 1):
        """

        Args:
            match_params: 匹配参数集
            modify_params: 修改参数集
            crc: 是否重算 CRC
            checksum: 是否重算 checksum

        Examples:
            >>> match_params = {
            >>>    "match_header":1, "match_offset":0, "match_size":1, "match_value":"0x0",
            >>> }
            >>> modify_params = {
            >>>     "modify_header":1, "modify_offset":0, "modify_size":1, "modify_value":"0xF",
            >>>     "random_rate":0.1,
            >>> }
            >>> rand = ModifyRandom(match_params = match_params,modify_params=modify_params,crc=1, checksum=1)
        """
        ...

    def __init__(self, **kwargs):
        super().__init__(_MODIFY_RANDOM_MODE)
        self.impairment_params[_MODIFY_TYPE_TAG] = _MODIFY_RANDOM_MODE
        match_switch = kwargs.get("match_switch")
        if match_switch is None:
            match_switch = 1
        self.impairment_params[_MODIFY_SWITCH_TAG] = match_switch
        # using default "match_params" first
        self.impairment_params[_MATCH_HEADER] = 1
        self.impairment_params[_MATCH_OFFSET] = 0
        self.impairment_params[_MATCH_SIZE] = 1
        self.impairment_params[_MATCH_VALUE] = "0x0"
        if kwargs.get("modify_params") is not None:
            modify_params = kwargs["modify_params"]
            if match_switch == 1:
                match_params = kwargs["match_params"]
                self.impairment_params[_MATCH_HEADER] = match_params["match_header"]
                self.impairment_params[_MATCH_OFFSET] = match_params["match_offset"]
                self.impairment_params[_MATCH_SIZE] = match_params["match_size"]
                self.impairment_params[_MATCH_VALUE] = match_params["match_value"]

            self.impairment_params[_MODIFY_HEADER] = modify_params["modify_header"]
            self.impairment_params[_MODIFY_OFFSET] = modify_params["modify_offset"]
            self.impairment_params[_MODIFY_SIZE] = modify_params["modify_size"]
            self.impairment_params[_MODIFY_VALUE] = modify_params["modify_value"]
            self.impairment_params[_MODIFY_RANDOM_RATE] = modify_params["random_rate"]
            crc = kwargs.get("crc")
            if crc is not None:
                self.impairment_params[_MODIFY_CRC] = crc
            self.impairment_params[_MODIFY_CHECKSUM] = kwargs["checksum"]
        else:
            self.impairment_params[_MATCH_HEADER] = kwargs["match_header"]
            self.impairment_params[_MATCH_OFFSET] = kwargs["match_offset"]
            self.impairment_params[_MATCH_SIZE] = kwargs["match_size"]
            self.impairment_params[_MATCH_VALUE] = kwargs["match_value"]
            self.impairment_params[_MODIFY_HEADER] = kwargs["modify_header"]
            self.impairment_params[_MODIFY_OFFSET] = kwargs["modify_offset"]
            self.impairment_params[_MODIFY_SIZE] = kwargs["modify_size"]
            self.impairment_params[_MODIFY_VALUE] = kwargs["modify_value"]
            self.impairment_params[_MODIFY_RANDOM_RATE] = kwargs["random_rate"]
            crc = kwargs.get("crc")
            if crc is not None:
                self.impairment_params[_MODIFY_CRC] = crc
            self.impairment_params[_MODIFY_CHECKSUM] = kwargs["checksum"]

        self._param_tag2str = _modify_tag2str_dict
        _modify_checker.check(self.impairment_params)
        self._update_params()

    # ===========================================================
    @property
    def random_rate(self) -> float:
        return self.impairment_params[_MODIFY_RANDOM_RATE]

    @random_rate.setter
    @check_parameter
    def random_rate(self, value: float) -> None:
        _modify_checker.valid_random_rate(value)
        self.impairment_params[_MODIFY_RANDOM_RATE] = value
        self._update_params()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'ModifyRandom':
        """由随机修改 xml 节点构造 ModifyRandom 对象

        Args:
            node: 随机修改 xml 节点

        Returns:
            ModifyRandom 对象
        """
        return ModifyRandom(
            match_header=int(node.findtext(_MATCH_HEADER)),
            match_offset=int(node.findtext(_MATCH_OFFSET)),
            match_size=int(node.findtext(_MATCH_SIZE)),
            match_value=node.findtext(_MATCH_VALUE),
            modify_header=int(node.findtext(_MODIFY_HEADER)),
            modify_offset=int(node.findtext(_MODIFY_OFFSET)),
            modify_size=int(node.findtext(_MODIFY_SIZE)),
            modify_value=node.findtext(_MODIFY_VALUE),
            random_rate=float(node.findtext(_MODIFY_RANDOM_RATE)),
            crc=int(node.findtext(_MODIFY_CRC)),
            checksum=int(node.findtext(_MODIFY_CHECKSUM))
        )


class ModifyRanges(Modify):
    """范围修改

    Notes:
        强制关键字传参
    """

    @overload
    def __init__(self, *,
                 match_header: int, match_offset: int, match_size: int, match_value: str,
                 modify_ranges: list,
                 checksum: int, crc: int = 0, match_switch: int = 1):
        """
        Args:
            match_header: 匹配报文头
            match_offset: 匹配偏移量
            match_size: 匹配大小
            match_value: 匹配值
            modify_ranges: 修改范围列表
            crc: 是否重算 CRC
            checksum: 是否重算 checksum
        """
        ...

    @overload
    def __init__(self, match_params: dict, modify_ranges: list, checksum: int, crc: int = 0, match_switch: int = 1):
        """
        Args:
            match_params: 匹配参数集
            modify_ranges: 修改范围列表
            crc: 是否重算 CRC
            checksum: 是否重算 checksum
        """
        ...

    def __init__(self, **kwargs):
        super().__init__(_MODIFY_RANGES_MODE)
        self.impairment_params[_MODIFY_TYPE_TAG] = _MODIFY_RANGES_MODE
        match_switch = kwargs.get("match_switch")
        if match_switch is None:
            match_switch = 1
        self.impairment_params[_MODIFY_SWITCH_TAG] = match_switch
        # 默认 match_params
        self.impairment_params[_MATCH_HEADER] = 1
        self.impairment_params[_MATCH_OFFSET] = 0
        self.impairment_params[_MATCH_SIZE] = 1
        self.impairment_params[_MATCH_VALUE] = "0x0"
        if kwargs.get("modify_ranges") is not None:
            modify_ranges = kwargs["modify_ranges"]
            if match_switch == 1 and kwargs.get("match_params") is not None:
                match_params = kwargs["match_params"]
                self.impairment_params[_MATCH_HEADER] = match_params["match_header"]
                self.impairment_params[_MATCH_OFFSET] = match_params["match_offset"]
                self.impairment_params[_MATCH_SIZE] = match_params["match_size"]
                self.impairment_params[_MATCH_VALUE] = match_params["match_value"]
            elif match_switch == 1:
                self.impairment_params[_MATCH_HEADER] = kwargs["match_header"]
                self.impairment_params[_MATCH_OFFSET] = kwargs["match_offset"]
                self.impairment_params[_MATCH_SIZE] = kwargs["match_size"]
                self.impairment_params[_MATCH_VALUE] = kwargs["match_value"]
            self.impairment_params[_MODIFY_RANGES] = modify_ranges
            crc = kwargs.get("crc")
            if crc is not None:
                self.impairment_params[_MODIFY_CRC] = crc
            self.impairment_params[_MODIFY_CHECKSUM] = kwargs["checksum"]
        else:
            raise ValueError("modify_ranges 参数不能为空")
        self._param_tag2str = _modify_tag2str_dict
        # 不做modify_ranges的check
        _modify_checker.check({k: v for k, v in self.impairment_params.items() if k != _MODIFY_RANGES})
        self._update_params()

    @property
    def modify_ranges(self) -> list:
        return self.impairment_params[_MODIFY_RANGES]

    @modify_ranges.setter
    @check_parameter
    def modify_ranges(self, value: list) -> None:
        self.impairment_params[_MODIFY_RANGES] = value
        self._update_params()

    def _construct_ranges(self):
        ranges_node = ET.Element(_MODIFY_RANGES)
        cnt_node = ET.Element(_MODIFY_RANGES_COUNT)
        cnt_node.text = str(len(self.impairment_params[_MODIFY_RANGES]))
        ranges_node.append(cnt_node)

        for idx, rg in enumerate(self.impairment_params[_MODIFY_RANGES]):
            node = ET.Element(_MODIFY_RANGES_RANGE + str(idx))
            offset_node = ET.Element(_MODIFY_RANGES_RANGE_OFFSET)
            offset_node.text = str(rg[_MODIFY_RANGES_RANGE_OFFSET])
            node.append(offset_node)

            size_node = ET.Element(_MODIFY_RANGES_RANGE_SIZE)
            size_node.text = str(_MODIFY_RANGES_RANGE_SIZE_CONSTANT)
            node.append(size_node)

            value_node = ET.Element(_MODIFY_RANGES_RANGE_VALUE)
            value_node.text = rg[_MODIFY_RANGES_RANGE_VALUE]
            node.append(value_node)

            ranges_node.append(node)

        return ranges_node

    def _update_params(self):
        self.clear_children()
        self.add_children(self.impairment_params)
        self.add_child(self._construct_ranges())

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'ModifyRanges':
        """由随机修改 xml 节点构造 ModifyRanges 对象

        Args:
            node: 随机修改 xml 节点

        Returns:
            ModifyRanges 对象
        """
        ranges_node = node.find(_MODIFY_RANGES)
        cnt = int(ranges_node.findtext(_MODIFY_RANGES_COUNT))
        ranges_ls = []
        for i in range(cnt):
            rangei_node = ranges_node.find(_MODIFY_RANGES_RANGE+str(i))
            ranges_ls.append({
                _MODIFY_RANGES_RANGE_SIZE:_MODIFY_RANGES_RANGE_SIZE_CONSTANT,
                _MODIFY_RANGES_RANGE_OFFSET:int(rangei_node.findtext(_MODIFY_RANGES_RANGE_OFFSET)),
                _MODIFY_RANGES_RANGE_VALUE:rangei_node.findtext(_MODIFY_RANGES_RANGE_VALUE)
            })

        return ModifyRanges(
            match_header=int(node.findtext(_MATCH_HEADER)),
            match_offset=int(node.findtext(_MATCH_OFFSET)),
            match_size=int(node.findtext(_MATCH_SIZE)),
            match_value=node.findtext(_MATCH_VALUE),
            modify_ranges=ranges_ls,
            crc=int(node.findtext(_MODIFY_CRC)),
            checksum=int(node.findtext(_MODIFY_CHECKSUM))
        )


class ModifyInsert(Modify):
    """插入修改

    Notes:
        强制关键字传参
    """

    @overload
    def __init__(self, *,
                 match_header: int, match_offset: int, match_size: int, match_value: str,
                 insert_offset: int, insert_value: str,
                 checksum: int, crc: int = 0, match_switch: int = 1):
        """
        Args:
            match_header: 匹配报文头
            match_offset: 匹配偏移量
            match_size: 匹配大小
            match_value: 匹配值
            insert_offset: 插入偏移量
            insert_value: 插入值
            crc: 是否重算 CRC
            checksum: 是否重算 checksum
        """
        ...

    @overload
    def __init__(self, match_params: dict, insert_params: dict, checksum: int, crc: int = 0, match_switch: int = 1):
        """
        Args:
            match_params: 匹配参数集
            insert_params: 插入参数集
            crc: 是否重算 CRC
            checksum: 是否重算 checksum
        """
        ...

    def __init__(self, **kwargs):
        super().__init__(_MODIFY_INSERT_MODE)
        self.impairment_params[_MODIFY_TYPE_TAG] = _MODIFY_INSERT_MODE
        match_switch = kwargs.get("match_switch")
        if match_switch is None:
            match_switch = 1
        self.impairment_params[_MODIFY_SWITCH_TAG] = match_switch
        # 默认 match_params
        self.impairment_params[_MATCH_HEADER] = 1
        self.impairment_params[_MATCH_OFFSET] = 0
        self.impairment_params[_MATCH_SIZE] = 1
        self.impairment_params[_MATCH_VALUE] = "0x0"
        if kwargs.get("insert_params") is not None:
            insert_params = kwargs["insert_params"]
            if match_switch == 1 and kwargs.get("match_params") is not None:
                match_params = kwargs["match_params"]
                self.impairment_params[_MATCH_HEADER] = match_params["match_header"]
                self.impairment_params[_MATCH_OFFSET] = match_params["match_offset"]
                self.impairment_params[_MATCH_SIZE] = match_params["match_size"]
                self.impairment_params[_MATCH_VALUE] = match_params["match_value"]
            elif match_switch == 1:
                self.impairment_params[_MATCH_HEADER] = kwargs["match_header"]
                self.impairment_params[_MATCH_OFFSET] = kwargs["match_offset"]
                self.impairment_params[_MATCH_SIZE] = kwargs["match_size"]
                self.impairment_params[_MATCH_VALUE] = kwargs["match_value"]
            self.impairment_params[_MODIFY_INSERT_OFFSET] = insert_params["insert_offset"]
            self.impairment_params[_MODIFY_INSERT_VALUE] = insert_params["insert_value"]
            crc = kwargs.get("crc")
            if crc is not None:
                self.impairment_params[_MODIFY_CRC] = crc
            self.impairment_params[_MODIFY_CHECKSUM] = kwargs["checksum"]
        else:
            self.impairment_params[_MATCH_HEADER] = kwargs["match_header"]
            self.impairment_params[_MATCH_OFFSET] = kwargs["match_offset"]
            self.impairment_params[_MATCH_SIZE] = kwargs["match_size"]
            self.impairment_params[_MATCH_VALUE] = kwargs["match_value"]
            self.impairment_params[_MODIFY_INSERT_OFFSET] = kwargs["insert_offset"]
            self.impairment_params[_MODIFY_INSERT_VALUE] = kwargs["insert_value"]
            crc = kwargs.get("crc")
            if crc is not None:
                self.impairment_params[_MODIFY_CRC] = crc
            self.impairment_params[_MODIFY_CHECKSUM] = kwargs["checksum"]
        self._param_tag2str = _modify_tag2str_dict
        # 不做insert_value的check
        _modify_checker.check({k: v for k, v in self.impairment_params.items() if k not in [_MODIFY_INSERT_OFFSET, _MODIFY_INSERT_VALUE]})
        self._update_params()

    @property
    def insert_offset(self) -> int:
        return self.impairment_params["insert_offset"]

    @insert_offset.setter
    @check_parameter
    def insert_offset(self, value: int) -> None:
        self.impairment_params["insert_offset"] = value
        self._update_params()

    @property
    def insert_value(self) -> str:
        return self.impairment_params["insert_value"]

    @insert_value.setter
    @check_parameter
    def insert_value(self, value: str) -> None:
        self.impairment_params["insert_value"] = value
        self._update_params()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'ModifyInsert':
        """由随机修改 xml 节点构造 ModifyInsert 对象

        Args:
            node: 随机修改 xml 节点

        Returns:
            ModifyInsert 对象
        """
        return ModifyInsert(
            match_header=int(node.findtext(_MATCH_HEADER)),
            match_offset=int(node.findtext(_MATCH_OFFSET)),
            match_size=int(node.findtext(_MATCH_SIZE)),
            match_value=node.findtext(_MATCH_VALUE),
            insert_offset=int(node.findtext(_MODIFY_INSERT_OFFSET)),
            insert_value=node.findtext(_MODIFY_INSERT_VALUE),
            crc=int(node.findtext(_MODIFY_CRC)),
            checksum=int(node.findtext(_MODIFY_CHECKSUM))
        )


class ModifyDelete(Modify):
    """删除修改

    Notes:
        强制关键字传参
    """

    @overload
    def __init__(self, *,
                 match_header: int, match_offset: int, match_size: int, match_value: str,
                 delete_offset: int, delete_size: int,
                 checksum: int, crc: int = 0, match_switch: int = 1):
        """
        Args:
            match_header: 匹配报文头
            match_offset: 匹配偏移量
            match_size: 匹配大小
            match_value: 匹配值
            delete_offset: 删除偏移量
            delete_size: 删除长度
            crc: 是否重算 CRC
            checksum: 是否重算 checksum
        """
        ...

    @overload
    def __init__(self, match_params: dict, delete_params: dict, checksum: int, crc: int = 0, match_switch: int = 1):
        """
        Args:
            match_params: 匹配参数集
            delete_params: 删除参数集
            crc: 是否重算 CRC
            checksum: 是否重算 checksum
        """
        ...

    def __init__(self, **kwargs):
        super().__init__(_MODIFY_DELETE_MODE)
        self.impairment_params[_MODIFY_TYPE_TAG] = _MODIFY_DELETE_MODE
        match_switch = kwargs.get("match_switch")
        if match_switch is None:
            match_switch = 1
        self.impairment_params[_MODIFY_SWITCH_TAG] = match_switch
        # 默认 match_params
        self.impairment_params[_MATCH_HEADER] = 1
        self.impairment_params[_MATCH_OFFSET] = 0
        self.impairment_params[_MATCH_SIZE] = 1
        self.impairment_params[_MATCH_VALUE] = "0x0"
        if kwargs.get("delete_params") is not None:
            delete_params = kwargs["delete_params"]
            if match_switch == 1 and kwargs.get("match_params") is not None:
                match_params = kwargs["match_params"]
                self.impairment_params[_MATCH_HEADER] = match_params["match_header"]
                self.impairment_params[_MATCH_OFFSET] = match_params["match_offset"]
                self.impairment_params[_MATCH_SIZE] = match_params["match_size"]
                self.impairment_params[_MATCH_VALUE] = match_params["match_value"]
            elif match_switch == 1:
                self.impairment_params[_MATCH_HEADER] = kwargs["match_header"]
                self.impairment_params[_MATCH_OFFSET] = kwargs["match_offset"]
                self.impairment_params[_MATCH_SIZE] = kwargs["match_size"]
                self.impairment_params[_MATCH_VALUE] = kwargs["match_value"]
            self.impairment_params[_MODIFY_DELETE] = {
                _MODIFY_DELETE_OFFSET: delete_params["delete_offset"],
                _MODIFY_DELETE_SIZE: delete_params["delete_size"]
            }
            crc = kwargs.get("crc")
            if crc is not None:
                self.impairment_params[_MODIFY_CRC] = crc
            self.impairment_params[_MODIFY_CHECKSUM] = kwargs["checksum"]
        else:
            self.impairment_params[_MATCH_HEADER] = kwargs["match_header"]
            self.impairment_params[_MATCH_OFFSET] = kwargs["match_offset"]
            self.impairment_params[_MATCH_SIZE] = kwargs["match_size"]
            self.impairment_params[_MATCH_VALUE] = kwargs["match_value"]
            self.impairment_params[_MODIFY_DELETE] = {
                _MODIFY_DELETE_OFFSET:kwargs["delete_offset"],
                _MODIFY_DELETE_SIZE:kwargs["delete_size"]
            }
            crc = kwargs.get("crc")
            if crc is not None:
                self.impairment_params[_MODIFY_CRC] = crc
            self.impairment_params[_MODIFY_CHECKSUM] = kwargs["checksum"]
        self._param_tag2str = _modify_tag2str_dict
        # 不做delete_value的check
        _modify_checker.check({k: v for k, v in self.impairment_params.items() if k not in ["delete_offset", "delete_size"]})
        self._update_params()

    @property
    def delete_offset(self) -> int:
        return self.impairment_params["delete_offset"]

    @delete_offset.setter
    @check_parameter
    def delete_offset(self, value: int) -> None:
        self.impairment_params["delete_offset"] = value
        self._update_params()

    @property
    def delete_size(self) -> int:
        return self.impairment_params["delete_size"]

    @delete_size.setter
    @check_parameter
    def delete_size(self, value: int) -> None:
        self.impairment_params["delete_size"] = value
        self._update_params()

    def _construct_delete(self):
        del_node = ET.Element(_MODIFY_DELETE)

        offset_node = ET.Element(_MODIFY_DELETE_OFFSET)
        offset_node.text = str(self.impairment_params[_MODIFY_DELETE][_MODIFY_DELETE_OFFSET])
        del_node.append(offset_node)

        size_node = ET.Element(_MODIFY_DELETE_SIZE)
        size_node.text = str(self.impairment_params[_MODIFY_DELETE][_MODIFY_DELETE_SIZE])
        del_node.append(size_node)

        return del_node

    def _update_params(self):
        self.clear_children()
        self.add_children(self.impairment_params)
        # self.add_child(self._construct_delete())

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'ModifyDelete':
        """由删除修改 xml 节点构造 ModifyDelete 对象

        Args:
            node: 删除修改 xml 节点

        Returns:
            ModifyDelete 对象
        """
        del_node = node.find(_MODIFY_DELETE)
        return ModifyDelete(
            match_header=int(node.findtext(_MATCH_HEADER)),
            match_offset=int(node.findtext(_MATCH_OFFSET)),
            match_size=int(node.findtext(_MATCH_SIZE)),
            match_value=node.findtext(_MATCH_VALUE),
            delete_offset=int(del_node.findtext(_MODIFY_DELETE_OFFSET)),
            delete_size=int(del_node.findtext(_MODIFY_DELETE_SIZE)),
            crc=int(node.findtext(_MODIFY_CRC)),
            checksum=int(node.findtext(_MODIFY_CHECKSUM))
        )


class ModifyExchange(Modify):
    """交换修改

    Notes:
        强制关键字传参
    """

    @overload
    def __init__(self, *,
                 match_header: int, match_offset: int, match_size: int, match_value: str,
                 previous_offset: int, previous_size: int, next_offset: int, next_size: int,
                 checksum: int, crc: int = 0, match_switch: int = 1):
        """
        Args:
            match_header: 匹配报文头
            match_offset: 匹配偏移量
            match_size: 匹配大小
            match_value: 匹配值
            previous_offset: 前段偏移量
            previous_size: 前段长度
            next_offset: 后段偏移量
            next_size: 后段长度
            crc: 是否重算 CRC
            checksum: 是否重算 checksum
        """
        ...

    @overload
    def __init__(self, match_params: dict, exchange_params: dict, checksum: int, crc: int = 0, match_switch: int = 1):
        """
        Args:
            match_params: 匹配参数集
            exchange_params: 交换参数集
            crc: 是否重算 CRC
            checksum: 是否重算 checksum
        """
        ...

    def __init__(self, **kwargs):
        super().__init__(_MODIFY_EXCHANGE_MODE)
        self.impairment_params[_MODIFY_TYPE_TAG] = _MODIFY_EXCHANGE_MODE
        match_switch = kwargs.get("match_switch")
        if match_switch is None:
            match_switch = 1
        self.impairment_params[_MODIFY_SWITCH_TAG] = match_switch
        # 默认 match_params
        self.impairment_params[_MATCH_HEADER] = 1
        self.impairment_params[_MATCH_OFFSET] = 0
        self.impairment_params[_MATCH_SIZE] = 1
        self.impairment_params[_MATCH_VALUE] = "0x0"
        if kwargs.get("exchange_params") is not None:
            exchange_params = kwargs["exchange_params"]
            if match_switch == 1 and kwargs.get("match_params") is not None:
                match_params = kwargs["match_params"]
                self.impairment_params[_MATCH_HEADER] = match_params["match_header"]
                self.impairment_params[_MATCH_OFFSET] = match_params["match_offset"]
                self.impairment_params[_MATCH_SIZE] = match_params["match_size"]
                self.impairment_params[_MATCH_VALUE] = match_params["match_value"]
            elif match_switch == 1:
                self.impairment_params[_MATCH_HEADER] = kwargs["match_header"]
                self.impairment_params[_MATCH_OFFSET] = kwargs["match_offset"]
                self.impairment_params[_MATCH_SIZE] = kwargs["match_size"]
                self.impairment_params[_MATCH_VALUE] = kwargs["match_value"]

            self._exchange_params = exchange_params
            crc = kwargs.get("crc")
            if crc is not None:
                self.impairment_params[_MODIFY_CRC] = crc
            self.impairment_params[_MODIFY_CHECKSUM] = kwargs["checksum"]
        else:
            self.impairment_params[_MATCH_HEADER] = kwargs["match_header"]
            self.impairment_params[_MATCH_OFFSET] = kwargs["match_offset"]
            self.impairment_params[_MATCH_SIZE] = kwargs["match_size"]
            self.impairment_params[_MATCH_VALUE] = kwargs["match_value"]

            self._exchange_params = {
                "previous_offset":kwargs["previous_offset"],
                "previous_size": kwargs["previous_size"],
                "next_offset": kwargs["next_offset"],
                "next_size": kwargs["next_size"],
            }
            crc = kwargs.get("crc")
            if crc is not None:
                self.impairment_params[_MODIFY_CRC] = crc
            self.impairment_params[_MODIFY_CHECKSUM] = kwargs["checksum"]
        self._param_tag2str = _modify_tag2str_dict
        # 不做exchange参数的check
        _modify_checker.check({k: v for k, v in self.impairment_params.items() if k not in ["previous_offset", "previous_size", "next_offset", "next_size"]})
        self._update_params()

    @property
    def previous_offset(self) -> int:
        return self._exchange_params["previous_offset"]

    @previous_offset.setter
    @check_parameter
    def previous_offset(self, value: int) -> None:
        self._exchange_params["previous_offset"] = value
        self._update_params()

    @property
    def previous_size(self) -> int:
        return self._exchange_params["previous_size"]

    @previous_size.setter
    @check_parameter
    def previous_size(self, value: int) -> None:
        self._exchange_params["previous_size"] = value
        self._update_params()

    @property
    def next_offset(self) -> int:
        return self._exchange_params["next_offset"]

    @next_offset.setter
    @check_parameter
    def next_offset(self, value: int) -> None:
        self._exchange_params["next_offset"] = value
        self._update_params()

    @property
    def next_size(self) -> int:
        return self._exchange_params["next_size"]

    @next_size.setter
    @check_parameter
    def next_size(self, value: int) -> None:
        self._exchange_params["next_size"] = value
        self._update_params()

    def _construct_exchange(self):
        ex_node = ET.Element(_MODIFY_EXCHANGE)

        prev_node = ET.Element(_MODIFY_EXCHANGE_PREVIOUS)
        prev_node.set("offset",str(self._exchange_params["previous_offset"]))
        prev_node.set("size", str(self._exchange_params["previous_size"]))
        ex_node.append(prev_node)

        next_node = ET.Element(_MODIFY_EXCHANGE_NEXT)
        next_node.set("offset", str(self._exchange_params["next_offset"]))
        next_node.set("size", str(self._exchange_params["next_size"]))
        ex_node.append(next_node)

        return ex_node

    def _update_params(self):
        self.clear_children()
        self.add_children(self.impairment_params)
        self.add_child(self._construct_exchange())

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'ModifyExchange':
        """由交换修改 xml 节点构造 ModifyExchange 对象

        Args:
            node: 交换修改 xml 节点

        Returns:
            ModifyExchange 对象
        """
        ex_node = node.find(_MODIFY_EXCHANGE)
        prev_node = ex_node.find(_MODIFY_EXCHANGE_PREVIOUS)
        next_node = ex_node.find(_MODIFY_EXCHANGE_NEXT)
        return ModifyExchange(
            match_header=int(node.findtext(_MATCH_HEADER)),
            match_offset=int(node.findtext(_MATCH_OFFSET)),
            match_size=int(node.findtext(_MATCH_SIZE)),
            match_value=node.findtext(_MATCH_VALUE),
            previous_offset=int(prev_node.get("offset")),
            previous_size=int(prev_node.get("size")),
            next_offset=int(next_node.get("offset")),
            next_size=int(next_node.get("size")),
            crc=int(node.findtext(_MODIFY_CRC)),
            checksum=int(node.findtext(_MODIFY_CHECKSUM))
        )


class ModifyCount(Modify):
    """计数修改

    Notes:
        强制关键字传参
    """

    @overload
    def __init__(self, *,
                 match_header: int, match_offset: int, match_size: int, match_value: str,
                 offset: int, value: str, modify_offset: int, modify_count: int,
                 checksum: int, crc: int = 0, match_switch: int = 1):
        """
        Args:
            match_header: 匹配报文头
            match_offset: 匹配偏移量
            match_size: 匹配大小
            match_value: 匹配值
            offset: 匹配计数偏移量
            value: 匹配计数值
            modify_offset: 修改计数偏移量
            modify_count: 修改计数次数
            crc: 是否重算 CRC
            checksum: 是否重算 checksum
        """
        ...

    @overload
    def __init__(self, match_params: dict, count_params: dict, checksum: int, crc: int = 0, match_switch: int = 1):
        """
        Args:
            match_params: 匹配参数集
            count_params: 计数参数集
            crc: 是否重算 CRC
            checksum: 是否重算 checksum
        """
        ...

    def __init__(self, **kwargs):
        super().__init__(_MODIFY_COUNT_MODE)
        self.impairment_params[_MODIFY_TYPE_TAG] = _MODIFY_COUNT_MODE
        match_switch = kwargs.get("match_switch")
        if match_switch is None:
            match_switch = 1
        self.impairment_params[_MODIFY_SWITCH_TAG] = match_switch
        # 默认 match_params
        self.impairment_params[_MATCH_HEADER] = 1
        self.impairment_params[_MATCH_OFFSET] = 0
        self.impairment_params[_MATCH_SIZE] = 1
        self.impairment_params[_MATCH_VALUE] = "0x0"
        if kwargs.get("count_params") is not None:
            count_params = kwargs["count_params"]
            if match_switch == 1 and kwargs.get("match_params") is not None:
                match_params = kwargs["match_params"]
                self.impairment_params[_MATCH_HEADER] = match_params["match_header"]
                self.impairment_params[_MATCH_OFFSET] = match_params["match_offset"]
                self.impairment_params[_MATCH_SIZE] = match_params["match_size"]
                self.impairment_params[_MATCH_VALUE] = match_params["match_value"]
            elif match_switch == 1:
                self.impairment_params[_MATCH_HEADER] = kwargs["match_header"]
                self.impairment_params[_MATCH_OFFSET] = kwargs["match_offset"]
                self.impairment_params[_MATCH_SIZE] = kwargs["match_size"]
                self.impairment_params[_MATCH_VALUE] = kwargs["match_value"]
            self._count_params = {
                _MODIFY_COUNT_OFFSET:count_params["offset"],
                _MODIFY_COUNT_VALUE:count_params["value"],
                _MODIFY_COUNT_START_OFFSET:count_params["modify_offset"],
                _MODIFY_COUNT_MODIFY_COUNT: count_params["modify_count"]
            }
            crc = kwargs.get("crc")
            if crc is not None:
                self.impairment_params[_MODIFY_CRC] = crc
            self.impairment_params[_MODIFY_CHECKSUM] = kwargs["checksum"]
        else:
            self.impairment_params[_MATCH_HEADER] = kwargs["match_header"]
            self.impairment_params[_MATCH_OFFSET] = kwargs["match_offset"]
            self.impairment_params[_MATCH_SIZE] = kwargs["match_size"]
            self.impairment_params[_MATCH_VALUE] = kwargs["match_value"]
            self._count_params = {
                _MODIFY_COUNT_OFFSET: kwargs["offset"],
                _MODIFY_COUNT_VALUE: kwargs["value"],
                _MODIFY_COUNT_START_OFFSET: kwargs["modify_offset"],
                _MODIFY_COUNT_MODIFY_COUNT: kwargs["modify_count"]
            }
            crc = kwargs.get("crc")
            if crc is not None:
                self.impairment_params[_MODIFY_CRC] = crc
            self.impairment_params[_MODIFY_CHECKSUM] = kwargs["checksum"]
        self._param_tag2str = _modify_tag2str_dict
        # 不做count参数的check
        _modify_checker.check({k: v for k, v in self.impairment_params.items() if k not in [_MODIFY_COUNT_OFFSET, _MODIFY_COUNT_VALUE, _MODIFY_COUNT_START_OFFSET, _MODIFY_COUNT_MODIFY_COUNT]})
        self._update_params()

    @property
    def offset(self) -> int:
        return self._count_params[_MODIFY_COUNT_OFFSET]

    @offset.setter
    @check_parameter
    def offset(self, value: int) -> None:
        self._count_params[_MODIFY_COUNT_OFFSET] = value
        self._update_params()

    @property
    def value(self) -> str:
        return self._count_params[_MODIFY_COUNT_VALUE]

    @value.setter
    @check_parameter
    def value(self, value_: str) -> None:
        self._count_params[_MODIFY_COUNT_VALUE] = value_
        self._update_params()

    @property
    def modify_offset(self) -> int:
        return self._count_params[_MODIFY_COUNT_START_OFFSET]

    @modify_offset.setter
    @check_parameter
    def modify_offset(self, value: int) -> None:
        self._count_params[_MODIFY_COUNT_START_OFFSET] = value
        self._update_params()

    @property
    def modify_count(self) -> int:
        return self._count_params[_MODIFY_COUNT_MODIFY_COUNT]

    @modify_count.setter
    @check_parameter
    def modify_count(self, value: int) -> None:
        self._count_params[_MODIFY_COUNT_MODIFY_COUNT] = value
        self._update_params()

    def _construct_count(self):
        cnt_node = ET.Element(_MODIFY_COUNT)
        nb_node = ET.Element(_MODIFY_RANGES_COUNT)
        nb_node.text = str(1)
        cnt_node.append(nb_node)

        rgnode = ET.Element(_MODIFY_RANGES_RANGE + str(0))
        offset_node = ET.Element(_MODIFY_COUNT_OFFSET)
        offset_node.text = str(self._count_params[_MODIFY_COUNT_OFFSET])
        rgnode.append(offset_node)

        size_node = ET.Element(_MODIFY_RANGES_RANGE_SIZE)
        size_node.text = str(_MODIFY_COUNT_SIZE_CONSTANT)
        rgnode.append(size_node)

        value_node = ET.Element(_MODIFY_COUNT_VALUE)
        value_node.text =str(self._count_params[_MODIFY_COUNT_VALUE])
        rgnode.append(value_node)

        cnt_node.append(rgnode)

        start_offset_node = ET.Element(_MODIFY_COUNT_START_OFFSET)
        start_offset_node.text = str(self._count_params[_MODIFY_COUNT_START_OFFSET])
        cnt_node.append(start_offset_node)

        modify_count_node = ET.Element(_MODIFY_COUNT_MODIFY_COUNT)
        modify_count_node.text = str(self._count_params[_MODIFY_COUNT_MODIFY_COUNT])
        cnt_node.append(modify_count_node)

        return cnt_node

    def _update_params(self):
        self.clear_children()
        self.add_children(self.impairment_params)
        self.add_child(self._construct_count())

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'ModifyCount':
        """由计数修改 xml 节点构造 ModifyCount 对象

        Args:
            node: 计数修改 xml 节点

        Returns:
            ModifyCount 对象
        """
        cnt_node = node.find(_MODIFY_COUNT)
        rg_node = cnt_node.find(_MODIFY_RANGES_RANGE+str(0))
        return ModifyCount(
            match_header=int(node.findtext(_MATCH_HEADER)),
            match_offset=int(node.findtext(_MATCH_OFFSET)),
            match_size=int(node.findtext(_MATCH_SIZE)),
            match_value=node.findtext(_MATCH_VALUE),
            offset=int(rg_node.findtext(_MODIFY_COUNT_OFFSET)),
            value=rg_node.findtext(_MODIFY_COUNT_VALUE),
            modify_offset=int(cnt_node.findtext(_MODIFY_COUNT_START_OFFSET)),
            modify_count=int(cnt_node.findtext(_MODIFY_COUNT_MODIFY_COUNT)),
            crc=int(node.findtext(_MODIFY_CRC)),
            checksum=int(node.findtext(_MODIFY_CHECKSUM))
        )

    def __str__(self):
        res = self.__class__.__name__ + ":{ "
        for k, v in self.impairment_params.items():
            if k == _MODIFY_TYPE_TAG:
                continue
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)

        for k,v in self._count_params.items():
            res += r"{0}:{1} ".format(k, v)
        res += "}"
        return res


def modify_factory(node: ET.Element) -> Modify:
    mode = int(node.findtext(_MODIFY_TYPE_TAG))
    if mode == _MODIFY_DISABLE_MODE:
        return ModifyDisable()
    elif mode == _MODIFY_NORMAL_MODE:
        return ModifyNormal.construct_from_node(node)
    elif mode == _MODIFY_CYCLE_MODE:
        return ModifyCycle.construct_from_node(node)
    elif mode == _MODIFY_RANDOM_MODE:
        return ModifyRandom.construct_from_node(node)
    elif mode == _MODIFY_RANGES_MODE:
        return ModifyRanges.construct_from_node(node)
    elif mode == _MODIFY_INSERT_MODE:
        return ModifyInsert.construct_from_node(node)
    elif mode == _MODIFY_DELETE_MODE:
        return ModifyDelete.construct_from_node(node)
    elif mode == _MODIFY_EXCHANGE_MODE:
        return ModifyExchange.construct_from_node(node)
    elif mode == _MODIFY_COUNT_MODE:
        return ModifyCount.construct_from_node(node)
    else:
        raise ValueError(r"Unknown modify type.")


if __name__ == '__main__':
    from holowan.v2.engine import Engine

    modify_normal = ModifyNormal(match_header=1, match_offset=0, match_size=1, match_value="0x00", modify_header=1,
                                 modify_offset=0, modify_size=1, modify_value="0xFF", checksum=1, crc=1)

    holowan_ip = "192.168.1.111"
    holowan_port = "8080"
    engine_id = 3
    engine = Engine(holowan_ip, holowan_port, engine_id)
    result = engine.apply_impairment(modify_normal, 1, 1)
    print(result)
    # 形式2 通过参数字典形式传参
    match_params = {
        "match_header": 1, "match_offset": 0, "match_size": 1, "match_value": "0x00",
    }
    modify_params = {
        "modify_header": 1, "modify_offset": 0, "modify_size": 1, "modify_value": "0xFF"
    }
    normal = ModifyNormal(match_params=match_params, modify_params=modify_params, checksum=1, crc=1)
    print(normal)
    result = engine.apply_impairment(modify_normal, 1, 1)
    print(result)

    cycle = ModifyCycle(match_header=1, match_offset=0, match_size=1, match_value="0x0", modify_header=1,
                        modify_offset=0, modify_size=1, modify_value="0xF",
                        cycle_period=1000, cycle_burst=100,
                        crc=1, checksum=1)
    print(cycle)
    result = engine.apply_impairment(cycle, 1, 1)
    print(result)

    match_params = {
        "match_header": 1, "match_offset": 0, "match_size": 1, "match_value": "0x0",
    }
    modify_params = {
        "modify_header": 1, "modify_offset": 0, "modify_size": 1, "modify_value": "0xF",
        "cycle_period": 1000, "cycle_burst": 100,
    }
    cycle = ModifyCycle(match_params=match_params, modify_params=modify_params, crc=1, checksum=1, match_switch=0)
    result = engine.apply_impairment(cycle, 1, 1)
    print(result)

    modify_ranges = [
        {"offset": 0, "value": "ff"},
        {"offset": 1, "value": "ee"}
    ]
    modify_rgs = ModifyRanges(match_header=1, match_offset=0, match_size=1, match_value="0x0",
                              modify_ranges=modify_ranges, crc=1, checksum=1)
    print(modify_rgs)
    print(modify_rgs.xml)

    result = engine.apply_impairment(modify_rgs, 1, 1)
    print(result)

    modify_ins = ModifyInsert(match_header=1, match_offset=0, match_size=1, match_value="0x0",
                              insert_offset=1,insert_value="ff", crc=1, checksum=1)
    print(modify_ins)
    print(modify_ins.xml)

    result = engine.apply_impairment(modify_ins, 1, 1)
    print(result)

    modify_del = ModifyDelete(match_header=1, match_offset=0, match_size=1, match_value="0x0",
                              delete_size=12, delete_offset=21, crc=1, checksum=1)
    print(modify_del)
    print(modify_del.xml)

    result = engine.apply_impairment(modify_del, 1, 1)
    print(result)

    modify_ex = ModifyExchange(match_header=1, match_offset=0, match_size=1, match_value="0x0",
                              previous_offset=12, previous_size=21,next_offset=34,next_size=43, crc=1, checksum=1)
    print(modify_ex)
    print(modify_ex.xml)

    result = engine.apply_impairment(modify_ex, 1, 1)
    print(result)

    modify_cnt = ModifyCount(match_header=1, match_offset=0, match_size=1, match_value="0x0",
                               offset=12, value="ff", modify_offset=34, modify_count=43, crc=1, checksum=1)
    print(modify_cnt)
    print(modify_cnt.xml)

    result = engine.apply_impairment(modify_cnt, 1, 1)
    print(result)



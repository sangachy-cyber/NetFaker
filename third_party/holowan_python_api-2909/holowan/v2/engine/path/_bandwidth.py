"""
HoloWAN Network Emulator
Python API
"""

import xml.etree.cElementTree as ET
from typing import overload

from holowan.v2._holowan_types import check_parameter
from holowan.v2.engine.path._change_mode import ChangeMode, _CHANGE_MODE_TAG
from holowan.v2.engine.path._impairment_base import Bandwidth, _IMPAIRMENT_TYPE

_BANDWIDTH_FIXED: str = r"fi"
_BANDWIDTH_FIXED_RATE: str = r"r"
_BANDWIDTH_UNIT: str = r"t"

_BANDWIDTH_JITTER = r"ji"
_BANDWIDTH_JITTER_PHASE = r"phase"
_BANDWIDTH_JITTER_SHAKE = r"shake"
_BANDWIDTH_JITTER_SHAKE_TYPE_ID = r"type_id"
_BANDWIDTH_JITTER_MAX = r"max"
_BANDWIDTH_JITTER_MIN = r"min"
_BANDWIDTH_JITTER_CYCLE = r"cycle"

_BANDWIDTH_TOKEN_BUCKET = r"tb"
_BANDWIDTH_TOKEN_BUCKET_TYPE = r"s"
_BANDWIDTH_TOKEN_BUCKET_AIR = r"air"
_BANDWIDTH_TOKEN_BUCKET_AIR_UNIT = r"air_t"
_BANDWIDTH_TOKEN_BUCKET_ABS = r"abs"
_BANDWIDTH_TOKEN_BUCKET_ABS_UNIT = r"abs_t"
_BANDWIDTH_TOKEN_BUCKET_BIR = r"bir"
_BANDWIDTH_TOKEN_BUCKET_BIR_UNIT = r"bir_t"
_BANDWIDTH_TOKEN_BUCKET_BBS = r"bbs"
_BANDWIDTH_TOKEN_BUCKET_BBS_UNIT = r"bbs_t"

_BANDWIDTH_BIDIRECTIONAL:str = r"bidirectional"
_BANDWIDTH_BIDIRECTIONAL_RATE: str = r"r"

_FIXED_MODE: int = 1
_JITTER_MODE: int = 2
_TOKEN_BUCKET_MODE: int = 3
_BIDIRECTIONAL_MODE: int = 4


class BandwidthFixed(Bandwidth):
    """固定带宽模式

    Args:
        rate: 速率。
        unit: 单位。

    Examples:
        >>> fixed = BandwidthFixed(rate=999.9, unit=1)
    """

    def __init__(self, *, rate: float, unit: int):
        super().__init__(_FIXED_MODE, _BANDWIDTH_FIXED)
        self.impairment_params[_BANDWIDTH_FIXED_RATE] = rate
        self.impairment_params[_BANDWIDTH_UNIT] = unit
        self._param_tag2str = {
            _BANDWIDTH_FIXED_RATE: r"rate",
            _BANDWIDTH_UNIT: r"unit"
        }
        self._update_param()

    @property
    def rate(self) -> None:
        """速率
        Note:
            可读可写。

        """
        return self.impairment_params[_BANDWIDTH_FIXED_RATE]

    @rate.setter
    @check_parameter
    def rate(self, value: float) -> None:
        self.impairment_params[_BANDWIDTH_FIXED_RATE] = value
        self._update_param()

    @property
    def unit(self) -> int:
        """单位
        Note:
            可读可写。
        """
        return self.impairment_params[_BANDWIDTH_UNIT]

    @unit.setter
    @check_parameter
    def unit(self, value: int) -> None:
        self.impairment_params[_BANDWIDTH_UNIT] = value
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'BandwidthFixed':
        """由一个固定带宽限制的 xml 节点构造 BandwidthFixed 对象。

        Args:
            node: 固定带宽的 xml 节点。

        Returns:
            BandwidthFixed 对象。
        """
        details = node.find(_BANDWIDTH_FIXED)
        return BandwidthFixed(
            rate=float(details.findtext(_BANDWIDTH_FIXED_RATE)),
            unit=int(details.findtext(_BANDWIDTH_UNIT))
        )


class BandwidthJitter(Bandwidth):
    """抖动带宽模式

    Notes:
        强制关键字传参

    Args
        unit(int): 单位
        change_mode(ChangeMode): 变化模式。

    Examples:
        >>> jitter = BandwidthJitter(unit=1, change_mode=ChangeMode(mode=1, max=99.9, min=11.1, phase=1, period=100))
    """

    def __init__(self, *, unit: int, change_mode: ChangeMode) -> None:
        super().__init__(_JITTER_MODE, _BANDWIDTH_JITTER)
        self.impairment_params[_BANDWIDTH_UNIT] = unit
        self._param_tag2str = {
            _BANDWIDTH_UNIT: r"unit"
        }
        self._change_mode = change_mode
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
    def unit(self) -> int:
        """单位

        Notes:
            可读可写

        """
        return self.impairment_params[_BANDWIDTH_UNIT]

    @unit.setter
    @check_parameter
    def unit(self, value: int) -> None:
        self.impairment_params[_BANDWIDTH_UNIT] = value
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'BandwidthJitter':
        """由一个抖动带宽模式的 xml 节点构造 BandwidthJitter 对象。

        Args:
            node: 抖动带宽模式的 xml 节点

        Returns:
            BandwidthJitter 对象。
        """
        details = node.find(_BANDWIDTH_JITTER)
        unit = int(details.findtext(_BANDWIDTH_UNIT))
        change_node = details.find(_CHANGE_MODE_TAG)
        chang_mode = ChangeMode.construct_from_node(change_node)
        return BandwidthJitter(
            unit=unit, change_mode=chang_mode
        )

    def __str__(self):
        res = self.__class__.__name__ + ":{"
        for k, v in self.impairment_params.items():
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)

        res += self._change_mode.__str__()
        res += "}"
        return res


class BandwidthTokenBucket(Bandwidth):
    """令牌桶带宽限制模式

    Notes：
        强制关键字传参

    """

    @overload
    def __init__(self, *, type: int,
                 rate: float, rate_unit: int,
                 burst: float, burst_unit: int):
        """构造方式 1

        Args:
            type: 令牌桶类型
            rate: 速率
            rate_unit: 速率单位
            burst: 突发
            burst_unit: 突发单位
        """
        ...

    @overload
    def __init__(self, *, type: int,
                 cir: float, cir_unit: int,
                 cbs: float, cbs_unit: int,
                 ebs: float, ebs_unit: int):
        """构造方式 2

        Args:
            type: 令牌桶类型
            cir: cir 速率
            cir_unit: cir 速率单位
            cbs: cbs 速率
            cbs_unit: cbs 速率单位
            ebs: ebs 速率
            ebs_unit: ebs 速率单位
        """
        ...

    @overload
    def __init__(self, *, type: int,
                 cir: float, cir_unit: int,
                 cbs: float, cbs_unit: int,
                 pir: float, pir_unit: int,
                 pbs: float, pbs_unit: int):
        """

        Args:
            type: 令牌桶类型
            cir: cir 速率
            cir_unit: cir 速率单位
            cbs: cbs 速率
            cbs_unit: cbs 速率单位
            pir: pir 速率
            pir_unit: pir 速率单位
            pbs: pbs 速率
            pbs_unit: pbs 速率单位
        """
        ...

    def __init__(self, **kwargs):
        super().__init__(_TOKEN_BUCKET_MODE, _BANDWIDTH_TOKEN_BUCKET)
        tb_type = kwargs["type"]
        if tb_type == 1:
            self._single_bucket(kwargs)
        elif tb_type == 2:
            self._srTCM(kwargs)
        elif tb_type == 3:
            self._trTCM(kwargs)
        else:
            raise ValueError("Unknown Bandwidth:token bucket type.")
        self._type = tb_type
        self._param_tag2str = _parse_dict[tb_type]
        self._update_param()

    def _update_param(self) -> None:
        self.clear_children()
        self.add_child(self._construct_details())

    def _construct_details(self) -> ET.Element:
        root = ET.Element(self._detail_tag)
        type_node = ET.Element(_BANDWIDTH_TOKEN_BUCKET_TYPE)
        type_node.text = str(self._type)
        root.append(type_node)
        for k, v in self.impairment_params.items():
            node = ET.Element(k)
            node.text = str(v)
            root.append(node)
        return root

    def _single_bucket(self, params: dict) -> None:
        self.impairment_params[_BANDWIDTH_TOKEN_BUCKET_AIR] = params["rate"]
        self.impairment_params[_BANDWIDTH_TOKEN_BUCKET_AIR_UNIT] = params["rate_unit"]
        self.impairment_params[_BANDWIDTH_TOKEN_BUCKET_ABS] = params["burst"]
        self.impairment_params[_BANDWIDTH_TOKEN_BUCKET_ABS_UNIT] = params["burst_unit"]
        self.impairment_params[_BANDWIDTH_TOKEN_BUCKET_BIR] = 0
        self.impairment_params[_BANDWIDTH_TOKEN_BUCKET_BIR_UNIT] = 1
        self.impairment_params[_BANDWIDTH_TOKEN_BUCKET_BBS] = 0
        self.impairment_params[_BANDWIDTH_TOKEN_BUCKET_BBS_UNIT] = 1

    def _srTCM(self, params: dict) -> None:
        self.impairment_params[_BANDWIDTH_TOKEN_BUCKET_AIR] = params["cir"]
        self.impairment_params[_BANDWIDTH_TOKEN_BUCKET_AIR_UNIT] = params["cir_unit"]
        self.impairment_params[_BANDWIDTH_TOKEN_BUCKET_ABS] = params["cbs"]
        self.impairment_params[_BANDWIDTH_TOKEN_BUCKET_ABS_UNIT] = params["cbs_unit"]
        self.impairment_params[_BANDWIDTH_TOKEN_BUCKET_BIR] = params["ebs"]
        self.impairment_params[_BANDWIDTH_TOKEN_BUCKET_BIR_UNIT] = params["ebs_unit"]
        self.impairment_params[_BANDWIDTH_TOKEN_BUCKET_BBS] = 0
        self.impairment_params[_BANDWIDTH_TOKEN_BUCKET_BBS_UNIT] = 1

    def _trTCM(self, params: dict) -> None:
        self.impairment_params[_BANDWIDTH_TOKEN_BUCKET_AIR] = params["cir"]
        self.impairment_params[_BANDWIDTH_TOKEN_BUCKET_AIR_UNIT] = params["cir_unit"]
        self.impairment_params[_BANDWIDTH_TOKEN_BUCKET_ABS] = params["cbs"]
        self.impairment_params[_BANDWIDTH_TOKEN_BUCKET_ABS_UNIT] = params["cbs_unit"]
        self.impairment_params[_BANDWIDTH_TOKEN_BUCKET_BIR] = params["pir"]
        self.impairment_params[_BANDWIDTH_TOKEN_BUCKET_BIR_UNIT] = params["pir_unit"]
        self.impairment_params[_BANDWIDTH_TOKEN_BUCKET_BBS] = params["pbs"]
        self.impairment_params[_BANDWIDTH_TOKEN_BUCKET_BBS_UNIT] = params["pbs_unit"]

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'BandwidthTokenBucket':
        """由一个令牌桶带宽限制的 xml 节点构造 BandwidthTokenBucket 对象

        Args:
            node: 令牌桶带宽限制的 xml 节点

        Returns:
            BandwidthTokenBucket 对象
        """
        details = node.find(_BANDWIDTH_TOKEN_BUCKET)
        parse_dict = _parse_dict[int(details.findtext(_BANDWIDTH_TOKEN_BUCKET_TYPE))]
        params = {}
        for node in list(details):
            name = parse_dict[node.tag]
            value = _parse_type[node.tag](node.text)
            params[name] = value

        return BandwidthTokenBucket(**params)


class BandwidthBidirectional(Bandwidth):
    """双向带宽模式

    Args:
        rate: 速率。
        unit: 单位。

    Examples:
        >>> bidir = BandwidthBidirectional(rate=1000.0, unit=1)
    """

    def __init__(self, *, rate: float, unit: int):
        super().__init__(_BIDIRECTIONAL_MODE, _BANDWIDTH_BIDIRECTIONAL)
        self.impairment_params[_BANDWIDTH_FIXED_RATE] = rate
        self.impairment_params[_BANDWIDTH_UNIT] = unit
        self._param_tag2str = {
            _BANDWIDTH_FIXED_RATE: r"rate",
            _BANDWIDTH_UNIT: r"unit"
        }
        self._update_param()

    @property
    def rate(self) -> float:
        """速率
        Note:
            可读可写。
        """
        return self.impairment_params[_BANDWIDTH_FIXED_RATE]

    @rate.setter
    @check_parameter
    def rate(self, value: float) -> None:
        self.impairment_params[_BANDWIDTH_FIXED_RATE] = value
        self._update_param()

    @property
    def unit(self) -> int:
        """单位
        Note:
            可读可写。
        """
        return self.impairment_params[_BANDWIDTH_UNIT]

    @unit.setter
    @check_parameter
    def unit(self, value: int) -> None:
        self.impairment_params[_BANDWIDTH_UNIT] = value
        self._update_param()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'BandwidthBidirectional':
        """由一个双向带宽限制的 xml 节点构造 BandwidthBidirectional 对象。

        Args:
            node: 双向带宽的 xml 节点。

        Returns:
            BandwidthBidirectional 对象。
        """
        details = node.find(_BANDWIDTH_BIDIRECTIONAL)
        return BandwidthBidirectional(
            rate=float(details.findtext(_BANDWIDTH_FIXED_RATE)),
            unit=int(details.findtext(_BANDWIDTH_UNIT))
        )

_tb1_params_tag2str_dict: dict = {
    _BANDWIDTH_TOKEN_BUCKET_TYPE: "type",
    _BANDWIDTH_TOKEN_BUCKET_AIR: "rate",
    _BANDWIDTH_TOKEN_BUCKET_AIR_UNIT: "rate_unit",
    _BANDWIDTH_TOKEN_BUCKET_ABS: "burst",
    _BANDWIDTH_TOKEN_BUCKET_ABS_UNIT: "burst_unit",
    _BANDWIDTH_TOKEN_BUCKET_BIR: "bir",
    _BANDWIDTH_TOKEN_BUCKET_BIR_UNIT: "bir_t",
    _BANDWIDTH_TOKEN_BUCKET_BBS: "bbs",
    _BANDWIDTH_TOKEN_BUCKET_BBS_UNIT: "bbs_t"
}

_tb2_params_tag2str_dict: dict = {
    _BANDWIDTH_TOKEN_BUCKET_TYPE: "type",
    _BANDWIDTH_TOKEN_BUCKET_AIR: "cir",
    _BANDWIDTH_TOKEN_BUCKET_AIR_UNIT: "cir_unit",
    _BANDWIDTH_TOKEN_BUCKET_ABS: "cbs",
    _BANDWIDTH_TOKEN_BUCKET_ABS_UNIT: "cbs_unit",
    _BANDWIDTH_TOKEN_BUCKET_BIR: "ebs",
    _BANDWIDTH_TOKEN_BUCKET_BIR_UNIT: "ebs_unit",
    _BANDWIDTH_TOKEN_BUCKET_BBS: "bbs",
    _BANDWIDTH_TOKEN_BUCKET_BBS_UNIT: "bbs_t"
}

_tb3_params_tag2str_dict: dict = {
    _BANDWIDTH_TOKEN_BUCKET_TYPE: "type",
    _BANDWIDTH_TOKEN_BUCKET_AIR: "cir",
    _BANDWIDTH_TOKEN_BUCKET_AIR_UNIT: "cir_unit",
    _BANDWIDTH_TOKEN_BUCKET_ABS: "cbs",
    _BANDWIDTH_TOKEN_BUCKET_ABS_UNIT: "cbs_unit",
    _BANDWIDTH_TOKEN_BUCKET_BIR: "pir",
    _BANDWIDTH_TOKEN_BUCKET_BIR_UNIT: "pir_unit",
    _BANDWIDTH_TOKEN_BUCKET_BBS: "pbs",
    _BANDWIDTH_TOKEN_BUCKET_BBS_UNIT: "pbs_unit"
}

_parse_dict: dict = {
    1: _tb1_params_tag2str_dict,
    2: _tb2_params_tag2str_dict,
    3: _tb3_params_tag2str_dict
}

_parse_type: dict = {
    _BANDWIDTH_TOKEN_BUCKET_TYPE: int,
    _BANDWIDTH_TOKEN_BUCKET_AIR: float,
    _BANDWIDTH_TOKEN_BUCKET_AIR_UNIT: int,
    _BANDWIDTH_TOKEN_BUCKET_ABS: float,
    _BANDWIDTH_TOKEN_BUCKET_ABS_UNIT: int,
    _BANDWIDTH_TOKEN_BUCKET_BIR: float,
    _BANDWIDTH_TOKEN_BUCKET_BIR_UNIT: int,
    _BANDWIDTH_TOKEN_BUCKET_BBS: float,
    _BANDWIDTH_TOKEN_BUCKET_BBS_UNIT: int
}


def bandwidth_factory(node: ET.Element) -> Bandwidth:
    mode = int(node.findtext(_IMPAIRMENT_TYPE))
    if mode == _FIXED_MODE:
        return BandwidthFixed.construct_from_node(node)
    elif mode == _JITTER_MODE:
        return BandwidthJitter.construct_from_node(node)
    elif mode == _TOKEN_BUCKET_MODE:
        return BandwidthTokenBucket.construct_from_node(node)
    elif mode == _BIDIRECTIONAL_MODE:
        return BandwidthBidirectional.construct_from_node(node)
    else:
        raise ValueError("Unknown Bandwidth type.")


if __name__ == '__main__':
    from holowan.v2.engine import Engine

    engine = Engine(holowan_ip="192.168.1.111", holowan_port="8080", engine_id=3)
    tb1 = BandwidthTokenBucket(type=1, rate=2, rate_unit=2, burst=3, burst_unit=1)
    result = engine.apply_impairment(tb1, 1, 1)
    print(result)

    tb2 = BandwidthTokenBucket(type=2, cir=2.1, cir_unit=2, cbs=3.2, cbs_unit=1, ebs=1, ebs_unit=1)
    result = engine.apply_impairment(tb2, 1, 1)
    print(result)

    tb3 = BandwidthTokenBucket(type=3, cir=1.1, cir_unit=1, cbs=2.2, cbs_unit=2, pir=3.3, pir_unit=3, pbs=4.4,
                               pbs_unit=1)
    result = engine.apply_impairment(tb3, 1, 3)
    print(result)

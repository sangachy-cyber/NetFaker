"""
HoloWAN Network Emulator
Python API
"""

import functools
import xml.etree.cElementTree as ET
from typing import Callable, Union

import holowan.v2.utils.my_util as mt
from holowan.v2.pixel._filter_base import Filter
from holowan.v2 import PortNumber, PortNumberRange, PortNumberList

_FILTER_NAME = r"mac"
_ANY: str = r"any"
_SRC: str = r"src"
_DST: str = r"dst"
_TYPE: str = r"type"


def _check_mac_rule_parameters(function: Callable):
    def __is_port_range(port_name: str, port_str: str):
        range = port_str.split("-")
        start, end = range[0], range[1]
        if mt.isPort(start) == False:
            raise ValueError("The start of '{0}' port range is not a valid port number".format(port_name))
        if mt.isPort(end) == False:
            raise ValueError("The end of '{0}' port range is not a valid port number".format(port_name))
        if int(start) - int(end) >= 0:
            raise ValueError(
                "The start of '{0}' port range(value: {1}) can not be greater(>=) the end of the range(value: {2})".format(
                    port_name, start, end))
        return True

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        new_kwargs = {}
        for k, v in kwargs.items():
            if k == _SRC or k == _DST:
                if isinstance(v, int):
                    if mt.isPort(str(v)) == False:
                        # int, not valid
                        raise TypeError(r"Argument {argument!r} is not a valid port number.".format(argument=k))
                    else:
                        # int, valid
                        new_kwargs[k] = v
                elif isinstance(v, str):
                    if mt.isPort(v):
                        # str, valid
                        new_kwargs[k] = v
                    else:
                        # str, range, any
                        if v == _ANY or __is_port_range(k, v):
                            new_kwargs[k] = v
                elif isinstance(v, list):
                    for idx, pt in enumerate(set(v)):
                        if mt.isPort(str(pt)):
                            new_kwargs[k] = v
                        else:
                            if pt == _ANY or __is_port_range(k, pt):
                                new_kwargs[k] = v
                            else:
                                raise ValueError(("The value:{value!r} in the {argument!r} port list"
                                                  "is not a valid port number").format(argument=k, value=pt))
                else:
                    raise TypeError('Argument {argument} must be {expected!r},but got {got!r},value {value!r}'.format(
                        argument=k, expected="PortNumber, PortNumberRange, List[PortNumber, List[PortNumberRange]",
                        got="unknown type", value=v
                    ))
            else:
                # TODO:has_path
                new_kwargs[k] = v
        return function(*tuple(args), **dict(new_kwargs))

    return wrapper


class MACFilter(Filter):
    """MAC分类规则。

    Note:
        强制关键字传参。

    Args:
        src : 源 mac 地址。
        dst: 目的 mac 地址。
        type(str): EtherType类型

    Examples:
        >>> mac = MACFilter(src="39-A9-FA-D7-F5-E8", dst=r"any", type="any")

    """

    def __init__(self, *, src: str,
                 dst: str,
                 type: str):
        super().__init__(_FILTER_NAME)
        super().__setattr__("filter_parameters", {})
        self.filter_parameters[_SRC] = src
        self.filter_parameters[_DST] = dst
        self.filter_parameters[_TYPE] = type
        self._param_tag2str = {
            _SRC: "src",
            _DST: "dst",
            _TYPE: "type"
        }
        self._update_filter()

    def _update_filter(self):
        enable_flag = self.enable
        self.clear_children()
        properties = {}
        # ===============src================== #
        if self.filter_parameters[_SRC] == _ANY:
            properties[_SRC] = {_ANY: 1}
        # ===============dst================== #
        if self.filter_parameters[_DST] == _ANY:
            properties[_DST] = {_ANY: 1}
        # ===============EtherType================== #
        if self.filter_parameters[_TYPE] == _ANY:
            properties[_TYPE] = {_ANY: 1}
        # ===============Action================== #
        # TODO: has path
        self.add_children(self.filter_parameters)
        self.set_properties(properties)
        if enable_flag:
            self.enable_filter()
        else:
            self.disable_filter()

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'MACFilter':
        if node.find(_SRC).get(_ANY) == "1":
            src = _ANY
        else:
            src = node.findtext(_SRC)
        if node.find(_DST).get(_ANY) == "1":
            dst = _ANY
        else:
            dst = node.findtext(_DST)
        if node.find(_TYPE).get(_ANY) == "1":
            type = _ANY
        else:
            type = node.findtext(_TYPE)
        mac = MACFilter(
            src=src,
            dst=dst,
            type=type,
        )
        enable = True if node.get("enable") == "1" else False
        if not enable:
            mac.disable_filter()
        return mac


if __name__ == '__main__':
    mac = MACFilter(src="39-A9-FA-D7-F5-E8", dst=r"any", type="any")
    print(mac.xml)

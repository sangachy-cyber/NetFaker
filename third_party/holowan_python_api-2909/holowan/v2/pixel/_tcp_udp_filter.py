"""
HoloWAN Network Emulator
Python API
"""

import functools
import xml.etree.cElementTree as ET
from typing import Union, Callable

import holowan.v2.utils.my_util as mt
import holowan.v2.utils.xml_util as xt
from holowan.v2 import (
    PortNumber,
    PortNumberRange,
    PortNumberList,
    check_parameter
)
from holowan.v2.pixel._filter_base import Filter

_TCP_FILTER = 1
_UDP_FILTER = 2
_SCTP_FILTER = 3

# ==================== TCP,UDP,STCP Key Begin=======================
_FILTER_NAME: str = r"tcp_udp"
_PORT: str = r"port"
_ANY: str = r"any"
_TYPE: str = r"type"
_SRC: str = r"src"
_DST: str = r"dst"
_CHECK_VERSION: str = r"check"
_CHECK_VERSION_SET = [0, 4, 6]

_tcp_udp_sctp_tag2str_dict = {
    _SRC: "src",
    _DST: "dst",
    _CHECK_VERSION: "check_version",
}


# ==================== TCP,UDP,STCP Key End ========================

def _check_tcp_udp_stcp_parameters(function: Callable):
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
            elif k == _CHECK_VERSION and v not in _CHECK_VERSION_SET:
                raise TypeError(r"Argument {argument!r} must be one of {value_set!r}, "
                                r"but got {value!r}".format(argument=k, value_set=_CHECK_VERSION_SET, value=v))
            else:
                # TODO:has_path
                new_kwargs[k] = v
        return function(*tuple(args), **dict(new_kwargs))

    return wrapper


_TCP_UDP_STCP_XML_FORMAT = """
<tcp_udp>                    /* 关于传输层（TCP or UDP）的报文分类 */
    <type>1</type>           /* 指定传输层类型，[1. TCP，2. UDP]*/
    <src any="1"/>           /* 匹配传输层的源始端口号*/
    <dst any="1"/>           /* 匹配传输层的目标端口号*/
    <check>0</check>         /* 是否检查传输层类型：
                                0. 不检查，4. 匹配IPv4，6. 匹配IPv6。*/
    <path_id></path_id>      /* 命中分类后进入的虚拟链路ID，取值范围：1-15编号 */
</tcp_udp>
"""


class TCPFilter(Filter):
    """TCP分类规则。

    Note:
        强制关键字传参。

    Args:
        src (Union[PortNumber, PortNumberRange, PortNumberList]): 源端口号。可以传入单个端口号，也可以通过列表传入多个端口号或端口号范围。
        dst(Union[PortNumber, PortNumberRange, PortNumberList]): 目的端口号。可以传入单个端口号，也可以通过列表传入多个端口号或端口号范围。
        check_version(int): 检查版本

    Examples:
        >>> tcp = TCPFilter(src=[123, "456", "578-999"], dst=["12", 56, "589-999", "any"], check_version=0)

    """
    _type: int = _TCP_FILTER

    @_check_tcp_udp_stcp_parameters
    def __init__(self, *, src: Union[PortNumber, PortNumberRange, PortNumberList],
                 dst: Union[PortNumber, PortNumberRange, PortNumberList],
                 check_version: int):
        self.filter_parameters = {}
        super().__init__(_FILTER_NAME, self.filter_parameters)
        self.filter_parameters[_SRC] = [src] if not isinstance(src, list) else src
        self.filter_parameters[_DST] = [dst] if not isinstance(dst, list) else dst
        self.filter_parameters[_CHECK_VERSION] = check_version
        self._param_tag2str = _tcp_udp_sctp_tag2str_dict
        self._update_filter()

    def _update_filter(self):

        tcpudp_node = self
        children_node_map = {
            _TYPE: self._type,
            _SRC: "", _DST: "",
            _CHECK_VERSION: self.filter_parameters[_CHECK_VERSION],
        }
        properties = {}
        # ===============src================== #
        if _ANY in self.filter_parameters[_SRC]:
            properties[_SRC] = {_ANY: 1}
        else:
            src_cnt = len(self.filter_parameters[_SRC])
            properties[_SRC] = {"num": src_cnt}
            children_node_map[_SRC] = {}
            for idx, val in enumerate(self.filter_parameters[_SRC]):
                children_node_map[_SRC].update({_PORT + str(idx + 1): val})
        # ===============dst================== #
        if _ANY in self.filter_parameters[_DST]:
            properties[_DST] = {_ANY: 1}
        else:
            dst_cnt = len(self.filter_parameters[_DST])
            properties[_DST] = {"num": dst_cnt}
            children_node_map[_DST] = {}
            for idx, val in enumerate(self.filter_parameters[_DST]):
                children_node_map[_DST].update({_PORT + str(idx + 1): val})
        # ===============Action================== #
        # TODO: has path

        xt.add_children(tcpudp_node, children_node_map)
        xt.add_properties(tcpudp_node, properties)

    @staticmethod
    def construct_from_node(node: ET.Element):
        src_node = node.find(_SRC)
        if src_node.get(_ANY) == "1":
            src = _ANY
        else:
            src = []
            for port in list(src_node):
                src.append(port.text)
        dst_node = node.find(_DST)
        if dst_node.get(_ANY) == "1":
            dst = _ANY
        else:
            dst = []
            for port in list(dst_node):
                dst.append(port.text)

        enable = True if node.get("enable") == "1" else False
        tcp = TCPFilter(
            src=src, dst=dst,
            check_version=int(node.findtext(_CHECK_VERSION)),
        )
        if not enable:
            tcp.disable_filter()
        return tcp


class UDPFilter(Filter):
    """UDP分类规则。

    Note:
        强制关键字传参。

    Args:
        src (Union[PortNumber, PortNumberRange, PortNumberList]): 源端口号。可以传入单个端口号，也可以通过列表传入多个端口号或端口号范围。
        dst(Union[PortNumber, PortNumberRange, PortNumberList]): 目的端口号。可以传入单个端口号，也可以通过列表传入多个端口号或端口号范围。
        check_version(int): 检查版本

    Examples:
        >>> udp = UDPFilter(src=[123, "456", "578-999"], dst=["12", 56, "589-999", "any"], check_version=0)

    """
    _type: int = _UDP_FILTER

    @_check_tcp_udp_stcp_parameters
    def __init__(self, *, src: Union[PortNumber, PortNumberRange, PortNumberList],
                 dst: Union[PortNumber, PortNumberRange, PortNumberList],
                 check_version: int):
        self.filter_parameters = {}
        super().__init__(_FILTER_NAME, self.filter_parameters)
        self.filter_parameters[_SRC] = [src] if not isinstance(src, list) else src
        self.filter_parameters[_DST] = [dst] if not isinstance(dst, list) else dst
        self.filter_parameters[_CHECK_VERSION] = check_version
        self._param_tag2str = _tcp_udp_sctp_tag2str_dict
        self._update_filter()

    # TODO: enable tunnel

    @check_parameter
    def _update_filter(self):
        self.clear_children()
        children_node_map = {
            _TYPE: self._type,
            _SRC: "", _DST: "",
            _CHECK_VERSION: self.filter_parameters[_CHECK_VERSION],
        }
        properties = {}
        # ===============src================== #
        if _ANY in self.filter_parameters[_SRC]:
            properties[_SRC] = {_ANY: 1}
        else:
            src_cnt = len(self.filter_parameters[_SRC])
            properties[_SRC] = {"num": src_cnt}
            children_node_map[_SRC] = {}
            for idx, val in enumerate(self.filter_parameters[_SRC]):
                children_node_map[_SRC].update({_PORT + str(idx + 1): val})
        # ===============dst================== #
        if _ANY in self.filter_parameters[_DST]:
            properties[_DST] = {_ANY: 1}
        else:
            dst_cnt = len(self.filter_parameters[_DST])
            properties[_DST] = {"num": dst_cnt}
            children_node_map[_DST] = {}
            for idx, val in enumerate(self.filter_parameters[_DST]):
                children_node_map[_DST].update({_PORT + str(idx + 1): val})
        # ===============Action================== #
        # TODO: has path
        xt.add_children(self, children_node_map)
        xt.add_properties(self, properties)

    @staticmethod
    def construct_from_node(node: ET.Element):
        src_node = node.find(_SRC)
        if src_node.get(_ANY) == "1":
            src = _ANY
        else:
            src = []
            for port in list(src_node):
                src.append(port.text)
        dst_node = node.find(_DST)
        if dst_node.get(_ANY) == "1":
            dst = _ANY
        else:
            dst = []
            for port in list(dst_node):
                src.append(port.text)

        enable = True if node.get("enable") == "1" else False
        udp = UDPFilter(
            src=src, dst=dst,
            check_version=int(node.findtext(_CHECK_VERSION)),
        )
        if not enable:
            udp.disable_filter()
        return udp


class SCTPFilter(Filter):
    """SCTP分类规则。

    Note:
        强制关键字传参。

    Args:
        src (Union[PortNumber, PortNumberRange, PortNumberList]): 源端口号。可以传入单个端口号，也可以通过列表传入多个端口号或端口号范围。
        dst(Union[PortNumber, PortNumberRange, PortNumberList]): 目的端口号。可以传入单个端口号，也可以通过列表传入多个端口号或端口号范围。
        check_version(int): 检查版本

    Examples:
        >>> stcp = SCTPFilter(src=[123, "456", "578-999"], dst=["12", 56, "589-999", "any"], check_version=0)
    """
    _type: int = _SCTP_FILTER

    @_check_tcp_udp_stcp_parameters
    def __init__(self, *, src: Union[PortNumber, PortNumberRange, PortNumberList],
                 dst: Union[PortNumber, PortNumberRange, PortNumberList],
                 check_version: int):
        super().__init__(_FILTER_NAME)
        super().__setattr__("filter_parameters", {})
        self.filter_parameters[_SRC] = [src] if not isinstance(src, list) else src
        self.filter_parameters[_DST] = [dst] if not isinstance(dst, list) else dst
        self.filter_parameters[_CHECK_VERSION] = check_version
        self._param_tag2str = _tcp_udp_sctp_tag2str_dict
        self._update_filter()

    @check_parameter
    def _update_filter(self):
        enable_flag = self.enable
        self.clear_children()
        children_node_map = {
            _TYPE: self._type,
            _SRC: "", _DST: "",
            _CHECK_VERSION: self.filter_parameters[_CHECK_VERSION],
        }
        properties = {}
        # ===============src================== #
        if _ANY in self.filter_parameters[_SRC]:
            properties[_SRC] = {_ANY: 1}
        else:
            src_cnt = len(self.filter_parameters[_SRC])
            properties[_SRC] = {"num": src_cnt}
            children_node_map[_SRC] = {}
            for idx, val in enumerate(self.filter_parameters[_SRC]):
                children_node_map[_SRC].update({_PORT + str(idx + 1): val})
        # ===============dst================== #
        if _ANY in self.filter_parameters[_DST]:
            properties[_DST] = {_ANY: 1}
        else:
            dst_cnt = len(self.filter_parameters[_DST])
            properties[_DST] = {"num": dst_cnt}
            children_node_map[_DST] = {}
            for idx, val in enumerate(self.filter_parameters[_DST]):
                children_node_map[_DST].update({_PORT + str(idx + 1): val})
        # ===============Action================== #
        # TODO: has path
        self.add_children(children_node_map)
        self.set_properties(properties)
        if enable_flag:
            self.enable_filter()
        else:
            self.disable_filter()

    @staticmethod
    def construct_from_node(node: ET.Element):
        src_node = node.find(_SRC)
        if src_node.get(_ANY) == "1":
            src = _ANY
        else:
            src = []
            for port in list(src_node):
                src.append(port.text)
        dst_node = node.find(_DST)
        if dst_node.get(_ANY) == "1":
            dst = _ANY
        else:
            dst = []
            for port in list(dst_node):
                src.append(port.text)

        enable = True if node.get("enable") == "1" else False
        sctp = SCTPFilter(
            src=src, dst=dst,
            check_version=int(node.findtext(_CHECK_VERSION)),
        )
        if not enable:
            sctp.disable_filter()
        return sctp


if __name__ == '__main__':
    tcp = TCPFilter(src=[123, "456", "578-999"], dst=["12", 56, "589-999", "any"], check_version=0)
    stcp = SCTPFilter(src=[123, "456", "578-999"], dst=["12", 56, "589-999", "any"], check_version=0)
    print(tcp)
    tcp2 = TCPFilter.construct_from_node(tcp)
    print("construct", tcp2)
    print(tcp.xml)
    # print(issubclass(TCPFilter, Filter))

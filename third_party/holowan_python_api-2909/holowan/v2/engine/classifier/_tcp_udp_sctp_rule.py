"""
HoloWAN Network Emulator
Python API

Packet Classifier: TCP, UDP, STCP Rule
"""
import functools
import xml.etree.cElementTree as ET
from typing import Union, Callable

import holowan.v2.utils.my_util as mt
import holowan.v2.utils.xml_util as xt
from holowan.v2._holowan_types import (
    PortNumber,
    PortNumberRange,
    PortNumberList,
    PathID,
    check_parameter
)
from holowan.v2.engine.classifier._rule_base import Rule, _RULE_ACTION, HoloWANRuleTag

_TCP_RULE = 1
_UDP_RULE = 2
_SCTP_RULE = 3

# ==================== TCP,UDP,SCTP Key Begin=======================
_RULE_NAME: str = HoloWANRuleTag.TCP_UDP_SCTP
_PORT: str = r"port"
_ANY: str = r"any"
_TYPE: str = r"type"
_SRC: str = r"src"
_DST: str = r"dst"
_CHECK_VERSION: str = r"check"
_CHECK_VERSION_SET = [0, 4, 6]
_ACTION: str = _RULE_ACTION

_TCP_UDP_SCTP_DEFAULT_CUSTOM_NAME: str = r"{}: {} to {}"

_tcp_udp_sctp_tag2str_dict = {
    _SRC: "src",
    _DST: "dst",
    _CHECK_VERSION: "check_version",
    _ACTION: "action"
}


# ==================== TCP,UDP,STCP Key End ========================

def _check_tcp_udp_sctp_parameters(function: Callable):
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
                        raise ValueError(r"Argument {argument!r} is not a valid port number, got {got!r}".format(
                            argument=k, got=v
                        ))
                    else:
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
                                raise ValueError(("The value:{value!r} in the {argument!r} port list "
                                                  "is not a valid port number").format(argument=k, value=pt))
                else:
                    raise TypeError('Argument {argument} must be {expected!r},but got {got!r},value {value!r}'.format(
                        argument=k, expected="PortNumber, PortNumberRange, List[PortNumber]",
                        got="unknown type", value=v
                    ))
            elif k == _CHECK_VERSION and v not in _CHECK_VERSION_SET:
                raise ValueError(r"Argument {argument!r} must be one of {value_set!r}, "
                                 r"but got {value!r}".format(argument=k, value_set=_CHECK_VERSION_SET, value=v))
            else:
                # TODO:has_path
                new_kwargs[k] = v
        return function(*tuple(args), **dict(new_kwargs))

    return wrapper


_TCP_UDP_SCTP_XML_FORMAT = """
<tcp_udp>                    /* 关于传输层（TCP or UDP）的报文分类 */
    <type>1</type>           /* 指定传输层类型，[1. TCP，2. UDP]*/
    <src any="1"/>           /* 匹配传输层的源始端口号*/
    <dst any="1"/>           /* 匹配传输层的目标端口号*/
    <check>0</check>         /* 是否检查传输层类型：
                                0. 不检查，4. 匹配IPv4，6. 匹配IPv6。*/
    <path_id></path_id>      /* 命中分类后进入的虚拟链路ID，取值范围：1-15编号 */
</tcp_udp>
"""


class TCPRule(Rule):
    """TCP分类规则。

    Note:
        强制关键字传参。

    Args:
        src (Union[PortNumber, PortNumberRange, PortNumberList]): 源端口号。可以传入单个端口号，也可以通过列表传入多个端口号或端口号范围。
        dst(Union[PortNumber, PortNumberRange, PortNumberList]): 目的端口号。可以传入单个端口号，也可以通过列表传入多个端口号或端口号范围。
        check_version(int): 检查版本
        action(PathID): 被此规则匹配的报文将被转发到的path id。

    Examples:
        >>> tcp = TCPRule(src=[123, "456", "578-999"], dst=["12", 56, "589-999", "any"], check_version=0, action=1)

    """
    _type: int = _TCP_RULE

    @_check_tcp_udp_sctp_parameters
    def __init__(self, *, src: Union[PortNumber, PortNumberRange, PortNumberList],
                 dst: Union[PortNumber, PortNumberRange, PortNumberList],
                 check_version: int, action: PathID):
        self.rule_parameters = {}
        super().__init__(_RULE_NAME, self.rule_parameters)
        self.rule_parameters[_SRC] = [src] if not isinstance(src, list) else src
        self.rule_parameters[_DST] = [dst] if not isinstance(dst, list) else dst
        self.rule_parameters[_CHECK_VERSION] = check_version
        self.rule_parameters[_ACTION] = action
        self._param_tag2str = _tcp_udp_sctp_tag2str_dict
        self._update_rule()
        self.set_custom_name(_TCP_UDP_SCTP_DEFAULT_CUSTOM_NAME.format(
            r"TCP",
            ",".join(str(x) for x in self.rule_parameters[_SRC]),
            ",".join(str(x) for x in self.rule_parameters[_DST]))
        )

    @property
    def action(self) -> PathID:
        """被此规则匹配的报文将被转发到的path id。

        Note:
            可读可写。

        """
        return self.rule_parameters[_ACTION]

    @action.setter
    @check_parameter
    def action(self, value: PathID) -> None:
        self.rule_parameters[_ACTION] = value
        self._update_rule()

    @property
    def src(self) -> Union[PortNumber, PortNumberRange, PortNumberList]:
        return self.rule_parameters[_SRC]

    @src.setter
    def src(self, value: Union[PortNumber, PortNumberRange, PortNumberList]) -> None:
        self.rule_parameters[_SRC] = value
        self._update_rule()

    @property
    def dst(self) -> Union[PortNumber, PortNumberRange, PortNumberList]:
        return self.rule_parameters[_DST]

    @dst.setter
    def dst(self, value: Union[PortNumber, PortNumberRange, PortNumberList]) -> None:
        self.rule_parameters[_DST] = value
        self._update_rule()

    @property
    def check_version(self) -> int:
        return self.rule_parameters[_CHECK_VERSION]

    @check_version.setter
    def check_version(self, value: int) -> None:
        self.rule_parameters[_CHECK_VERSION] = value
        self._update_rule()

    def _update_rule(self):

        tcpudp_node = self
        children_node_map = {
            _TYPE: self._type,
            _SRC: "", _DST: "",
            _CHECK_VERSION: self.rule_parameters[_CHECK_VERSION],
            _ACTION: self.rule_parameters[_ACTION]}
        properties = {}
        # ===============src================== #
        if _ANY in self.rule_parameters[_SRC]:
            properties[_SRC] = {_ANY: 1}
        else:
            src_cnt = len(self.rule_parameters[_SRC])
            properties[_SRC] = {"num": src_cnt}
            children_node_map[_SRC] = {}
            for idx, val in enumerate(self.rule_parameters[_SRC]):
                children_node_map[_SRC].update({_PORT + str(idx + 1): val})
        # ===============dst================== #
        if _ANY in self.rule_parameters[_DST]:
            properties[_DST] = {_ANY: 1}
        else:
            dst_cnt = len(self.rule_parameters[_DST])
            properties[_DST] = {"num": dst_cnt}
            children_node_map[_DST] = {}
            for idx, val in enumerate(self.rule_parameters[_DST]):
                children_node_map[_DST].update({_PORT + str(idx + 1): val})
        # ===============Action================== #
        # TODO: has path

        xt.add_children(tcpudp_node, children_node_map)
        xt.add_properties(tcpudp_node, properties)

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'TCPRule':
        """由一个 TCP 分类规则的 xml 节点构造 TCPRule 对象。

        Args:
            node (ET.Element): TCP 分类规则的 xml 节点。

        Returns:
            TCPRule 对象。

        """
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
        rule = TCPRule(
            src=src, dst=dst,
            check_version=int(node.findtext(_CHECK_VERSION)),
            action=int(node.findtext(_ACTION))
        )
        # 由于 WEB GUI 2.0 支持设置分类规则的 label(是 xml 节点的一个属性)
        # 所以此处处理 xml 节点中的属性, 直接拿过来就行
        # 如果用户想要通过 API 去改规则的名字, 通过新增的 set_custom_name (实现在规则基类 Rule)
        rule.attrib = node.attrib
        return rule


class UDPRule(Rule):
    """UDP分类规则。

    Note:
        强制关键字传参。

    Args:
        src (Union[PortNumber, PortNumberRange, PortNumberList]): 源端口号。可以传入单个端口号，也可以通过列表传入多个端口号或端口号范围。
        dst(Union[PortNumber, PortNumberRange, PortNumberList]): 目的端口号。可以传入单个端口号，也可以通过列表传入多个端口号或端口号范围。
        check_version(int): 检查版本
        action(PathID): 被此规则匹配的报文将被转发到的path id。

    Examples:
        >>> udp = UDPRule(src=[123, "456", "578-999"], dst=["12", 56, "589-999", "any"], check_version=0, action=1)

    """
    _type: int = _UDP_RULE

    @_check_tcp_udp_sctp_parameters
    def __init__(self, *, src: Union[int, str, PortNumberList],
                 dst: Union[int, str, PortNumberList],
                 check_version: int, action: PathID):
        self.rule_parameters = {}
        super().__init__(_RULE_NAME, self.rule_parameters)
        self.rule_parameters[_SRC] = [src] if not isinstance(src, list) else src
        self.rule_parameters[_DST] = [dst] if not isinstance(dst, list) else dst
        self.rule_parameters[_CHECK_VERSION] = check_version
        self.rule_parameters[_ACTION] = action
        self._param_tag2str = _tcp_udp_sctp_tag2str_dict
        self._update_rule()
        self.set_custom_name(_TCP_UDP_SCTP_DEFAULT_CUSTOM_NAME.format(
            r"UDP",
            ",".join(str(x) for x in self.rule_parameters[_SRC]),
            ",".join(str(x) for x in self.rule_parameters[_DST]))
        )

    @property
    def action(self) -> PathID:
        """被此规则匹配的报文将被转发到的path id。

        Note:
            可读可写。

        """
        return self.rule_parameters[_ACTION]

    @action.setter
    @check_parameter
    def action(self, value: PathID) -> None:
        self.rule_parameters[_ACTION] = value
        self._update_rule()

    @property
    def src(self) -> Union[PortNumber, PortNumberRange, PortNumberList]:
        return self.rule_parameters[_SRC]

    @src.setter
    def src(self, value: Union[PortNumber, PortNumberRange, PortNumberList]) -> None:
        self.rule_parameters[_SRC] = value
        self._update_rule()

    @property
    def dst(self) -> Union[PortNumber, PortNumberRange, PortNumberList]:
        return self.rule_parameters[_DST]

    @dst.setter
    def dst(self, value: Union[PortNumber, PortNumberRange, PortNumberList]) -> None:
        self.rule_parameters[_DST] = value
        self._update_rule()

    @property
    def check_version(self) -> int:
        return self.rule_parameters[_CHECK_VERSION]

    @check_version.setter
    def check_version(self, value: int) -> None:
        self.rule_parameters[_CHECK_VERSION] = value
        self._update_rule()

    # TODO: enable tunnel

    @check_parameter
    def _update_rule(self):
        self.clear_children()
        children_node_map = {
            _TYPE: self._type,
            _SRC: "", _DST: "",
            _CHECK_VERSION: self.rule_parameters[_CHECK_VERSION],
            _ACTION: self.rule_parameters[_ACTION]}
        properties = {}
        # ===============src================== #
        if _ANY in self.rule_parameters[_SRC]:
            properties[_SRC] = {_ANY: 1}
        else:
            src_cnt = len(self.rule_parameters[_SRC])
            properties[_SRC] = {"num": src_cnt}
            children_node_map[_SRC] = {}
            for idx, val in enumerate(self.rule_parameters[_SRC]):
                children_node_map[_SRC].update({_PORT + str(idx + 1): val})
        # ===============dst================== #
        if _ANY in self.rule_parameters[_DST]:
            properties[_DST] = {_ANY: 1}
        else:
            dst_cnt = len(self.rule_parameters[_DST])
            properties[_DST] = {"num": dst_cnt}
            children_node_map[_DST] = {}
            for idx, val in enumerate(self.rule_parameters[_DST]):
                children_node_map[_DST].update({_PORT + str(idx + 1): val})
        # ===============Action================== #
        # TODO: has path
        xt.add_children(self, children_node_map)
        xt.add_properties(self, properties)

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'UDPRule':
        """由一个 UDP 分类规则的 xml 节点构造 UDPRule 对象。

        Args:
            node (ET.Element): UDP 分类规则的 xml 节点。

        Returns:
            UDPRule 对象。
        """
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
        rule = UDPRule(
            src=src, dst=dst,
            check_version=int(node.findtext(_CHECK_VERSION)),
            action=int(node.findtext(_ACTION))
        )
        # 由于 WEB GUI 2.0 支持设置分类规则的 label(是 xml 节点的一个属性)
        # 所以此处处理 xml 节点中的属性, 直接拿过来就行
        # 如果用户想要通过 API 去改规则的名字, 通过新增的 set_custom_name (实现在规则基类 Rule)
        rule.attrib = node.attrib
        return rule


class SCTPRule(Rule):
    """STCP分类规则。

    Note:
        强制关键字传参。

    Args:
        src (Union[PortNumber, PortNumberRange, PortNumberList]): 源端口号。可以传入单个端口号，也可以通过列表传入多个端口号或端口号范围。
        dst(Union[PortNumber, PortNumberRange, PortNumberList]): 目的端口号。可以传入单个端口号，也可以通过列表传入多个端口号或端口号范围。
        check_version(int): 检查版本
        action(PathID): 被此规则匹配的报文将被转发到的path id。

    Examples:
        >>> sctp = SCTPRule(src=[123, "456", "578-999"], dst=["12", 56, "589-999", "any"], check_version=0, action=1)
    """
    _type: int = _SCTP_RULE

    @_check_tcp_udp_sctp_parameters
    def __init__(self, *, src: Union[PortNumber, PortNumberRange, PortNumberList],
                 dst: Union[PortNumber, PortNumberRange, PortNumberList],
                 check_version: int, action: PathID):
        super().__init__(_RULE_NAME)
        super().__setattr__("rule_parameters", {})
        self.rule_parameters[_SRC] = [src] if not isinstance(src, list) else src
        self.rule_parameters[_DST] = [dst] if not isinstance(dst, list) else dst
        self.rule_parameters[_CHECK_VERSION] = check_version
        self.rule_parameters[_ACTION] = action
        self._param_tag2str = _tcp_udp_sctp_tag2str_dict
        self._update_rule()
        self.set_custom_name(_TCP_UDP_SCTP_DEFAULT_CUSTOM_NAME.format(
            r"SCTP",
            ",".join(str(x) for x in self.rule_parameters[_SRC]),
            ",".join(str(x) for x in self.rule_parameters[_DST]))
        )

    @property
    def action(self) -> PathID:
        """被此规则匹配的报文将被转发到的path id。

        Note:
            可读可写。

        """
        return self.rule_parameters[_ACTION]

    @action.setter
    @check_parameter
    def action(self, value: PathID) -> None:
        self.rule_parameters[_ACTION] = value
        self._update_rule()

    @property
    def src(self) -> Union[PortNumber, PortNumberRange, PortNumberList]:
        return self.rule_parameters[_SRC]

    @src.setter
    def src(self, value: Union[PortNumber, PortNumberRange, PortNumberList]) -> None:
        self.rule_parameters[_SRC] = value
        self._update_rule()

    @property
    def dst(self) -> Union[PortNumber, PortNumberRange, PortNumberList]:
        return self.rule_parameters[_DST]

    @dst.setter
    def dst(self, value: Union[PortNumber, PortNumberRange, PortNumberList]) -> None:
        self.rule_parameters[_DST] = value
        self._update_rule()

    @property
    def check_version(self) -> int:
        return self.rule_parameters[_CHECK_VERSION]

    @check_version.setter
    def check_version(self, value: int) -> None:
        self.rule_parameters[_CHECK_VERSION] = value
        self._update_rule()

    @check_parameter
    def _update_rule(self):
        self.clear_children()
        children_node_map = {
            _TYPE: self._type,
            _SRC: "", _DST: "",
            _CHECK_VERSION: self.rule_parameters[_CHECK_VERSION],
            _ACTION: self.rule_parameters[_ACTION]}
        properties = {}
        # ===============src================== #
        if _ANY in self.rule_parameters[_SRC]:
            properties[_SRC] = {_ANY: 1}
        else:
            src_cnt = len(self.rule_parameters[_SRC])
            properties[_SRC] = {"num": src_cnt}
            children_node_map[_SRC] = {}
            for idx, val in enumerate(self.rule_parameters[_SRC]):
                children_node_map[_SRC].update({_PORT + str(idx + 1): val})
        # ===============dst================== #
        if _ANY in self.rule_parameters[_DST]:
            properties[_DST] = {_ANY: 1}
        else:
            dst_cnt = len(self.rule_parameters[_DST])
            properties[_DST] = {"num": dst_cnt}
            children_node_map[_DST] = {}
            for idx, val in enumerate(self.rule_parameters[_DST]):
                children_node_map[_DST].update({_PORT + str(idx + 1): val})
        # ===============Action================== #
        # TODO: has path
        self.add_children(children_node_map)
        self.set_properties(properties)

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'SCTPRule':
        """由一个 SCTP 分类规则的 xml 节点构造 SCTPRule 对象。

        Args:
            node (ET.Element): SCTP 分类规则的 xml 节点。

        Returns:
            SCTPRule 对象。

        """
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
        rule = SCTPRule(
            src=src, dst=dst,
            check_version=int(node.findtext(_CHECK_VERSION)),
            action=int(node.findtext(_ACTION))
        )
        # 由于 WEB GUI 2.0 支持设置分类规则的 label(是 xml 节点的一个属性)
        # 所以此处处理 xml 节点中的属性, 直接拿过来就行
        # 如果用户想要通过 API 去改规则的名字, 通过新增的 set_custom_name (实现在规则基类 Rule)
        rule.attrib = node.attrib
        return rule

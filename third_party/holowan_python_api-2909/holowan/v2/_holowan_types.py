"""
HoloWAN Network Emulator
Python API dtype
"""
import builtins
import functools
import inspect
from typing import TypeVar, List, Callable, Dict, Union, NewType

from holowan.v2.utils import my_util as mt

__all__ = [
    "IPv4Address", "IPv6Address", "IPAddress", "PortNumber", "PortNumberRange",
    "PortNumberList", "EngineID", "PathID", "PathName", "HoloWANReturnXML", "HoloWANConfigXML", "check_parameter",
    "MACAddress", "MACAddressList" ,"HoloWANReturn"
]

# IP 地址类型
IPv6Addr = NewType('IPv6Address', str)
IPv4Addr = NewType('IPv4Address', str)
IPv4Address = Union[str,IPv4Addr]
IPv6Address = Union[str,IPv6Addr]
# IPAddress = TypeVar("IPAddress", IPv4Address, IPv6Address)
# IPv6Address = builtins.str
# IPv4Address = builtins.str
IPAddress = TypeVar("IPAddress", IPv4Address, IPv6Address)

# MAC
MACAddress = builtins.str
MACAddressList = List[MACAddress]

# 端口号类型
PortNumberStr = builtins.str
PortNumberInt = builtins.int
PortNumber = Union[PortNumberStr, PortNumberInt]

# 端口号范围
PortNumberRange = builtins.str
PortNumberList = List[PortNumber]

# engine ID类型
EngineID = builtins.int

# Path
PathID = builtins.int
PathName = builtins.str

# HoloWAN function returns
HoloWANReturnXML = builtins.str
HoloWANConfigXML = builtins.str


class HoloWANReturn(object):
    """HoloWAN returns info parser.
    The HoloWAN returns info is a string type. This class can parse the return string.

    Args:
        result (str): The HoloWAN return string.
    """

    def __init__(self, result: str) -> None:
        self._ori_data = result
        _res_dict: dict = eval(result)
        # There are so many types of HoloWAN return
        # for example:
        # 1: {"code": xxx, "msg":xxx}
        # 2: {"errCode": xxx, "errMsg": xxx, "errReason": xxx}
        # 3: {"code": xxx, "msg": xxx, "passwordType": xxx, "type": xxx}
        # So far, the seen return value of holowan has "code" and "msg"
        # This class regards the 2nd type as the standard.
        # So: errCode = code; errMsg = msg; errReason = errReason/msg.
        # Using the errCode to implement __bool__, so the return value can be used just like a normal bool.
        # The rest info(if exist) will not be processed, just store in self._ori_data(readonly property: data).
        if _res_dict.get("code") is not None:
            self.res_dict = {}
            self.res_dict["errCode"] = _res_dict["code"]
            self.res_dict["errMsg"] = _res_dict["msg"]
            self.res_dict["errReason"] = _res_dict["msg"]
        else:
            self.res_dict: Dict = eval(result)

    def __bool__(self) -> bool:
        return int(self.res_dict["errCode"]) >= 0

    def __str__(self) -> str:
        return self._ori_data

    @property
    def data(self) -> str:
        return self._ori_data

    @property
    def err_code(self) -> int:
        return int(self.res_dict["errCode"])

    @property
    def err_msg(self) -> str:
        return self.res_dict["errMsg"]

    @property
    def err_reason(self) -> str:
        return self.res_dict["errReason"]


def check_param_decimal(name: str, value, decimal: int):
    # 检查数值是否满足给定的小数位精度
    if isinstance(value, int):
        pass
    elif isinstance(value, float):
        if not len(str(value).split(".")[1]) <= int(decimal):
            raise ValueError(
                r"Argument {argument!r} supports {expect!r}, got {got!r}.".format(
                    argument=name, expect=decimal,
                    got=value
                ))
    else:
        error = "Argument {argument} must be {expected!r}, but got {got!r}, value {value!r}".format(
            argument=name, expected="number", got=type(value), value=value
        )
        raise TypeError(error)


def check_param_in_range(name: str, value, minv, maxv, closed_interval: bool):
    # 检查一个数值是否在一个范围内（开闭区间）
    if closed_interval:
        if value < minv or value > maxv:
            raise ValueError(
                r"Argument {argument!r} must >={minv!r} and <={maxv!r}, but got {got!r}.".format(
                    argument=name, minv=minv, maxv=maxv, got=value
                )
            )
    else:
        if value <= minv or value >= maxv:
            raise ValueError(
                r"Argument {argument!r} must >{minv!r} and <{maxv!r}, but got {got!r}.".format(
                    argument=name, minv=minv, maxv=maxv, got=value
                )
            )


def check_param_in_choices(name: str, value, choices: list):
    # 检查名字为 name 的参数是否满足特定的值集合
    # 比如合法值是 [0,1,2,3,4] 中的一个数
    if isinstance(value, type(choices[0])):
        if value not in choices:
            raise ValueError(
                r"Argument {argument!r} must be one of {expect!r}, got {got!r}. See enable_reordering.".format(
                    argument=name, expect=choices,
                    got=value
                ))
    else:
        error = "Argument {argument} must be {expected!r}, but got {got!r}, value {value!r}".format(
            argument=name, expected=type(choices[0]), got=type(value), value=value
        )
        raise TypeError(error)


# 清理 IP 地址字符串，去掉协议前缀和 IPv6 中括号
def _clean_ip_address(ip_str: str) -> str:
    """
    清理 IP 地址字符串：
    1. 去掉 https:// 或 http:// 前缀
    2. 去掉 IPv6 地址的中括号
    """
    ipaddr = str(ip_str)
    # 去掉 https:// 或 http:// 前缀
    if "//" in ipaddr:
        ipaddr = ipaddr.split("//")[-1]
    # 去掉 IPv6 地址可能的中括号
    if ipaddr.startswith("[") and ipaddr.endswith("]"):
        ipaddr = ipaddr[1:-1]
    return ipaddr


# 检测参数类型的装饰器
# 这个装饰器除了基本的类型检查外, 增加了对 ip 和 port 的检查
def check_parameter(function: Callable):
    msg = 'Argument {argument} must be {expected!r}, but got {got!r}, value {value!r}'

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        new_args = list()
        new_kwargs = dict()
        sig = inspect.signature(function)  # 提取函数签名
        params = sig.parameters
        va = list(params.values())
        for arg, param in zip(args, va):
            if type(arg) == int and param.annotation == float:
                arg = float(arg)
            if param.annotation == IPAddress:
                # 清理 IP 地址（去掉协议前缀和 IPv6 中括号）
                cleaned_ip = _clean_ip_address(arg)
                if mt.isIP(cleaned_ip) == False:
                    error = msg.format(argument=param.name, expected=param.annotation.__name__,
                                       got=type(arg), value=arg)
                    raise TypeError(error)
                else:
                    new_args.append(arg)
                    continue
            if param.annotation == PortNumber:
                if mt.isPort(str(arg)) == False:
                    error = msg.format(argument=param.name, expected=param.annotation.__name__,
                                       got=type(arg), value=arg)
                    raise TypeError(error)
                else:
                    new_args.append(arg)
                    continue

            if param.annotation != inspect._empty and not isinstance(arg, param.annotation):
                error = msg.format(argument=param.name, expected=param.annotation.__name__,
                                   got=type(arg), value=arg)
                raise TypeError(error)

            new_args.append(arg)
        for k, v in kwargs.items():
            if isinstance(v, int) and params[k].annotation == float:
                v = float(v)
            if params[k].annotation == IPAddress:
                # 清理 IP 地址（去掉协议前缀和 IPv6 中括号）
                cleaned_ip = _clean_ip_address(v)
                if mt.isIP(cleaned_ip) == False:
                    error = msg.format(argument=params[k].name, expected=params[k].annotation.__name__,
                                       got=type(v), value=v)
                    raise TypeError(error)
            elif params[k].annotation == PortNumber:
                if mt.isPort(str(v)) == False:
                    error = msg.format(argument=params[k].name, expected=params[k].annotation.__name__,
                                       got=type(v), value=v)
                    raise TypeError(error)
            else:
                if params[k].annotation != inspect._empty and not isinstance(v, params[k].annotation):
                    error = msg.format(argument=params[k].name, expected=params[k].annotation.__name__,
                                       got=type(v), value=v)
                    raise TypeError(error)

            new_kwargs[k] = v
        # return function(*args, **kwargs)
        return function(*tuple(new_args), **dict(new_kwargs))

    return wrapper

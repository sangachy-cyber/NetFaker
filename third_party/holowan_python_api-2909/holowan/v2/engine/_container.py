"""
HoloWAN Network Emulator
Python API
"""
import operator
import xml.etree.cElementTree as ET
from collections import OrderedDict
from itertools import islice
from typing import overload, Union, Iterator, Dict, TypeVar, List

from holowan.v2._holowan_types import check_parameter
from holowan.v2._xml_holder import XMLHolder

T = TypeVar('T', bound=XMLHolder)


class Sequential(object):
    """序列容器
    能够将元素按照其在构造函数中传入的顺序，插入到容器中。

    Examples:

        >>> # 用作报文分类器的 Rule 容器
        >>> rules = Sequential(
        >>>           TCPRule(...),
        >>>           UDPRule(...),
        >>>         )
        >>>
        >>> # Using Sequential with OrderedDict. This is functionally the
        >>> # same as the above code
        >>> rules = Sequential(OrderedDict([
        >>>           ('tcp_rule', TCPRule(...)),
        >>>           ('udp_rule', UDPRule(...)),
        >>>         ]))
        >>>
        >>> # 用作引擎下的 Path 容器
        >>> paths = Sequential(
        >>>           Path(...),
        >>>           Path(...),
        >>>         )
    """

    _items: Dict[str, XMLHolder]  # type: ignore[assignment]

    @overload
    def __init__(self, *args: XMLHolder) -> None:
        ...

    @overload
    def __init__(self, arg: 'OrderedDict[str, XMLHolder]') -> None:
        ...

    def __init__(self, *args):
        super().__init__()
        self._items = OrderedDict()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, value in args[0].items():
                self.add_item(key, value)
        else:
            for idx, value in enumerate(args):
                self.add_item(str(idx), value)

    @check_parameter
    def add_item(self, key: str, item: XMLHolder):
        """增加元素到容器末尾

        Args:
            key (str): 键。
            item (XMLHolder): 值。

        """
        if not isinstance(item, XMLHolder) and item is not None:
            raise TypeError(f"{type(item)} is not a XMLHolder subclass")
        elif not isinstance(key, str):
            raise TypeError(f"rule name should be a string. Got {type(key)}")
        elif '.' in key:
            raise KeyError(f"rule name can't contain \".\", got: {key}")
        elif key == '':
            raise KeyError("rule name can't be empty string \"\"")
        self._items[key] = item

    def set_item(self,key:str,value:XMLHolder)->None:
        if key not in self._items.keys():
            raise KeyError("Unknown key: {0}".format(key))
        self._items[key] = value

    def get_item_by_key(self, key):
        """通过键取值

        Args:
            key (any): 键。

        Returns:
            XMLHolder子类
        """
        for k, v in self._items.items():
            if k == str(key):
                return v

    def remove_item_by_key(self,key:str)->None:
        self._items.pop(key)

    def _get_item_by_idx(self, iterator, idx) -> T:  # type: ignore[misc, type-var]
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError(f'index {idx} is out of range')
        idx %= size
        return next(islice(iterator, idx, None))

    def remove_item_by_idx(self,index:int):
        key = list(self._items.keys())[index]
        self._items.pop(key)

    def __getitem__(self, idx: Union[slice, int]) -> 'Sequential':
        if isinstance(idx, slice):
            return OrderedDict(list(self._items.items())[idx])
        else:
            return self._get_item_by_idx(self._items.values(), idx)

    def __len__(self) -> int:
        return len(self._items)

    def __add__(self, other) -> 'Sequential':
        if isinstance(other, Sequential):
            ret = Sequential()
            for layer in self:
                ret.append(layer)
            for layer in other:
                ret.append(layer)
            return ret
        else:
            raise ValueError('add operator supports only objects '
                             'of Sequential class, but {} is given.'.format(
                str(type(other))))

    def __iadd__(self, other) -> 'Sequential':
        if isinstance(other, Sequential):
            offset = len(self)
            for i, value in enumerate(other):
                self.add_item(str(i + offset), value)
            return self
        else:
            raise ValueError('add operator supports only objects '
                             'of Sequential class, but {} is given.'.format(
                str(type(other))))

    def __iter__(self) -> Iterator[XMLHolder]:
        return iter(self._items.values())

    @check_parameter
    def append(self, module: XMLHolder) -> 'Sequential':
        self.add_item(str(len(self)+1), module)
        return self

    @check_parameter
    def insert(self, index: int,item: XMLHolder) -> 'Sequential':
        """将元素插入到指定位置

        Args:
            index (int): 待插入位置。
            item(XMLHolder):待 插入元素。

        Returns:
            Sequential: 容器自身
        """
        if not isinstance(item, XMLHolder):
            raise AssertionError(
                f'Item should be of type: {XMLHolder}')
        n = len(self._items)
        if not (-n <= index <= n):
            raise IndexError(
                f'Index out of range: {index}')
        if index < 0:
            index += n

        itemls = list(self._items.values())[:index] + [item] + list(self._items.values())[index:]
        self._items = OrderedDict()
        for idx, value in enumerate(itemls):
            self.add_item(str(idx), value)
        return self

    def rearrange(self,order:list):
        new_one = OrderedDict()
        for idx in order:
            key = list(self._items.keys())[idx]
            value = self._items[key]
            new_one[key] = value

        self._items = new_one

    @property
    def nodes(self) -> List[ET.Element]:
        return [item.node for item in self._items.values()]

    def __str__(self):
        res = self.__class__.__name__ + ":[\n"
        for item in self._items.values():
            res += "\t" + item.__str__()+"\n"
        res += "]"
        return res


if __name__ == '__main__':
    # 延迟导入以避免循环引用
    from holowan.v2.engine.classifier import VLANRule
    from holowan.v2.engine.classifier._tcp_udp_sctp_rule import TCPRule, SCTPRule
    # print(issubclass(TCPRule,Rule))
    port1 = Sequential(
        TCPRule(src=[123, "456", "578-999"], dst=["12", 56, "589-999", "any"], check_version=0, action=1),
        SCTPRule(src=[123, "456", "578-999"], dst=["12", 56, "589-999", "any"], check_version=0, action=1)
    )
    port1.append(VLANRule(fpcp="any", fpid="any", action=1))
    port1.append(VLANRule(fpcp="any", fpid="any", action=1))
    print(port1)
    for rule in port1:
        print(rule)

"""
HoloWAN Network Emulator
Python API
"""

import xml.etree.cElementTree as ET

from holowan.v2._holowan_types import check_parameter
from holowan.v2.pixel._filter_base import Filter

_FILTER_NAME: str = r"length"
_LENGTH_FROM: str = r"min"
_LENGTH_TO: str = r"max"


class LengthFilter(Filter):
    r"""
        强制关键字传参
    """

    def __init__(self, *, length_from: int, length_to: int):
        super().__init__(_FILTER_NAME)
        super().__setattr__("filter_parameters", {})
        self.filter_parameters[_LENGTH_FROM] = length_from
        self.filter_parameters[_LENGTH_TO] = length_to
        self._param_tag2str = {
            _LENGTH_FROM: "length_from",
            _LENGTH_TO: "length_to"
        }
        self._update_filter()

    @property
    def length_from(self) -> int:
        return self.filter_parameters[_LENGTH_FROM]

    @length_from.setter
    @check_parameter
    def length_from(self, value: int) -> None:
        self.filter_parameters[_LENGTH_FROM] = value
        self.set_node_text(_LENGTH_FROM, self.filter_parameters[_LENGTH_FROM])

    @property
    def length_to(self) -> int:
        return self.filter_parameters[_LENGTH_TO]

    @length_to.setter
    @check_parameter
    def length_to(self, value: int) -> None:
        self.filter_parameters[_LENGTH_TO] = value
        self.set_node_text(_LENGTH_TO, self.filter_parameters[_LENGTH_TO])

    @staticmethod
    def construct_from_node(node: ET.Element):
        enable = True if node.get("enable") == "1" else False
        len_filter = LengthFilter(
            length_from=int(node.findtext(_LENGTH_FROM)), length_to=int(node.findtext(_LENGTH_TO))
        )
        if not enable:
            len_filter.disable_filter()

        return len_filter


if __name__ == '__main__':
    length = LengthFilter(length_from=0, length_to=2500)
    print(length)
    print(length.xml)

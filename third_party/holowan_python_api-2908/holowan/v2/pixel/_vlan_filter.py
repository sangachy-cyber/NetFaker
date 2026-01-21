"""
HoloWAN Network Emulator
Python API
"""

import xml.etree.cElementTree as ET

from holowan.v2.pixel._filter_base import Filter

_ANY: str = r"any"
_FILTER_NAME: str = r"vlan"
_FPCP: str = r"fpcp"
_FPID: str = r"fpid"
_ENABLE_STAG: str = r"enable_stag"
_SPCP: str = r"spcp"
_SPID: str = r"spid"


class VLANFilter(Filter):
    """VLAN 分类规则。

    Note:
        强制关键字传参。

    Args:
        fpcp (str): VLAN 的 fpcp。
        fpid(str): VLAN 的 fpid。

    Examples:
        >>> vlan = VLANFilter(fpcp="any", fpid="any")
        >>> vlan.enable_second_tag(spcp="any", spid="any")

    """

    def __init__(self, *, fpcp: str, fpid: str):
        super().__init__(_FILTER_NAME)
        super().__setattr__("filter_parameters", {})
        self.filter_parameters[_FPCP] = fpcp
        self.filter_parameters[_FPID] = fpid
        self.filter_parameters[_ENABLE_STAG] = 0
        self.filter_parameters[_SPCP] = _ANY
        self.filter_parameters[_SPID] = _ANY
        self._param_tag2str = {
            _FPCP: "src",
            _FPID: "dst",
            _ENABLE_STAG: "enable_second_tag",
            _SPCP: "spcp",
            _SPID: "spid"
        }
        self._update_filter()

    def enable_second_tag(self, spcp: str = _ANY, spid: str = _ANY):
        self.filter_parameters[_ENABLE_STAG] = 1
        self.filter_parameters[_SPCP] = spcp
        self.filter_parameters[_SPID] = spid
        self._update_filter()

    def _update_filter(self):
        self.clear_children()
        children_node_Map = {
            _FPCP: "",
            _FPID: "",
            _ENABLE_STAG: self.filter_parameters[_ENABLE_STAG],
        }
        properties = {}

        if self.filter_parameters[_FPCP] == _ANY:
            properties[_FPCP] = {_ANY: 1}

        children_node_Map[_FPCP] = self.filter_parameters[_FPCP]

        if self.filter_parameters[_FPID] == _ANY:
            properties[_FPID] = {_ANY: 1}

        children_node_Map[_FPID] = self.filter_parameters[_FPID]

        if self.filter_parameters[_ENABLE_STAG] == 1:
            if self.filter_parameters[_SPCP] == _ANY:
                children_node_Map[_SPCP] = _ANY
                properties[_SPCP] = {_ANY: 1}
            else:
                children_node_Map[_SPCP] = self.filter_parameters[_SPCP]

            if self.filter_parameters[_SPID] == _ANY:
                children_node_Map[_SPID] = _ANY
                properties[_SPID] = {_ANY: 1}
            else:
                children_node_Map[_SPID] = self.filter_parameters[_SPID]

        self.add_children(children_node_Map)
        self.set_properties(properties)

    @staticmethod
    def construct_from_node(node: ET.Element):
        if node.find(_FPCP).get(_ANY) == "1":
            fpcp = _ANY
        else:
            fpcp = node.findtext(_FPCP)

        if node.find(_FPID).get(_ANY) == "1":
            fpid = _ANY
        else:
            fpid = node.findtext(_FPID)

        enable_stag = node.findtext(_ENABLE_STAG)
        vlan = VLANFilter(
            fpcp=fpcp, fpid=fpid
        )
        if enable_stag == 1:
            if node.find(_SPCP).get(_ANY) == "1":
                spcp = _ANY
            else:
                spcp = node.findtext(_SPCP)

            if node.find(_SPID).get(_ANY) == "1":
                spid = _ANY
            else:
                spid = node.findtext(_SPID)
            vlan.enable_second_tag(spcp, spid)

        enable = True if node.get("enable") == "1" else False
        if not enable:
            vlan.disable_filter()

        return vlan


if __name__ == '__main__':
    vlan = VLANFilter(fpcp="any", fpid="any")
    print(vlan.xml)

    vlan.enable_second_tag(spcp="any", spid="any")
    print(vlan.xml)

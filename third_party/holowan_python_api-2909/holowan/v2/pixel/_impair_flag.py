"""
HoloWAN Network Emulator
Python API
"""

from holowan.v2.pixel._filter_base import Filter
import xml.etree.cElementTree as ET
from typing import List

_FILTER_NAME: str = r"impair_flag"


class Damage(object):
    DAMAGE_BEFORE: str = r"before"
    DAMAGE_AFTER: str = r"after"

    DAMAGE_ALL_TYPES: str = r"all"

    DAMAGE_CLS_DROP: str = r"cls_drop"
    DAMAGE_NO_IMPAIRMENT: str = r"no_impair"
    DAMAGE_MODIFICATION: str = r"mod"
    DAMAGE_FRAGMENT: str = r"frg_af"
    DAMAGE_DUPLICATION: str = r"dup"
    DAMAGE_BANDWIDTH_DROP: str = r"bw_drop"
    DAMAGE_LOSS: str = r"los"
    DAMAGE_BIT_ERROR: str = r"ber"
    DAMAGE_REORDERING: str = r"reo"
    DAMAGE_BYPASS: str = r"bypass"

    _DAMAGE_LIST = [
        DAMAGE_CLS_DROP,
        DAMAGE_NO_IMPAIRMENT,
        DAMAGE_MODIFICATION,
        DAMAGE_FRAGMENT,
        DAMAGE_DUPLICATION,
        DAMAGE_BANDWIDTH_DROP,
        DAMAGE_LOSS,
        DAMAGE_BIT_ERROR,
        DAMAGE_REORDERING,
        DAMAGE_BYPASS
    ]


_param_tag2str = {
    Damage.DAMAGE_BEFORE: r"before",
    Damage.DAMAGE_AFTER: r"after",
    Damage.DAMAGE_CLS_DROP: r"cls_drop",
    Damage.DAMAGE_ALL_TYPES: r"all_types",
    Damage.DAMAGE_NO_IMPAIRMENT: r"no_impairment",
    Damage.DAMAGE_MODIFICATION: r"modification",
    Damage.DAMAGE_FRAGMENT: r"fragment",
    Damage.DAMAGE_DUPLICATION: r"duplication",
    Damage.DAMAGE_BANDWIDTH_DROP: r"bandwidth_drop",
    Damage.DAMAGE_LOSS: r"loss",
    Damage.DAMAGE_BIT_ERROR: r"bit_error",
    Damage.DAMAGE_REORDERING: r"reordering",
    Damage.DAMAGE_BYPASS: r"bypass",
}


class TypeFilter(Filter):
    r"""
        强制关键字传参
    """

    def __init__(self, *, capture_moment: int = 1) -> None:
        super().__init__(_FILTER_NAME)
        super().__setattr__("filter_parameters", {})
        if capture_moment == 0:
            self.filter_parameters[Damage.DAMAGE_BEFORE] = 1
            self.filter_parameters[Damage.DAMAGE_AFTER] = 0
        else:
            self.filter_parameters[Damage.DAMAGE_BEFORE] = 0
            self.filter_parameters[Damage.DAMAGE_AFTER] = 1

        self.filter_parameters[Damage.DAMAGE_CLS_DROP] = 0
        self.filter_parameters[Damage.DAMAGE_ALL_TYPES] = 1
        self.filter_parameters[Damage.DAMAGE_NO_IMPAIRMENT] = 0
        self.filter_parameters[Damage.DAMAGE_MODIFICATION] = 0
        self.filter_parameters[Damage.DAMAGE_FRAGMENT] = 0
        self.filter_parameters[Damage.DAMAGE_DUPLICATION] = 0
        self.filter_parameters[Damage.DAMAGE_BANDWIDTH_DROP] = 0
        self.filter_parameters[Damage.DAMAGE_LOSS] = 0
        self.filter_parameters[Damage.DAMAGE_BIT_ERROR] = 0
        self.filter_parameters[Damage.DAMAGE_REORDERING] = 0
        self.filter_parameters[Damage.DAMAGE_BYPASS] = 0
        self._param_tag2str = _param_tag2str
        self._update_filter()

    def set_capture_moment(self,damage:str):
        self.filter_parameters[damage] = 1
        if damage == Damage.DAMAGE_BEFORE:
            self.filter_parameters[Damage.DAMAGE_AFTER] = 0
        elif damage == Damage.DAMAGE_AFTER:
            self.filter_parameters[Damage.DAMAGE_BEFORE] = 0
        else:
            raise ValueError("The capture moment must be in [ Damage.DAMAGE_BEFORE, Damage.DAMAGE_AFTER ].")

        self.set_node_text(damage, self.filter_parameters[damage])
        self._update_filter()

    def add_damage(self, damage:str) -> None:
        if damage == Damage.DAMAGE_BEFORE or damage == Damage.DAMAGE_AFTER:
            self.set_capture_moment(damage)
        else:
            self.filter_parameters[damage] = 1
            if damage != Damage.DAMAGE_ALL_TYPES:
                self.filter_parameters[Damage.DAMAGE_ALL_TYPES] = 0

        self.set_node_text(damage, self.filter_parameters[damage])
        self._update_filter()

    def remove_damage(self, damage:str) -> None:
        if damage == Damage.DAMAGE_BEFORE or damage == Damage.DAMAGE_AFTER:
            self.set_capture_moment(damage)
        else:
            self.filter_parameters[damage] = 0

        self.set_node_text(damage, self.filter_parameters[damage])
        self._update_filter()

    def __str__(self):
        res = self.__class__.__name__ + ":{ "
        for k, v in self.filter_parameters.items():
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)

        res += "}"
        return res

    @staticmethod
    def construct_from_node(node: ET.Element):
        enable = True if node.get("enable") == "1" else False
        after = int(node.findtext(Damage.DAMAGE_AFTER))
        _t = TypeFilter(capture_moment=after)
        if not enable:
            _t.disable_filter()

        for n in list(node):
            if n.text == "1":
                _t.add_damage(n.tag)

        return _t


if __name__ == '__main__':
    t = TypeFilter(capture_moment=1)
    print(t)
    print(t.xml)

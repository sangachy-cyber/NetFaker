"""
HoloWAN Network Emulator
Python API
"""

import xml.etree.cElementTree as ET
from typing import overload

from holowan.v2._xml_holder import XMLHolder

_CHANGE_MODE_TAG: str = r"shake"
_CHANGE_MODE_TYPE: str = r"type_id"

# common
_CHANGE_MODE_MAX: str = r"max"
_CHANGE_MODE_MIN: str = r"min"
_CHANGE_MODE_PERIOD: str = r"cycle"
# 1
_CHANGE_MODE_PHASE: str = r"phase"
# 2
_CHANGE_MODE_RISE: str = r"proportion_up"
_CHANGE_MODE_FALL: str = r"proportion_down"
# 5,6
_CHANGE_MODE_SECTION: str = r"proportion"


class ChangeMode(XMLHolder):
    """变化模式

    Notes:
        强制关键字传参。共有 6 种变化模式，通过不同的参数传入方式构造。

    """

    @overload
    def __init__(self, *, mode: int = 0):
        ...

    @overload
    def __init__(self, *, mode: int = 1, max: float, min: float, phase: int, period: int):
        ...

    @overload
    def __init__(self, *, mode: int = 2, max: float, min: float, rise: float, fall: float, period: int):
        ...

    @overload
    def __init__(self, *, mode: int = 3, max: float, min: float, period: int):
        ...

    @overload
    def __init__(self, *, mode: int = 4, max: float, min: float, period: int):
        ...

    @overload
    def __init__(self, *, mode: int = 5, max: float, min: float, section: float, period: int):
        ...

    @overload
    def __init__(self, *, mode: int = 6, max: float, min: float, section: float, period: int):
        ...

    def __init__(self, **kwargs):
        super().__init__(_CHANGE_MODE_TAG)
        self._mode = kwargs["mode"]
        self.parameters = {}
        self._param_tag2str = {
            _CHANGE_MODE_PHASE: "phase",
            _CHANGE_MODE_RISE: "rise",
            _CHANGE_MODE_FALL: "fall",
            _CHANGE_MODE_SECTION: "section",
            _CHANGE_MODE_MAX: "max",
            _CHANGE_MODE_MIN: "min",
            _CHANGE_MODE_PERIOD: "period"
        }
        if self._mode == 0:
            pass
        else:
            if self._mode == 1 and len(kwargs) == 5:
                if kwargs["phase"] == None:
                    raise ValueError("Missing argument 'phase' in mode 1.")
                else:
                    self.parameters[_CHANGE_MODE_PHASE] = kwargs["phase"] / 100
            elif self._mode == 2 and len(kwargs) == 6:
                if kwargs["rise"] == None:
                    raise ValueError("Missing argument 'rise' in mode 2.")
                else:
                    self.parameters[_CHANGE_MODE_RISE] = kwargs["rise"]

                if kwargs["fall"] == None:
                    raise ValueError("Missing argument 'fall' in mode 2.")
                else:
                    self.parameters[_CHANGE_MODE_FALL] = kwargs["fall"]
            elif self._mode == 3 and len(kwargs) == 4:
                pass
            elif self._mode == 4 and len(kwargs) == 4:
                pass
            elif self._mode == 5 and len(kwargs) == 5:
                self.parameters[_CHANGE_MODE_SECTION] = kwargs["section"] / 100
            elif self._mode == 6 and len(kwargs) == 5:
                self.parameters[_CHANGE_MODE_SECTION] = kwargs["section"] / 100
            else:
                raise ValueError("The arguments do not match its mode.")

            self.parameters[_CHANGE_MODE_MAX] = kwargs["max"]
            self.parameters[_CHANGE_MODE_MIN] = kwargs["min"]
            self.parameters[_CHANGE_MODE_PERIOD] = kwargs["period"]

        self._update()

    def _update(self):
        self.clear_children()
        self.add_children(self.parameters)
        self.set(_CHANGE_MODE_TYPE, str(self._mode))

    @property
    def mode(self) -> int:
        return self._mode

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'ChangeMode':
        """由一个变化模式的 xml 节点构造 ChangeMode 对象

        Args:
            node: 变化模式的 xml 节点

        Returns:
            ChangeMode 对象
        """
        mode = int(node.get(_CHANGE_MODE_TYPE))
        if mode == 0:
            return ChangeMode(mode=0)
        elif mode == 1:
            return ChangeMode(
                mode=mode, max=float(node.findtext(_CHANGE_MODE_MAX)),
                min=float(node.findtext(_CHANGE_MODE_MIN)),
                phase=int(float(node.findtext(_CHANGE_MODE_PHASE)) * 100),
                period=int(node.findtext(_CHANGE_MODE_PERIOD))
            )
        elif mode == 2:
            return ChangeMode(
                mode=mode, max=float(node.findtext(_CHANGE_MODE_MAX)),
                min=float(node.findtext(_CHANGE_MODE_MIN)), rise=float(node.findtext(_CHANGE_MODE_RISE)),
                fall=float(node.findtext(_CHANGE_MODE_FALL)), period=int(node.findtext(_CHANGE_MODE_PERIOD))
            )
        elif mode == 3:
            return ChangeMode(
                mode=mode, max=float(node.findtext(_CHANGE_MODE_MAX)),
                min=float(node.findtext(_CHANGE_MODE_MIN)), period=int(node.findtext(_CHANGE_MODE_PERIOD))
            )
        elif mode == 4:
            return ChangeMode(
                mode=mode, max=float(node.findtext(_CHANGE_MODE_MAX)),
                min=float(node.findtext(_CHANGE_MODE_MIN)), period=int(node.findtext(_CHANGE_MODE_PERIOD))
            )
        elif mode == 5:
            return ChangeMode(
                mode=mode, max=float(node.findtext(_CHANGE_MODE_MAX)),
                min=float(node.findtext(_CHANGE_MODE_MIN)),
                section=int(float(node.findtext(_CHANGE_MODE_SECTION)) * 100),
                period=int(node.findtext(_CHANGE_MODE_PERIOD))
            )
        elif mode == 6:
            return ChangeMode(
                mode=mode, max=float(node.findtext(_CHANGE_MODE_MAX)),
                min=float(node.findtext(_CHANGE_MODE_MIN)),
                section=int(float(node.findtext(_CHANGE_MODE_SECTION)) * 100),
                period=int(node.findtext(_CHANGE_MODE_PERIOD))
            )
        else:
            raise ValueError("Unknown ChangeMode type.")

    def __str__(self):
        res = self.__class__.__name__ + ":{ "
        for k, v in self.parameters.items():
            if k == _CHANGE_MODE_PHASE:
                v *= 100
            if k == _CHANGE_MODE_SECTION:
                v *= 100
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)

        res += "}"
        return res

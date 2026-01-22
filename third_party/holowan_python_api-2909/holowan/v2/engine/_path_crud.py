"""
HoloWAN Network Emulator
Python API

Path CRUD tool
"""

from holowan.v2 import EngineID
from holowan.v2 import PathID
from holowan.v2._xml_holder import XMLHolder
from holowan.v2.utils import xml_util as xt
from holowan.v2 import check_parameter
import os

_CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
_PATH_CRUD_XML_FORMAT_FILE: str = os.path.join(_CURRENT_DIR, r"../../resources/path_crud.xml")
_PATH_CRUD_TAG: str = r"holowan_modify_path_config_info"

_MODIFY_SWITCH: str = r"modify_switch"
_ENGINE_ID: str = r"engine_id"
_PATH_ID: str = r"path_id"
_PATH_NAME: str = r"path_name"
_IF_ENABLE: str = r"if_enable"


class PathCRUD(XMLHolder):
    """Path 增删改模板
    此类设计为 path 的增删改模板，作为 Engine 类的工具组件，对此类的属性的设置不会改变 HoloWAN 的配置。
    
    Args:
        modify_switch: 操作类型。
        engine_id: 引擎 id。
        path_id: Path id。
        path_name: Path 名称。
        if_enable: 是否启用。
    """

    def __init__(self):
        root = xt.xmlFile_to_Object(_PATH_CRUD_XML_FORMAT_FILE).getroot()
        super().__init__(_PATH_CRUD_TAG, list(root))
        self._params = {}
        self._params[_MODIFY_SWITCH] = int(self.findtext(_MODIFY_SWITCH))
        self._params[_ENGINE_ID] = int(self.findtext(_ENGINE_ID))
        self._params[_PATH_ID] = int(self.findtext(_PATH_ID))
        self._params[_PATH_NAME] = self.findtext(_PATH_NAME)
        self._params[_IF_ENABLE] = int(self.findtext(_IF_ENABLE))

    # ===========================================================
    @property
    def modify_switch(self) -> int:
        return self._params[_MODIFY_SWITCH]

    @modify_switch.setter
    @check_parameter
    def modify_switch(self, value: int) -> None:
        self._params[_MODIFY_SWITCH] = value
        self.set_node_text(_MODIFY_SWITCH, self._params[_MODIFY_SWITCH])

    # ===========================================================

    @property
    def engine_id(self) -> EngineID:
        return self._params[_ENGINE_ID]

    @engine_id.setter
    @check_parameter
    def engine_id(self, value: EngineID) -> None:
        self._params[_ENGINE_ID] = value
        self.set_node_text(_ENGINE_ID, self._params[_ENGINE_ID])

    # ===========================================================
    @property
    def path_id(self) -> PathID:
        return self._params[_PATH_ID]

    @path_id.setter
    @check_parameter
    def path_id(self, value: PathID) -> None:
        self._params[_PATH_ID] = value
        self.set_node_text(_PATH_ID, self._params[_PATH_ID])

    # ===========================================================
    @property
    def path_name(self) -> str:
        return self._params[_PATH_NAME]

    @path_name.setter
    @check_parameter
    def path_name(self, value: str) -> None:
        self._params[_PATH_NAME] = value
        self.set_node_text(_PATH_NAME, self._params[_PATH_NAME])

    # ===========================================================
    @property
    def if_enable(self) -> int:
        return self._params[_IF_ENABLE]

    @if_enable.setter
    @check_parameter
    def if_enable(self, value: int) -> None:
        self._params[_IF_ENABLE] = value
        self.set_node_text(_IF_ENABLE, self._params[_IF_ENABLE])

    def __str__(self):
        return self._params.__str__()


if __name__ == '__main__':
    p = PathCRUD()
    print(p)
    print(p.xml)

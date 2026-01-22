"""
HoloWAN Network Emulator
Python API
"""

from holowan.v2._holowan_types import (
    PortNumber,
    HoloWANConfigXML,
    check_parameter
)
from holowan.v2._xml_holder import XMLHolder
from holowan.v2.utils import xml_util as xt

# ==================== Preferences Key Begin========================
CLEAN_BUFFER: str = r"clean_buffer"
ZERO_LINE: str = r"zero_line"
IGNORE_FRAME_OVERHEAD: str = r"ignore_frame_overhead"
ENABLE_JUMBO_FRAME: str = r"enable_jumbo_frame"
LANGUAGE: str = r"language"
CAPTURE_SIZE: str = r"capture_size"
PRINTOUT: str = r"printout"
STATISTICS_FREQUENCY: str = r"statistics_frequency"
HTTP_PORT: PortNumber = r"http_port"
GUI: str = r"gui"
# ==================== Preferences Key End =========================

PREFERENCE_XML_FORMAT = """
<?xml version="1.0" encoding="UTF-8"?>
<holowan_admin_preferences>
    <clean_buffer></clean_buffer>
    <zero_line></zero_line>
    <ignore_frame_overhead></ignore_frame_overhead>
    <enable_jumbo_frame></enable_jumbo_frame>
    <language></language>
    <capture_size></capture_size>
    <printout></printout>
    <statistics_frequency></statistics_frequency>
    <http_port></http_port>
    <gui></gui>
</holowan_admin_preferences>
"""


class Preference(XMLHolder):
    """HoloWAN 首选项信息。
    此类设计为解析、组织 HoloWAN 的首选项配置信息。
    并且，可以通过设置该类对象的属性，更改设备的首选项信息，如语言、GUI 版本。

        Note:
            更改属性并不会立即改变 HoloWAN 的配置，该类不具备与设备进行通讯的能力。

    Args:
        preference_xml (HoloWANConfigXML, optional): 首选项配置信息 xml 字符串。默认为 None。
    
    Examples:
        >>> preference = HoloWANAdmin.get_preferences(holowan_ip, holowan_port)
        >>> preference.language = 1
        >>> res = HoloWANAdmin.set_preference(holowan_ip, holowan_port, preference)

    """

    @check_parameter
    def __init__(self, preference_xml: HoloWANConfigXML = None) -> None:
        self._preference_parameters = {
            CLEAN_BUFFER: str,
            ZERO_LINE: str,
            IGNORE_FRAME_OVERHEAD: int,
            ENABLE_JUMBO_FRAME: int,
            LANGUAGE: int,
            CAPTURE_SIZE: int,
            PRINTOUT: int,
            STATISTICS_FREQUENCY: int,
            HTTP_PORT: int,
            GUI: int
        }
        if preference_xml != None:
            # TODO: 异常处理
            root = xt.xmlString_to_Object(preference_xml)
            self._preference_parameters[CLEAN_BUFFER] = root.findtext(CLEAN_BUFFER)
            self._preference_parameters[ZERO_LINE] = root.findtext(ZERO_LINE)
            self._preference_parameters[IGNORE_FRAME_OVERHEAD] = int(root.findtext(IGNORE_FRAME_OVERHEAD))
            self._preference_parameters[ENABLE_JUMBO_FRAME] = root.findtext(ENABLE_JUMBO_FRAME)
            self._preference_parameters[LANGUAGE] = int(root.findtext(LANGUAGE))
            self._preference_parameters[CAPTURE_SIZE] = int(root.findtext(CAPTURE_SIZE))
            self._preference_parameters[PRINTOUT] = root.findtext(PRINTOUT)
            self._preference_parameters[STATISTICS_FREQUENCY] = int(root.findtext(STATISTICS_FREQUENCY))
            self._preference_parameters[HTTP_PORT] = root.findtext(HTTP_PORT)
            self._preference_parameters[GUI] = int(root.findtext(GUI))
        super().__init__("holowan_admin_preferences", self._preference_parameters)

    # ==================================================================
    @property
    def clean_buffer_when_apply_path_config(self) -> bool:
        """下发Path配置时，是否清空buffer，True：是；False：否。

        Notes:
            可读可写

        """
        return True if self._preference_parameters[CLEAN_BUFFER] == "true" else False

    @clean_buffer_when_apply_path_config.setter
    @check_parameter
    def clean_buffer_when_apply_path_config(self, value: bool) -> None:
        self._preference_parameters[CLEAN_BUFFER] = "true" if value else "false"
        self.set_node_text(CLEAN_BUFFER, self._preference_parameters[CLEAN_BUFFER])

    # ==================================================================
    @property
    def packets_delay_start_from(self) -> str:
        """报文时延起点，取值：receive_time或bandwitdh_time。

        Notes:
            可读可写

        """
        return self._preference_parameters[ZERO_LINE]

    @packets_delay_start_from.setter
    @check_parameter
    def packets_delay_start_from(self, value) -> None:
        self._preference_parameters[ZERO_LINE] = value
        self.set_node_text(ZERO_LINE, self._preference_parameters[ZERO_LINE])

    # ==================================================================
    @property
    def bandwidth_eveluation(self) -> int:
        """带宽计算逻辑，0: 物理模式，1: 软件模式。

        Notes:
            可读可写

        """
        return self._preference_parameters[IGNORE_FRAME_OVERHEAD]

    @bandwidth_eveluation.setter
    @check_parameter
    def bandwidth_eveluation(self, value) -> None:
        self._preference_parameters[IGNORE_FRAME_OVERHEAD] = value
        self.set_node_text(IGNORE_FRAME_OVERHEAD, self._preference_parameters[IGNORE_FRAME_OVERHEAD])

    # ===================================================================
    @property
    def enable_jumbo_frame(self) -> bool:
        """巨型帧是否开启，0: 关闭，1: 开启。

        Notes:
            可读可写

        """
        return True if self._preference_parameters[ENABLE_JUMBO_FRAME] == 1 else False

    @enable_jumbo_frame.setter
    @check_parameter
    def enable_jumbo_frame(self, value: bool) -> None:
        self._preference_parameters[ENABLE_JUMBO_FRAME] = 1 if value else 0
        self.set_node_text(ENABLE_JUMBO_FRAME, self._preference_parameters[ENABLE_JUMBO_FRAME])

    # ==================================================================
    @property
    def language(self) -> int:
        """语言，0：English，1：中文。

        Notes:
            可读可写

        """
        return self._preference_parameters[LANGUAGE]

    @language.setter
    @check_parameter
    def language(self, value: int) -> None:
        self._preference_parameters[LANGUAGE] = value
        self.set_node_text(LANGUAGE, self._preference_parameters[LANGUAGE])

    # ==================================================================
    @property
    def pixel_capture_size(self) -> int:
        """pixel 抓包大小。

        Notes:
            可读可写

        """
        return self._preference_parameters[CAPTURE_SIZE]

    @pixel_capture_size.setter
    @check_parameter
    def pixel_capture_size(self, value: int) -> None:
        self._preference_parameters[CAPTURE_SIZE] = value
        self.set_node_text(CAPTURE_SIZE, self._preference_parameters[CAPTURE_SIZE])

    # ==================================================================
    @property
    def printout(self) -> bool:
        """是否启用输出。

        Notes:
            可读可写

        """
        return True if self._preference_parameters[PRINTOUT] == 1 else False

    @printout.setter
    @check_parameter
    def printout(self, value: bool) -> None:
        self._preference_parameters[PRINTOUT] = 1 if value else 0
        self.set_node_text(PRINTOUT, self._preference_parameters[PRINTOUT])

    # ==================================================================
    @property
    def statistics_frequency(self) -> int:
        """统计频率。

        Notes:
            可读可写

        """
        return self._preference_parameters[STATISTICS_FREQUENCY]

    @statistics_frequency.setter
    @check_parameter
    def statistics_frequency(self, value) -> None:
        self._preference_parameters[STATISTICS_FREQUENCY] = value
        self.set_node_text(STATISTICS_FREQUENCY, self._preference_parameters[STATISTICS_FREQUENCY])

    # ==================================================================
    @property
    def http_port(self) -> PortNumber:
        """HTTP 端口号。

        Notes:
            可读可写

        """
        return self._preference_parameters[HTTP_PORT]

    @http_port.setter
    @check_parameter
    def http_port(self, value: PortNumber) -> None:
        self._preference_parameters[HTTP_PORT] = value
        self.set_node_text(HTTP_PORT, self._preference_parameters[HTTP_PORT])

    # ==================================================================
    @property
    def gui_version(self) -> int:
        """GUI 版本

        Notes:
            可读可写

        """
        return self._preference_parameters[GUI]

    @gui_version.setter
    @check_parameter
    def gui_version(self, value: int) -> None:
        self._preference_parameters[GUI] = value
        self.set_node_text(GUI, self._preference_parameters[GUI])

    # ==================================================================

    # def _update_xml(self, target_node: str, value: Any) -> None:
    #     root = xt.xmlString_to_Object(self._preference_xml)
    #     node = xt.get_node(root, target_node)
    #     xt.set_node_text(node, value)
    #     self._preference_xml = xt.xmlObject_to_string(root)

    def __str__(self) -> str:
        res = self.__class__.__name__ + ":\n"
        for k, v in self._preference_parameters.items():
            res += "\t" + str(k).replace("\n", "") + ":" + str(v).replace("\n", "") + "\n"
        return res

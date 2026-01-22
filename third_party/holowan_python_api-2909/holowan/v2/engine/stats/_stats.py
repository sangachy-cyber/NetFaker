"""
HoloWAN Network Emulator
Python API
"""

from typing import Dict, List

import requests
import holowan
from holowan.v2._holowan_types import IPAddress, PortNumber, EngineID, PathID, check_parameter, HoloWANReturn
from holowan.v2.utils import xml_util as xt
import xml.etree.cElementTree as ET

from holowan.v2._commons import (
    get_current_paths_info, _stats_csv_url, _stats_current_url, _stats_clear_url, close_engine_emulation
)

_PATH_STATS_PATH_TAG = r"path"
_PATH_STATS_TIME: str = r"time"
_PATH_STATS_PATH_ID: str = r"path_id"
_PATH_STATS_L2R: str = r"left_to_right"
_PATH_STATS_R2L: str = r"right_to_left"

_PATH_STATS_DICT = {

}
_RX_BYTES: str = r"rx_bytes"
_RX_FRAMES: str = r"rx_frames"
_RX_RATE: str = r"rx_rate"
_RX_RATE_FRAMES: str = r"rx_rate_frames"
_TX_BYTES: str = r"tx_bytes"
_TX_FRAMES: str = r"tx_frames"
_TX_RATE: str = r"tx_rate"
_TX_RATE_FRAMES: str = r"tx_rate_frames"
_LOSS_FRAMES: str = r"loss_frames"
_DROP_FRAMES: str = r"drop_frames"
_QUEUE_BYTES: str = r"queue_bytes"
_QUEUE_FRAMES: str = r"queue_frames"
_MODIFY_FRAMES: str = r"modify_frames"
_REORDERED_FRAMES: str = r"reordered_frames"
_DUPLICATED_FRAMES: str = r"duplicated_frames"
_CORRUPTION_FRAMES: str = r"corruption_frames"
_BACKGROUND_RX_BYTES: str = r"background_rx_bytes"
_BACKGROUND_RX_FRAMES: str = r"background_rx_frames"
_BACKGROUND_TX_BYTES: str = r"background_tx_bytes"
_BACKGROUND_TX_FRAMES: str = r"background_tx_frames"
_BACKGROUND_TX_RATE: str = r"background_tx_rate"
_BACKGROUND_DROP_FRAMES: str = r"background_drop_frames"
_CUR_BANDWIDTH: str = r"cur_bandwidth"


class PathStatistics():
    """每条路径的统计数据
    此类设计为解析、组织 Path 的统计数据。类的数据权限为 readonly

    Args:
        待解析的包含统计数据的 xml 字符串

    """
    _time: str
    _data_l2r: Dict
    _data_r2l: Dict
    _path_id: PathID

    def __init__(self, xml_str: str) -> None:
        self._data_l2r = {}
        self._data_r2l = {}
        self._time = ""
        self._path_id = 0
        root = xt.xmlString_to_Object(xml_str).find(_PATH_STATS_PATH_TAG)
        self._time = root.findtext(_PATH_STATS_TIME)
        self._path_id = int(root.findtext(_PATH_STATS_PATH_ID))

        self._extract_data(root)

    def _extract_data(self, root: ET.Element) -> None:
        l2r_root = root.find(_PATH_STATS_L2R)
        r2l_root = root.find(_PATH_STATS_R2L)
        for d in list(l2r_root):
            key = d.tag
            val = eval(d.text)
            self._data_l2r[key] = val

        for d in list(r2l_root):
            key = d.tag
            val = eval(d.text)
            self._data_r2l[key] = val

    @property
    def time(self) -> str:
        return self._time

    @property
    def l2r_data(self) -> Dict:
        """port1 -> port2 方向的统计数据

        Returns:
            包含数据的Dict
        """
        return self._data_l2r

    @property
    def r2l_data(self) -> Dict:
        """port2 -> port1 方向的统计数据

        Returns:
            包含数据的Dict
        """
        return self._data_r2l

    def __str__(self) -> str:
        str = "{0}: {1}".format(_PATH_STATS_TIME, self._time) + "\n"
        str += "{0}: {1}".format(_PATH_STATS_PATH_ID, self._path_id) + "\n"
        str += "left to right: \n"
        for k, v in self._data_l2r.items():
            str += "\t{0}: {1}".format(k, v) + "\n"

        str += "right to left: \n"
        for k, v in self._data_r2l.items():
            str += "\t{0}: {1}".format(k, v) + "\n"

        return str


class Statistics(object):
    """HoloWAN Statistics

    Args:
        holowan_ip (IPAddress): HoloWAN 的 ip 地址。
        holowan_port(PortNumber): HoloWAN 的控制端口的端口号。
        engine_id (EngineID): 引擎 id。

    """
    _path_stats: dict

    def __init__(self, holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID) -> None:
        self._holowan_ip = holowan_ip
        self._holowan_port = holowan_port
        self._engine_id = engine_id

        self._path_stats = {}
        self._extract_stats_data()

    def _extract_stats_data(self) -> None:
        path_ids = get_current_paths_info(self._holowan_ip, self._holowan_port, self._engine_id)
        path_cnt = len(path_ids)
        for id in range(1, path_cnt + 1):
            self._path_stats[id] = self.get_path_statistics_data(id)

    @check_parameter
    def get_path_statistics_data(self, path_id: PathID):
        url = _stats_current_url(self._holowan_ip, self._holowan_port, self._engine_id, path_id)
        response_str = requests.get(url).text
        return PathStatistics(response_str)

    def clear_data(self):
        url = _stats_clear_url(
            self._holowan_ip, self._holowan_port, self._engine_id
        )
        return HoloWANReturn(requests.get(url).text)

    def create_csv(self, path_id: PathID, file_path: str):
        close_engine_emulation(self._holowan_ip, self._holowan_port, self._engine_id)
        url1 = _stats_csv_url(
            self._holowan_ip, self._holowan_port, self._engine_id,
            path_id, 1
        )
        response = HoloWANReturn(requests.get(url1).text)
        if response:
            url2 = _stats_csv_url(
                self._holowan_ip, self._holowan_port, self._engine_id,
                path_id, 2
            )
            response2 = requests.get(url2)
            with open(file_path, "w") as f:
                f.write(response2.text)
        return response

    @property
    def all_paths_statistics_data(self) -> dict:
        self._extract_stats_data()
        return self._path_stats

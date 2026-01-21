"""
HoloWAN Network Emulator
Python API
"""

import ntpath

import requests

import holowan.v2.utils.my_util as mt
from holowan.v2 import EngineID, IPAddress, PathID, PortNumber
from holowan.v2._commons import (
    _playback_apply_url,
    _playback_delete_url,
    _playback_file_list_url,
    _playback_get_data_url,
    _playback_get_status,
    _playback_release_url,
    _playback_set_cursor_status_url,
    _playback_set_default_status_url,
    _playback_set_switch_url,
    _playback_upload_url,
)


class PlayBack(object):
    """HoloWAN playback

    Args:
        holowan_ip (IPAddress): HoloWAN 的 ip 地址。
        holowan_port(PortNumber): HoloWAN 的控制端口的端口号。

    """

    def __init__(self, holowan_ip: IPAddress, holowan_port: PortNumber) -> None:
        self._holowan_ip = holowan_ip
        self._holowan_port = holowan_port

    def upload_playback_file(self, file_path: str):
        file_type = mt.getFileType(file_path)
        if file_type == "txt":
            file_name = ntpath.basename(file_path)
            url = _playback_upload_url(
                self._holowan_ip, self._holowan_port, file_name
            )
            with open(file_path, "rb") as file:
                request_file = {str(len(open(file_path, "rb").read())): (file_name, file, "text/plain")}
                return requests.post(url, files=request_file).text
        else:
            return '{"errCode":"-400","errMsg":"error","errReason":"上传的回放文件必须为.txt格式"}'

    def delete_playback_file(self, file_name: str):
        url = _playback_delete_url(
            self._holowan_ip, self._holowan_port, file_name
        )
        return requests.get(url).text

    def get_playback_file_data(self, engine_id: EngineID, path_id: PathID, file_name: str, is_brief: bool):
        brief = "true" if (is_brief == True) else "false"
        url = _playback_get_data_url(
            self._holowan_ip, self._holowan_port, engine_id,
            path_id, file_name, brief
        )
        return requests.get(url).text

    def apply_playback_file(self, engine_id: EngineID, path_id: PathID, file_name: str):
        url = _playback_apply_url(
            self._holowan_ip, self._holowan_port,
            engine_id, path_id, file_name
        )
        return requests.get(url).text

    def release_playback_file(self, engine_id: EngineID, path_id: PathID):
        url = _playback_release_url(
            self._holowan_ip, self._holowan_port,
            engine_id, path_id
        )
        return requests.get(url).text

    def get_playback_file_list(self):
        url = _playback_file_list_url(self._holowan_ip, self._holowan_port)
        return requests.get(url, verify=False).text

    def set_playback_default_status(self, engine_id: EngineID, path_id: PathID, action: str):
        if action not in ["play", "pause"]:
            raise ValueError("Argument direction must be play or pause, got {got!r}, value {value!r}".format(
                got=type(action), value=action
            ))

        if action == "play":
            action_num = 1
        elif action == "pause":
            action_num = 2
        else:
            return '{"errCode":"-400","errMsg":"error","errReason":"action的值必须为play(播放)或pause(暂停)"}'

        url = _playback_set_default_status_url(
            self._holowan_ip, self._holowan_port,
            engine_id, path_id,
            action_num
        )
        return requests.get(url, verify=False).text

    def set_playback_cursor_status(self, engine_id: EngineID, path_id: PathID, action: int, cursor: int):
        if action not in ["play", "pause"]:
            raise ValueError("Argument direction must be play or pause, got {got!r}, value {value!r}".format(
                got=type(action), value=action
            ))

        if action == "play":
            action_num = 1
        elif action == "pause":
            action_num = 2
        else:
            return '{"errCode":"-400","errMsg":"error","errReason":"action的值必须为play(播放)或pause(暂停)"}'

        url = _playback_set_cursor_status_url(
            self._holowan_ip, self._holowan_port,
            engine_id, path_id,
            action_num, cursor
        )
        return requests.get(url, verify=False).text

    def set_playback_switch(self, engine_id: int, path_id: int, switch_list: list):
        if len(switch_list) != 6:
            raise ValueError("The length of the argument switch_list(List[int]) must be 6, got {got!r}".format(
                got=len(switch_list)
            ))

        for i in range(len(switch_list)):
            if switch_list[i] not in [0, 1]:
                raise ValueError(
                    "The item in the argument switch_list must be 0 or 1, got {got!r}, value {value!r}, at index {index!r}".format(
                        got=type(switch_list[i]), value=switch_list[i], index=i
                    ))

        url = _playback_set_switch_url(
            self._holowan_ip, self._holowan_port,
            engine_id, path_id, switch_list
        )
        return requests.get(url, verify=False).text

    def get_playback_status(self, engine_id: int, path_id: int):
        url = _playback_get_status(
            self._holowan_ip, self._holowan_port, engine_id, path_id
        )
        return requests.get(url, verify=False).text


if __name__ == '__main__':
    from holowan.v2.playback import PlayBack

    holowan_ip = "192.168.1.111"
    holowan_port = "8080"

    playback = PlayBack(holowan_ip=holowan_ip, holowan_port=holowan_port)

    result = playback.set_playback_status(engine_id=3, path_id=1, action=1, cursor=12)

    # 打印 txt 文件列表
    print(result)

"""
HoloWAN Network Emulator
Python API
"""

import ntpath

import requests

import holowan.v2.utils.my_util as mt
from holowan.v2 import IPAddress, PortNumber, EngineID
from holowan.v2 import PathID

from holowan.v2._commons import (
    _playback_apply_url, _playback_delete_url, _playback_upload_url,
    _playback_release_url, _playback_get_data_url, _playback_file_list_url,
    _playback_set_default_status_url, _playback_set_cursor_status_url, _playback_set_switch_url, _playback_get_status
)

from holowan.v2 import HoloWANReturn


class PlayBack(object):
    """HoloWAN playback

    Args:
        holowan_ip (IPAddress): HoloWAN 的 ip 地址。
        holowan_port(PortNumber): HoloWAN 的控制端口的端口号。
        username (str): 登录用户名，默认为 "admin"。
        password (str): 登录密码，默认为 "holowan"。

    Notes:
        构造函数会自动执行登录操作，获取 Cookie session_id。
        所有后续的 API 调用都会自动携带该 Cookie。

    """

    def __init__(self, holowan_ip: IPAddress, holowan_port: PortNumber,
                 username: str = "admin", password: str = "holowan") -> None:
        self._holowan_ip = holowan_ip
        self._holowan_port = holowan_port
        
        # Session 对象用于保存 Cookie
        self.session = requests.Session()
        
        # 在调用任何 API 之前先执行登录
        self._login(username, password)

    def _login(self, username: str = "admin", password: str = "holowan") -> HoloWANReturn:
        """登录接口（私有方法），获取 Cookie session_id。
        
        Args:
            username: 用户名，默认为 "admin"
            password: 密码，默认为 "holowan"
            
        Returns:
            HoloWANReturn: 登录响应结果，成功后会保存 Cookie session_id 到 Session 中
        """
        try:
            from holowan.v2._commons import _api_protocol
            requestURL = _api_protocol(self._holowan_ip) + "{0}:{1}/login".format(self._holowan_ip, self._holowan_port)
            json_data = {"username": username, "password": password}
            response = self.session.post(requestURL, json=json_data, verify=False)
            # Session 会自动保存响应中的 Cookie
            return HoloWANReturn(response.text)
        except requests.exceptions.ConnectionError:
            return HoloWANReturn(r'{"errCode":"-200","errMsg":"ConnectionError","errReason":"Failed to establish a new connection"}')
        except Exception as e:
            return HoloWANReturn(r'{"errCode":"-500","errMsg":"ERROR","errReason":"' + str(e) + '"}')

    def login(self, username: str = "admin", password: str = "holowan") -> HoloWANReturn:
        """登录接口（公共方法），用于重新登录或更新 Cookie。
        
        Args:
            username: 用户名，默认为 "admin"
            password: 密码，默认为 "holowan"
            
        Returns:
            HoloWANReturn: 登录响应结果，成功后会保存 Cookie session_id 到 Session 中
        """
        return self._login(username, password)

    def upload_playback_file(self, file_path: str):
        file_type = mt.getFileType(file_path)
        if file_type == "txt":
            file_name = ntpath.basename(file_path)
            url = _playback_upload_url(
                self._holowan_ip, self._holowan_port, file_name
            )
            with open(file_path, "rb") as file:
                request_file = {str(len(open(file_path, "rb").read())): (file_name, file, "text/plain")}
                return self.session.post(url, files=request_file, verify=False).text
        else:
            return '{"errCode":"-400","errMsg":"error","errReason":"上传的回放文件必须为.txt格式"}'

    def delete_playback_file(self, file_name: str):
        url = _playback_delete_url(
            self._holowan_ip, self._holowan_port, file_name
        )
        return self.session.get(url, verify=False).text

    def get_playback_file_data(self, engine_id: EngineID, path_id: PathID, file_name: str, is_brief: bool):
        brief = "true" if (is_brief == True) else "false"
        url = _playback_get_data_url(
            self._holowan_ip, self._holowan_port, engine_id,
            path_id, file_name, brief
        )
        return self.session.get(url, verify=False).text

    def apply_playback_file(self, engine_id: EngineID, path_id: PathID, file_name: str):
        url = _playback_apply_url(
            self._holowan_ip, self._holowan_port,
            engine_id, path_id, file_name
        )
        return self.session.get(url, verify=False).text

    def release_playback_file(self, engine_id: EngineID, path_id: PathID):
        url = _playback_release_url(
            self._holowan_ip, self._holowan_port,
            engine_id, path_id
        )
        return self.session.get(url, verify=False).text

    def get_playback_file_list(self):
        url = _playback_file_list_url(self._holowan_ip, self._holowan_port)
        return self.session.get(url, verify=False).text

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
        return self.session.get(url, verify=False).text

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
        return self.session.get(url, verify=False).text

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
        return self.session.get(url, verify=False).text

    def get_playback_status(self, engine_id: int, path_id: int):
        url = _playback_get_status(
            self._holowan_ip, self._holowan_port, engine_id, path_id
        )
        return self.session.get(url, verify=False).text


if __name__ == '__main__':
    from holowan.v2.playback import PlayBack

    holowan_ip = "192.168.1.111"
    holowan_port = "8080"

    playback = PlayBack(holowan_ip=holowan_ip, holowan_port=holowan_port)

    result = playback.set_playback_status(engine_id=3, path_id=1, action=1, cursor=12)

    # 打印 txt 文件列表
    print(result)

"""
HoloWAN Network Emulator
Python API
"""

import requests, time

from holowan.v2._holowan_types import IPAddress, PortNumber, EngineID, HoloWANReturn, PathID
from holowan.v2.utils import xml_util as xt


class HoloWANAPI(object):
    STATISTICS_INFORMATION_API: str = r"statistics_information"
    EMULATOR_CONFIG_API: str = r"emulator_config"


def _api_protocol(holowan_ip: str) -> str:
    if holowan_ip.find(r"https://") == -1:
        return r"http://"
    else:
        return r""


def _set_engine_name_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: int, engine_name: str) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/set_engine_name?engine_id={2}&engine_name={3}&_t={4}".format(
        holowan_ip, holowan_port, engine_id, engine_name, int(time.time() * 1000)
    )


def _hold_engine_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID, old_password: int,
                     new_password: int) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}//hold_engine?engine={2}&passwd={3}&new_passwd={4}&type=short".format(
        holowan_ip, holowan_port, engine_id, old_password, new_password
    )


def _reset_engine_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/reset_engine?eid={2}".format(
        holowan_ip, holowan_port, engine_id
    )


def _static_info_url(holowan_ip: IPAddress, holowan_port: PortNumber) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/{2}".format(
        holowan_ip, holowan_port, HoloWANAPI.STATISTICS_INFORMATION_API
    )


def _emulator_cfg_url(holowan_ip: IPAddress, holowan_port: PortNumber) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/{2}".format(
        holowan_ip, holowan_port, HoloWANAPI.EMULATOR_CONFIG_API
    )


def _clf_cfg_info_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID):
    return _api_protocol(holowan_ip) + r"{0}:{1}/classifier_config_info_{2}.xml".format(holowan_ip, holowan_port,
                                                                                        engine_id)


def _reset_path_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID, path_id: PathID) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/reset_path?eid={2}&pid={3}".format(
        holowan_ip, holowan_port, engine_id, path_id
    )


def _add_path_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/add_path?eid={2}".format(holowan_ip, holowan_port, engine_id)


def _path_cfg_info_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID, path_id: PathID) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/path_config_info_{2}_{3}.xml".format(
        holowan_ip, holowan_port, engine_id, path_id
    )


def _csv_data_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID, path_id: PathID) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/csv_data?type=2&engine={2}&path={3}".format(
        holowan_ip, holowan_port, engine_id, path_id
    )


def _start_engine_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/start_running?engine={2}".format(
        holowan_ip, holowan_port, engine_id
    )


def _stop_engine_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/stop_running?engine={2}".format(
        holowan_ip, holowan_port, engine_id
    )


def _playback_upload_url(holowan_ip: IPAddress, holowan_port: PortNumber, file_name: str) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/playback_file_upload?filename={2}".format(
        holowan_ip, holowan_port, file_name
    )


def _playback_delete_url(holowan_ip: IPAddress, holowan_port: PortNumber, file_name: str) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/playback_file_delete?filename={2}".format(
        holowan_ip, holowan_port, file_name
    )


def _playback_get_data_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID, path_id: PathID,
                           file_name: str, brief: str) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/get_playback_data?eid={2}&pid={3}&filename={4}&brief={5}".format(
        holowan_ip, holowan_port, engine_id,
        path_id, file_name, brief
    )


def _playback_apply_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID, path_id: PathID,
                        file_name: str) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/playback_apply?filename={2}&eid={3}&pid={4}".format(
        holowan_ip, holowan_port, file_name, engine_id, path_id
    )


def _playback_release_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID, path_id: PathID) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/playback_release?eid={2}&pid={3}".format(
        holowan_ip, holowan_port, engine_id, path_id
    )


def _playback_file_list_url(holowan_ip: IPAddress, holowan_port: PortNumber) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/playback_file_list".format(
        holowan_ip, holowan_port
    )


def _playback_get_status(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: int, path_id: int):
    return _api_protocol(holowan_ip) + r"{0}:{1}/get_playback_status?eid={2}&pid={3}".format(
        holowan_ip, holowan_port, engine_id, path_id
    )


def _playback_set_default_status_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID, path_id: int,
                                     action: int) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/set_playback_status?eid={2}&pid={3}&action={4}".format(
        holowan_ip, holowan_port, engine_id, path_id, action
    )


def _playback_set_cursor_status_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID, path_id: int,
                                    action: int, cursor: int) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/set_playback_status?eid={2}&pid={3}&action={4}&cursor={5}".format(
        holowan_ip, holowan_port, engine_id, path_id, action, cursor
    )


def _playback_set_switch_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID, path_id: int,
                             switch_list: list):
    return _api_protocol(
        holowan_ip) + r"{0}:{1}/set_playback_switch?eid={2}&pid={3}&bandwidth1={4}&bandwidth2={5}&delay1={6}&delay2={7}&loss1={8}&loss2={9}".format(
        holowan_ip, holowan_port, engine_id, path_id,
        switch_list[0], switch_list[1], switch_list[2], switch_list[3], switch_list[4], switch_list[5]
    )


def _capture_filter_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/capture_filter?cid={2}".format(
        holowan_ip, holowan_port, engine_id
    )


def _capture_settings_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/capture_filter_rule_{2}.xml".format(
        holowan_ip, holowan_port, engine_id
    )


def _capture_start_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/capture_config?type=run&cid={2}".format(
        holowan_ip, holowan_port, engine_id
    )


def _capture_stop_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/capture_config?type=stop&cid={2}".format(
        holowan_ip, holowan_port, engine_id
    )


def _capture_clean_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/capture_config?type=clean&cid={2}".format(
        holowan_ip, holowan_port, engine_id
    )


def _capture_pcap_export_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/capture_pcap?cid={2}&type=filtered".format(
        holowan_ip, holowan_port, engine_id
    )


def _capture_mode_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID, path_id: PathID,
                      mode: str) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/capture_config?type=change_mode&cid={2}&pid={3}&mode={4}".format(
        holowan_ip, holowan_port, engine_id, path_id, mode
    )


def _capture_load_file_url(holowan_ip: IPAddress, holowan_port: PortNumber, file_path: str) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/capture_file?type=load&file_name={2}".format(
        holowan_ip, holowan_port, file_path
    )


def _capture_release_file_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID,
                              file_name: str) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/capture_file?type=release&cid={2}&file_name={3}".format(
        holowan_ip, holowan_port, engine_id, file_name
    )


def _capture_delete_file_url(holowan_ip: IPAddress, holowan_port: PortNumber, file_name: str) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/capture_file?type=delete&file_name={2}".format(
        holowan_ip, holowan_port, file_name
    )


def _capture_get_all_file_url(holowan_ip: IPAddress, holowan_port: PortNumber) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/capture_file?type=list".format(
        holowan_ip, holowan_port
    )

def _capture_get_pixel_status(holowan_ip:IPAddress,holowan_port:PortNumber)->str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/capture_status".format(
        holowan_ip,holowan_port
    )

def _capture_get_pixel_data(holowan_ip:IPAddress, holowan_port: PortNumber,engine_id:EngineID,offset:int,count:int)->str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/capture_data?cid={2}&type=brief&offset={3}&count={4}".format(
        holowan_ip,holowan_port,engine_id,offset,count
    )


def _preference_get_url(holowan_ip: IPAddress, holowan_port: PortNumber) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/get_preferences".format(
        holowan_ip, holowan_port
    )


def _preference_set_url(holowan_ip: IPAddress, holowan_port: PortNumber) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/set_preferences".format(
        holowan_ip, holowan_port
    )


def _work_port_info_url(holowan_ip: IPAddress, holowan_port: PortNumber) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/worker_port_info".format(
        holowan_ip, holowan_port
    )


def _network_info_url(holowan_ip: IPAddress, holowan_port: PortNumber) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/network_info".format(
        holowan_ip, holowan_port
    )


def network_set_url(holowan_ip: IPAddress, holowan_port: PortNumber) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/setting_network".format(
        holowan_ip, holowan_port
    )


def _sync_system_time_url(holowan_ip: IPAddress, holowan_port: PortNumber, time: int) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/sync_system_time?time={2}".format(
        holowan_ip, holowan_port, time
    )


def _get_system_time_url(holowan_ip: IPAddress, holowan_port: PortNumber) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/get_system_time".format(
        holowan_ip, holowan_port
    )


def _reboot_url(holowan_ip: IPAddress, holowan_port: PortNumber) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/reboot".format(
        holowan_ip, holowan_port
    )


def _stats_current_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID, path_id: PathID) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/current_resault_data?engine={2}&path={3}".format(
        holowan_ip, holowan_port, engine_id, path_id
    )


def _stats_clear_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/clean_engine_resault_data?engine={2}".format(
        holowan_ip, holowan_port, engine_id
    )


def _stats_csv_url(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID, path_id: PathID,
                   type: int) -> str:
    return _api_protocol(holowan_ip) + r"{0}:{1}/csv_data?type={2}&engine={3}&path={4}".format(
        holowan_ip, holowan_port, type, engine_id, path_id
    )


def _route_select_config_url(holowan_ip: IPAddress, holowan_port: PortNumber):
    return _api_protocol(holowan_ip) + r"{0}:{1}/route_select_config".format(
        holowan_ip, holowan_port
    )


def get_holowan_info(holowan_ip: IPAddress, holowan_port: PortNumber) -> str:
    url = _static_info_url(holowan_ip, holowan_port)
    return requests.get(url).text


def get_current_paths_info(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID):
    path_dic = {}
    info_str = get_holowan_info(holowan_ip, holowan_port)
    engine_nodes = xt.xmlString_to_Object(info_str)
    for node in engine_nodes.findall("e"):
        each_engine_id = xt.get_node(node, "ei").text
        if int(each_engine_id) == engine_id:
            path_nodes = xt.get_nodes(node, "./ep/p")
            for path_node in path_nodes:
                pathID = int(xt.get_node(path_node, "./pi").text)
                path_name = xt.get_node(path_node, "./pn").text
                is_enable = xt.get_node(path_node, "./ie").text
                path_dic[pathID] = {
                    "path_name": path_name,
                    "is_enable": is_enable
                }
    return path_dic


def get_available_engines(holowan_ip: IPAddress, holowan_port: PortNumber) -> list:
    response = get_holowan_info(holowan_ip=holowan_ip, holowan_port=holowan_port)
    engine_nodes = xt.xmlString_to_Object(response)
    idls = []
    for node in engine_nodes.findall("e"):
        each_engine_id = int(xt.get_node(node, "ei").text)
        idls.append(each_engine_id)
    return idls


def start_engine_emulation(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID):
    """开启引擎。

            Returns:
                HoloWANReturn: HoloWAN 返回值。
            """
    url = _start_engine_url(
        holowan_ip, holowan_port, engine_id
    )
    return HoloWANReturn(requests.get(url).text)


def close_engine_emulation(holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID):
    """关闭引擎。

            Returns:
                HoloWANReturn: HoloWAN 返回值。
            """
    url = _stop_engine_url(
        holowan_ip, holowan_port, engine_id
    )
    return HoloWANReturn(requests.get(url).text)

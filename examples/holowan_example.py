import json

from holowan.v2.engine import Engine
from holowan.v2.engine.classifier import RawByteRule
from holowan.v2.playback import PlayBack

# holowan 相关信息
holowan_ip = "160.100.15.195"
holowan_port = "8080"

engine_id = 1

# 创建 Engine 实例对象时，将从后端获取对应 engine 的信息与配置（状态1）
engine1 = Engine(holowan_ip, holowan_port, engine_id)
# 如果在创建了 engine1 后，在 Web GUI 上⼿动对引擎进⾏了配置（状态2）
# 那么 engine1 将不是最新的“状态2”，仍是 “状态1”
# 如果需要获取引擎最新的状态，请使⽤ Engine.update()⽅法
engine1.update()

classifier = engine1.packet_classifier

target_ip = "10.10.10.10"

last_octet = int(target_ip.split('.')[-1])
hex_str_value = f"0x{last_octet:02X}"

# print(classifier.port1.get_item_by_key(f"AutoGenFor_{target_ip}_1"))

#
# # 定义 Raw 1-Byte 字节偏移分类规则⽅法
# raw1 = RawByteRule(type=1, action=-1)
# # 设置 Raw 1-Byte 字节偏移分类规则具体参数
# raw1.add_raw_byte(layer=3, offset=59, mask="0xFF", value=hex_str_value)
# raw1.set_custom_name(f"AutoGenFor_{target_ip}_1")
#
#
# # 定义 Raw 1-Byte 字节偏移分类规则⽅法
# raw2 = RawByteRule(type=1, action=-1)
# # 设置 Raw 1-Byte 字节偏移分类规则具体参数
# raw2.add_raw_byte(layer=3, offset=63, mask="0xFF", value=hex_str_value)
# raw2.set_custom_name(f"AutoGenFor_{target_ip}_2")
#
# # 将 Raw 1-Byte 字节偏移分类规则应⽤到 port1 上
# result1 = engine1.apply_rule_to_classifier(raw1, port=1)
# if result1.err_code == 0:
#     print("Success")
# else:
#     print("Failed")
# result2 = engine1.apply_rule_to_classifier(raw2, port=2)
# if result2.err_code == 0:
#     print("Success")
# else:
#     print("Failed")
#



# 回放⽂件在电脑中的路径
playback_name = "sim_task_20260122_103424_udw81q.txt"

upload_file_path = "../output/holowan/sim_task_20260122_103424_udw81q.txt"
# 创建 playback
playback = PlayBack(holowan_ip=holowan_ip,holowan_port=holowan_port)


# 获取并检查回放文件是否存在
result = playback.get_playback_file_list()
result_json = json.loads( result)
print(result_json["data"]["list"])

playback_exist = False
for item in result_json["data"]["list"]:
    if item["name"] == playback_name:
        playback_exist = True
        break


if playback_exist:
    print("Playback file exists")
else:
    # 上传回放文件到 Holowan
    result = playback.upload_playback_file(file_path=upload_file_path)
    if result == '{"code": 0, "msg": "successfully"}':
        print("Success")
    else:
        print("Failed")

# 使用回放文件
result = playback.apply_playback_file(engine_id=engine_id,path_id=15,file_name=playback_name)
result_json = json.loads( result)
if result_json["code"] == 0:
    print("Success")
else:
    print(f"Failed， {result}")


# 应用完成后，释放回放文件
result = playback.release_playback_file(engine_id=engine_id,path_id=15)
if result_json["code"] == 0:
    print("Success")
else:
    print(f"Failed， {result}")


# 删除回放文件
result = playback.delete_playback_file(file_name=playback_name)
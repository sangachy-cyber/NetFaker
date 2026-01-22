# 导⼊相关 module
from holowan.v2.engine import Engine

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

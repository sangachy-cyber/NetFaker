import unittest

from settings import *


class ExampleBackgroundUtilization(unittest.TestCase):

    def test_bg_disable(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import BackgroundUtilizationDisable

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.background_utilization = BackgroundUtilizationDisable()

        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_bg_random(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import BackgroundUtilizationRandom

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.background_utilization = BackgroundUtilizationRandom(rate=1.23, burst=61)

        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_bg_pcap(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import BackgroundUtilizationPCAP

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.background_utilization = BackgroundUtilizationPCAP(rate=2.34, pcap_name="mlx.pcap")

        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result


if __name__ == '__main__':
    unittest.main()

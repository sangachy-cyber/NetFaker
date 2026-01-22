import unittest

from settings import *


class ExampleFrameOverhead(unittest.TestCase):

    def test_fo4(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import FrameOverhead4Ethernet

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.frame_overhead = FrameOverhead4Ethernet()

        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_fo24(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import FrameOverhead24Ethernet

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.frame_overhead = FrameOverhead24Ethernet()

        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_fo_custom(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import FrameOverheadCustom

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.frame_overhead = FrameOverheadCustom(size=32)

        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result


if __name__ == '__main__':
    unittest.main()

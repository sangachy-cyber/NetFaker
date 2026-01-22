import unittest

from settings import *


class ExampleReordering(unittest.TestCase):

    def test_reordering_cycle(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import ReorderingCycle

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.reordering = ReorderingCycle(type=1, period=999, count=9,
                                               min_delay=11.1, max_delay=99.9)

        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_reordering_normal(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import ReorderingNormal

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.reordering = ReorderingNormal(probability=5, min_delay=11.1, max_delay=99.9)

        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_reordering_jitter1(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import ChangeMode, ReorderingJitter

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.reordering = ReorderingJitter(min_delay=11.1, max_delay=99.9,
                                                change_mode=ChangeMode(mode=1, max=99.9, min=11.1, phase=60,
                                                                       period=100))

        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_reordering_jitter2(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import ChangeMode, ReorderingJitter

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.reordering = ReorderingJitter(min_delay=11.1, max_delay=99.9,
                                                change_mode=ChangeMode(mode=2, max=99.9, min=11.1, rise=0.1, fall=0.2,
                                                                       period=100))

        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_reordering_jitter3(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import ChangeMode, ReorderingJitter

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.reordering = ReorderingJitter(min_delay=11.1, max_delay=99.9,
                                                change_mode=ChangeMode(mode=3, max=99.9, min=11.1, period=100))

        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_reordering_jitter4(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import ChangeMode, ReorderingJitter

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.reordering = ReorderingJitter(min_delay=11.1, max_delay=99.9,
                                                change_mode=ChangeMode(mode=4, max=99.9, min=11.1, period=100))

        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_reordering_jitter5(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import ChangeMode, ReorderingJitter

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.reordering = ReorderingJitter(min_delay=11.1, max_delay=99.9,
                                                change_mode=ChangeMode(mode=5, max=99.9, min=11.1, section=20,
                                                                       period=100))

        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_reordering_jitter6(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import ChangeMode, ReorderingJitter

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.reordering = ReorderingJitter(min_delay=11.1, max_delay=99.9,
                                                change_mode=ChangeMode(mode=6, max=99.9, min=11.1, section=20,
                                                                       period=100))

        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result


if __name__ == '__main__':
    unittest.main()

import unittest

from settings import *


class ExampleBandwidth(unittest.TestCase):
    def setUp(self):
        from holowan.v2.engine import Engine
        self.holowan_ip = HOLOWAN_IP
        self.holowan_port = HOLOWAN_PORT
        self.engine_id = ENGINE_ID
        self.engine = Engine(self.holowan_ip, self.holowan_port, self.engine_id)
        self.path_id = PATH_ID
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.reset()

    def test_bandwidth_unit_conversion(self):
        """测试带宽单位转换"""
        from holowan.v2.engine.path import BandwidthFixed

        # 测试不同单位的带宽设置
        units = [1, 2, 3]  # 1: bps, 2: Kbps, 3: Mbps
        rates = [100, 100, 100]  # 相同带宽，不同单位表示

        for unit, rate in zip(units, rates):
            self.path = self.engine.get_path_by_id(self.path_id)
            self.path.l2r.bandwidth = BandwidthFixed(rate=rate, unit=unit)
            result = self.engine.apply_path_configuration(self.path)
            print(result)  # 打印result值
            self.assertTrue(result)

    def test_bandwidth_params_validation(self):
        """测试带宽参数验证"""
        from holowan.v2.engine.path import BandwidthFixed

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.l2r.bandwidth = BandwidthFixed(rate=100, unit=1)
        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

        # 测试边界值
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.l2r.bandwidth = BandwidthFixed(rate=0.001, unit=1)
        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

    def test_bandwidth_fixed(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import BandwidthFixed

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.bandwidth = BandwidthFixed(rate=999.9, unit=1)

        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_bandwidth_jitter1(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import BandwidthJitter, ChangeMode

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.bandwidth = BandwidthJitter(unit=1,
                                              change_mode=ChangeMode(mode=1, max=99.9, min=11.1, phase=1, period=100))

        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_bandwidth_jitter2(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import BandwidthJitter, ChangeMode

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.bandwidth = BandwidthJitter(unit=1,
                                              change_mode=ChangeMode(mode=2, max=99.9, min=11.1, rise=0.1, fall=0.2,
                                                                     period=100))

        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_bandwidth_jitter3(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import BandwidthJitter, ChangeMode

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.bandwidth = BandwidthJitter(unit=1, change_mode=ChangeMode(mode=3, max=99.9, min=11.1, period=100))

        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_bandwidth_jitter4(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import BandwidthJitter, ChangeMode

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.bandwidth = BandwidthJitter(unit=1, change_mode=ChangeMode(mode=4, max=99.9, min=11.1, period=100))

        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_bandwidth_jitter5(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import BandwidthJitter, ChangeMode

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.bandwidth = BandwidthJitter(unit=1, change_mode=ChangeMode(mode=5, max=99.9, min=11.1, section=20,
                                                                             period=100))

        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_bandwidth_jitter6(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import BandwidthJitter, ChangeMode

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.bandwidth = BandwidthJitter(unit=1, change_mode=ChangeMode(mode=6, max=99.9, min=11.1, section=20,
                                                                             period=100))

        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_bandwidth_tokenbucket1(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import BandwidthTokenBucket

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.bandwidth = BandwidthTokenBucket(type=1, rate=1.1, rate_unit=1, burst=1.1, burst_unit=1)

        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_bandwidth_tokenbucket2(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import BandwidthTokenBucket

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.bandwidth = BandwidthTokenBucket(type=2, cir=10, cir_unit=3, cbs=10, cbs_unit=3,
                                                   ebs=50, ebs_unit=3)

        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_bandwidth_tokenbucket3(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import BandwidthTokenBucket

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.bandwidth = BandwidthTokenBucket(type=3, cir=10, cir_unit=3, cbs=10, cbs_unit=3,
                                                   pir=10, pir_unit=3, pbs=50, pbs_unit=3)

        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_bandwidth_bidirectional(self):
        """测试双向带宽参数"""
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import BandwidthBidirectional

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.bandwidth = BandwidthBidirectional(rate=998.0, unit=1)
        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result


if __name__ == '__main__':
    unittest.main()

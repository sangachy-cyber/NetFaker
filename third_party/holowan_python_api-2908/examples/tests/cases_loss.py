import unittest

from settings import *


class ExampleLoss(unittest.TestCase):

    def setUp(self):
        from holowan.v2.engine import Engine
        self.holowan_ip = HOLOWAN_IP
        self.holowan_port = HOLOWAN_PORT
        self.engine_id = ENGINE_ID
        self.engine = Engine(self.holowan_ip, self.holowan_port, self.engine_id)
        self.path_id = PATH_ID
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.reset()

    def test_loss_rate_validation(self):
        """测试丢包率参数验证"""
        from holowan.v2.engine.path import LossRandom

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.l2r.loss = LossRandom(loss_rate=0.01)
        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

        # 测试边界值
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.l2r.loss = LossRandom(loss_rate=0.001)
        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

    def test_loss_burst_validation(self):
        """测试突发丢包参数验证"""
        from holowan.v2.engine.path import LossBurst

        # 测试不同突发长度
        min_bursts = [1, 5, 10]
        max_bursts = [10, 20, 50]

        for min_burst, max_burst in zip(min_bursts, max_bursts):
            self.path = self.engine.get_path_by_id(self.path_id)
            self.path.l2r.loss = LossBurst(probability=0.01, min=min_burst, max=max_burst)
            result = self.engine.apply_path_configuration(self.path)
            print(result)  # 打印result值
            self.assertTrue(result)

    def test_loss_cycle(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import LossCycle

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.loss = LossCycle(period=900, burst=10)
        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_loss_burst(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import LossBurst
        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.loss = LossBurst(probability=0.01, min=10, max=50)
        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)

    def test_loss_random(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import LossRandom

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.loss = LossRandom(loss_rate=0.001)
        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_loss_markov(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import LossMarkov

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.loss = LossMarkov(p13=0.008, p31=0.05, p32=0.00001, p23=1, p14=0.0007)
        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_loss_GilbertElliott(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import LossGilbertElliott

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.loss = LossGilbertElliott(good_state_loss=0.1, good_state_change=0.2, bad_state_loss=0.3,
                                            bad_state_change=0.4)
        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_loss_jitter1(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import ChangeMode, LossJitter

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.loss = LossJitter(change_mode=ChangeMode(mode=1, max=99.9, min=11.1, phase=1, period=100))
        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_loss_jitter2(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import ChangeMode, LossJitter

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.loss = LossJitter(change_mode=ChangeMode(mode=2, max=99.9, min=11.1, rise=0.1, fall=0.2, period=100))
        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_loss_jitter3(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import ChangeMode, LossJitter

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.loss = LossJitter(change_mode=ChangeMode(mode=3, max=99.9, min=11.1, period=100))
        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_loss_jitter4(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import ChangeMode, LossJitter

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.loss = LossJitter(change_mode=ChangeMode(mode=4, max=99.9, min=11.1, period=100))
        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_loss_jitter5(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import ChangeMode, LossJitter

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.loss = LossJitter(change_mode=ChangeMode(mode=5, max=99.9, min=11.1, section=20, period=100))
        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_loss_jitter6(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import ChangeMode, LossJitter

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.loss = LossJitter(change_mode=ChangeMode(mode=6, max=99.9, min=11.1, section=20, period=100))
        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_loss_count(self):
        """测试计数丢包参数"""
        from holowan.v2.engine import Engine
        from holowan.v2.engine.path import LossCount

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine = Engine(holowan_ip, holowan_port, engine_id)
        path_id = PATH_ID

        # 获取 path1
        path1 = engine.get_path_by_id(path_id)
        path1.reset()  # 重置 path1

        path1.l2r.loss = LossCount(start_offset=100, loss_count=10)
        # 应用损伤参数
        result = engine.apply_path_configuration(path1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result


if __name__ == '__main__':
    unittest.main()

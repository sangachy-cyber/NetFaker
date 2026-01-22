#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from holowan.v2._holowan_types import HoloWANReturn
from holowan.v2.pixel import (
    LengthFilter,
    MACFilter,
    Pixel,
    SCTPFilter,
    TCPFilter,
    TypeFilter,
    UDPFilter,
)
from holowan.v2.pixel._ip_filter import IPv4Filter, IPv6Filter
from holowan.v2.pixel._vlan_filter import VLANFilter

from tests.settings import ENGINE_ID, HOLOWAN_IP, HOLOWAN_PORT


class ExamplePixelFilter(unittest.TestCase):
    def setUp(self):
        """初始化测试环境"""
        self.pixel = Pixel(HOLOWAN_IP, HOLOWAN_PORT, ENGINE_ID)
        self.vlan = VLANFilter(fpcp="any", fpid="any")
        self.ipv4 = IPv4Filter(src="any", smask=32, dst="any", dmask=32, tos="any")
        self.ipv6 = IPv6Filter(src="any", dst="any")

    def test_basic_vlan_filter(self):
        """测试基本的VLAN过滤器功能"""
        self.assertEqual(self.vlan.filter_parameters["fpcp"], "any")
        self.assertEqual(self.vlan.filter_parameters["fpid"], "any")
        self.assertEqual(self.vlan.filter_parameters["enable_stag"], 0)

    def test_enable_second_tag(self):
        """测试启用第二个VLAN标签"""
        self.vlan.enable_second_tag(spcp="1", spid="100")
        self.assertEqual(self.vlan.filter_parameters["enable_stag"], 1)
        self.assertEqual(self.vlan.filter_parameters["spcp"], "1")
        self.assertEqual(self.vlan.filter_parameters["spid"], "100")

    def test_vlan_enable_disable_filter(self):
        """测试VLAN过滤器的启用和禁用功能"""
        self.assertTrue(self.vlan.enable)
        self.vlan.disable_filter()
        self.assertFalse(self.vlan.enable)
        self.vlan.enable_filter()
        self.assertTrue(self.vlan.enable)

    def test_basic_ipv4_filter(self):
        """测试基本的IPv4过滤器功能"""
        self.assertEqual(self.ipv4.filter_parameters["src"], "any")
        self.assertEqual(self.ipv4.filter_parameters["dst"], "any")
        self.assertEqual(self.ipv4.filter_parameters["smask"], 32)
        self.assertEqual(self.ipv4.filter_parameters["dmask"], 32)
        self.assertEqual(self.ipv4.filter_parameters["tos"], "any")

    def test_ipv4_enable_disable_filter(self):
        """测试IPv4过滤器的启用和禁用功能"""
        self.assertTrue(self.ipv4.enable)
        self.ipv4.disable_filter()
        self.assertFalse(self.ipv4.enable)
        self.ipv4.enable_filter()
        self.assertTrue(self.ipv4.enable)

    def test_basic_ipv6_filter(self):
        """测试基本的IPv6过滤器功能"""
        self.assertEqual(self.ipv6.filter_parameters["src"], "any")
        self.assertEqual(self.ipv6.filter_parameters["dst"], "any")

    def test_ipv6_enable_disable_filter(self):
        """测试IPv6过滤器的启用和禁用功能"""
        self.assertTrue(self.ipv6.enable)
        self.ipv6.disable_filter()
        self.assertFalse(self.ipv6.enable)
        self.ipv6.enable_filter()
        self.assertTrue(self.ipv6.enable)

    def test_set_mac_filter(self):
        """测试设置MAC过滤器"""
        mac_filter = MACFilter(src="39-A9-FA-D7-F5-E8", dst="any", type="any")
        result = self.pixel.set_filter(mac_filter)
        print(f"MAC Filter Result: {result}")
        self.assertIsInstance(result, HoloWANReturn)
        self.assertTrue(result)

    def test_set_ipv4_filter(self):
        """测试设置IPv4过滤器"""
        ipv4_filter = IPv4Filter(src="192.168.1.1", smask=24, dst="any", dmask=32, tos="any")
        result = self.pixel.set_filter(ipv4_filter)
        print(f"IPv4 Filter Result: {result}")
        self.assertIsInstance(result, HoloWANReturn)
        self.assertTrue(result)

    def test_set_ipv6_filter(self):
        """测试设置IPv6过滤器"""
        ipv6_filter = IPv6Filter(src="2001:db8::1", dst="any")
        result = self.pixel.set_filter(ipv6_filter)
        print(f"IPv6 Filter Result: {result}")
        self.assertIsInstance(result, HoloWANReturn)
        self.assertTrue(result)

    def test_set_tcp_filter(self):
        """测试设置TCP过滤器"""
        tcp_filter = TCPFilter(src="80", dst="any", check_version=0)
        result = self.pixel.set_filter(tcp_filter)
        print(f"TCP Filter Result: {result}")
        self.assertIsInstance(result, HoloWANReturn)
        self.assertTrue(result)

    def test_set_udp_filter(self):
        """测试设置UDP过滤器"""
        udp_filter = UDPFilter(src=[123, "456", "578-999"], dst=["12", 56, "589-999", "any"], check_version=0)
        result = self.pixel.set_filter(udp_filter)
        print(f"UDP Filter Result: {result}")
        self.assertIsInstance(result, HoloWANReturn)
        self.assertTrue(result)

    def test_set_sctp_filter(self):
        """测试设置SCTP过滤器"""
        sctp_filter = SCTPFilter(src=[123, "456", "578-999"], dst=["12", 56, "589-999", "any"], check_version=0)
        result = self.pixel.set_filter(sctp_filter)
        print(f"SCTP Filter Result: {result}")
        self.assertIsInstance(result, HoloWANReturn)
        self.assertTrue(result)

    def test_set_vlan_filter(self):
        """测试设置VLAN过滤器"""
        vlan_filter = VLANFilter(fpcp="any", fpid="any")
        vlan_filter.enable_second_tag(spid="any", spcp="any")
        result = self.pixel.set_filter(vlan_filter)
        print(f"VLAN Filter Result: {result}")
        self.assertIsInstance(result, HoloWANReturn)
        self.assertTrue(result)

    def test_set_type_filter(self):
        """测试设置类型过滤器"""
        type_filter = TypeFilter(capture_moment=1)
        result = self.pixel.set_filter(type_filter)
        print(f"Type Filter Result: {result}")
        self.assertIsInstance(result, HoloWANReturn)
        self.assertTrue(result)

    def test_set_length_filter(self):
        """测试设置长度过滤器"""
        length_filter = LengthFilter(length_from=64, length_to=1500)
        result = self.pixel.set_filter(length_filter)
        print(f"Length Filter Result: {result}")
        self.assertIsInstance(result, HoloWANReturn)
        self.assertTrue(result)

    def test_set_invalid_filter(self):
        """测试设置无效过滤器"""
        class InvalidFilter:
            pass

        with self.assertRaises(TypeError):
            self.pixel.set_filter(InvalidFilter())

if __name__ == '__main__':
    unittest.main()

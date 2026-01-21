#!/usr/bin/env python3
"""
集成验证脚本，用于端到端测试NetFaker API的完整流程。

测试内容包括：
1. 提交仿真任务
2. 查询任务状态
3. 下载生成的HoloWAN文件
4. 验证文件格式是否正确
"""

import os
import time

import pytest
from fastapi.testclient import TestClient

from src.netfaker.app.main import app

# 创建TestClient实例
client = TestClient(app)


class TestApiIntegration:
    """API集成测试类。"""

    @pytest.fixture(scope="class")
    def task_id(self):
        """创建仿真任务并返回任务ID。"""
        print("=== 测试仿真任务提交 ===")

        # 请求体
        payload = {
            "strategy": "rule",
            "target_ip": "172.30.153.236",
            "description": "集成测试任务",
            "segments": [{"type": "s0", "duration": 60}]
        }

        # 发送请求
        response = client.post("/api/v1/simulate", json=payload)

        # 验证响应
        assert response.status_code == 200, f"仿真任务提交失败: {response.status_code}"

        result = response.json()
        print("✓ 仿真任务提交成功")
        print(f"  任务ID: {result['task_id']}")
        print(f"  状态: {result['status']}")
        print(f"  消息: {result['message']}")

        return result['task_id']

    @pytest.fixture(scope="class")
    def task_status(self, task_id):
        """获取任务状态并返回。"""
        print(f"\n=== 测试任务状态查询 (ID: {task_id}) ===")

        # 轮询任务状态，直到完成或超时
        max_retries = 10
        retry_interval = 1  # 秒

        for i in range(max_retries):
            # 发送请求
            response = client.get(f"/api/v1/tasks/{task_id}")

            # 验证响应
            assert response.status_code == 200, f"任务状态查询失败: {response.status_code}"

            task_status = response.json()
            print(f"  尝试 {i+1}/{max_retries}: 状态 = {task_status['status']}")

            if task_status['status'] == "completed":
                print("✓ 任务完成")
                print(f"  创建时间: {task_status['created_at']}")
                print(f"  更新时间: {task_status['updated_at']}")
                print(f"  结果: {task_status['result']}")
                return task_status
            elif task_status['status'] == "failed":
                print("✗ 任务失败")
                print(f"  失败原因: {task_status}")
                pytest.fail("任务执行失败")

            # 等待一段时间后重试
            time.sleep(retry_interval)

        print(f"✗ 任务超时，未在 {max_retries} 秒内完成")
        pytest.fail(f"任务超时，未在 {max_retries} 秒内完成")

    @pytest.fixture(scope="class")
    def downloaded_file(self, task_status):
        """下载生成的HoloWAN文件并返回文件路径。"""
        print("\n=== 测试文件下载 ===")

        # 提取文件URL
        file_url = task_status["result"]["config_file_url"]

        # 提取文件名
        filename = os.path.basename(file_url)

        # 构建下载URL
        download_path = f"/api/v1/outputs/{filename}"

        # 发送请求
        response = client.get(download_path)

        # 验证响应
        assert response.status_code == 200, f"文件下载失败: {response.status_code}"
        assert response.headers["Content-Type"] == "text/plain; charset=utf-8", f"文件类型不正确: {response.headers['Content-Type']}"

        # 保存文件
        file_path = f"test_{filename}"
        with open(file_path, "wb") as f:
            f.write(response.content)

        print("✓ 文件下载成功")
        print(f"  文件名: {filename}")
        print(f"  文件大小: {len(response.content)} 字节")
        print(f"  保存路径: {file_path}")

        yield file_path

        # 清理测试文件
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"\n✓ 测试文件已清理: {file_path}")

    def test_task_submission(self, task_id):
        """测试仿真任务提交。"""
        assert task_id is not None, "任务ID不能为空"
        assert len(task_id) > 0, "任务ID长度不能为0"

    def test_task_completion(self, task_status):
        """测试任务完成状态。"""
        assert task_status["status"] == "completed", f"任务未完成，状态: {task_status['status']}"
        assert "result" in task_status, "任务结果不存在"
        assert "config_file_url" in task_status["result"], "配置文件URL不存在"

    def test_file_download(self, downloaded_file):
        """测试文件下载功能。"""
        assert os.path.exists(downloaded_file), "下载的文件不存在"
        assert os.path.getsize(downloaded_file) > 0, "下载的文件大小为0"

    def test_holowan_file_format(self, downloaded_file):
        """验证HoloWAN文件格式是否正确。"""
        print("\n=== 验证HoloWAN文件格式 ===")

        required_headers = [
            "HoloWAN Recorder File",
            "Operator:",
            "test_name:",
            "Destination:",
            "Start Time:",
            "End Time:",
            "Interval(sec):",
            "Packet Size(byte):",
            "Loss Average:",
            "Enable Reordering:",
            "Contents:",
            "Switch:",
            "------------------------------------------------"
        ]

        with open(downloaded_file, "r") as f:
            lines = f.readlines()

        # 检查文件是否为空
        assert len(lines) > 0, "文件为空"

        # 检查必需的头部字段
        found_headers = []
        for line in lines:
            stripped_line = line.strip()
            for header in required_headers:
                if stripped_line.startswith(header):
                    found_headers.append(header)
                    break

        # 验证所有必需的头部都存在
        missing_headers = [header for header in required_headers if header not in found_headers]
        assert len(missing_headers) == 0, f"缺少必需的头部字段: {missing_headers}"

        # 检查数据行
        data_lines = []
        in_data_section = False
        for line in lines:
            stripped_line = line.strip()
            if stripped_line == "------------------------------------------------":
                in_data_section = True
                continue
            if in_data_section and stripped_line:
                data_lines.append(stripped_line)

        assert len(data_lines) > 0, "文件中没有数据行"

        # 验证数据行格式
        for data_line in data_lines[:5]:  # 只验证前5行
            parts = data_line.split(",")
            assert len(parts) >= 6, f"数据行格式不正确: {data_line}"
            # 尝试转换为数值，验证数据类型
            for part in parts:
                float(part)  # 所有数据字段都应该是数值

        print("✓ HoloWAN文件格式验证成功")
        print(f"  总行数: {len(lines)}")
        print(f"  数据行数: {len(data_lines)}")
        print(f"  示例数据行: {data_lines[0]}")

"""端到端集成测试。

测试从API请求到生成仿真文件的完整流程，包括任务创建、状态查询、文件下载和结果分析。
"""

import os
import time

import pytest
from fastapi.testclient import TestClient

from netfaker.app.main import app
from netfaker.core.config import config
from netfaker.simcore.clustering.state_predictor import StatePredictor


@pytest.fixture(scope="module")
def client():
    """创建测试客户端。"""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def state_predictor():
    """创建状态预测器。"""
    return StatePredictor()


def _submit_simulation_request(client, expected_states):
    """提交仿真请求。"""
    simulation_request = {
        "strategy": "rule",
        "target_ip": "192.168.1.100",
        "description": "端到端测试仿真",
        "segments": [
            {"type": f"s{expected_states[0]}", "duration": 10},
            {"type": f"s{expected_states[1]}", "duration": 10}
        ]
    }

    response = client.post("/api/v1/simulate", json=simulation_request)
    assert response.status_code == 200
    task_result = response.json()
    assert task_result["status"] == "queued"
    assert task_result["message"] == "任务已提交"
    return task_result["task_id"]


def _poll_task_status(client, task_id):
    """轮询任务状态，直到完成。"""
    max_retries = 30
    retry_interval = 1
    task_status = None

    for _ in range(max_retries):
        response = client.get(f"/api/v1/tasks/{task_id}")
        assert response.status_code == 200
        task_status = response.json()
        if task_status["status"] == "completed":
            break
        time.sleep(retry_interval)
    else:
        pytest.fail(f"任务超时未完成，当前状态: {task_status['status']}")
    return task_status


def _get_file_path(task_status):
    """获取生成文件的路径。"""
    assert "result" in task_status
    assert "config_file_url" in task_status["result"]
    file_url = task_status["result"]["config_file_url"]
    file_path = os.path.join(config.output_dir, file_url)
    assert os.path.exists(file_path), f"生成的文件不存在: {file_path}"
    return file_path


def _parse_file(file_path):
    """解析文件内容，返回数据点和文件头。"""
    with open(file_path, "r") as f:
        lines = f.readlines()

    # 跳过文件头，找到数据开始位置
    data_start_idx = 0
    for i, line in enumerate(lines):
        if line.strip() == "------------------------------------------------":
            data_start_idx = i + 1
            break
    else:
        pytest.fail("文件格式错误，未找到数据分隔线")

    # 提取数据点
    data_points = []
    for line in lines[data_start_idx:]:
        line = line.strip()
        if not line:
            continue
        values = list(map(float, line.split(",")))
        if len(values) != 6:
            continue
        data_points.append(values)

    return data_points, "".join(lines[:data_start_idx])


def _validate_file_content(data_points, file_header):
    """验证文件内容。"""
    # 验证数据点数量
    expected_points = 200  # 2个segments，每个10秒，每秒10个数据点
    assert len(data_points) == expected_points, f"数据点数量不符，预期 {expected_points}，实际 {len(data_points)}"

    # 验证延迟值范围
    delay_values = []
    for point in data_points:
        delay_values.extend([point[0], point[3]])
    max_delay = max(delay_values)
    min_delay = min(delay_values)
    assert min_delay >= 0, f"延迟值过小: {min_delay}"
    assert max_delay <= 2000, f"延迟值过大: {max_delay}"

    # 验证丢包率范围
    loss_values = []
    for point in data_points:
        loss_values.extend([point[1], point[4]])
    max_loss = max(loss_values)
    min_loss = min(loss_values)
    assert min_loss >= 0, f"丢包率过小: {min_loss}"
    assert max_loss <= 100.0, f"丢包率过大: {max_loss}"

    # 验证文件头信息
    assert "HoloWAN Recorder File" in file_header, "文件头缺少HoloWAN标识"
    assert "Operator:" in file_header, "文件头缺少Operator信息"
    assert "test_name:" in file_header, "文件头缺少test_name信息"
    assert "Destination:" in file_header, "文件头缺少Destination信息"


def _predict_and_validate_states(data_points, expected_states, state_predictor):
    """预测并验证状态。"""
    # 将数据点划分为窗口
    window_size = 100
    windows = []
    for i in range(0, len(data_points), window_size):
        window = data_points[i:i+window_size]
        if len(window) == window_size:
            windows.append(window)

    # 验证窗口数量
    expected_windows = len(expected_states)
    assert len(windows) == expected_windows, f"窗口数量不符，预期 {expected_windows}，实际 {len(windows)}"

    # 对每个窗口预测状态并验证
    for window in windows:
        predicted_state = state_predictor.predict_from_data_points(window)
        assert isinstance(predicted_state, int), f"预测状态不是整数: {predicted_state}"
        assert predicted_state >= 0, f"预测状态为负数: {predicted_state}"


def _cleanup_file(file_path):
    """清理测试生成的文件。"""
    if os.path.exists(file_path):
        os.remove(file_path)


def test_end_to_end_simulation(client, state_predictor):
    """测试端到端仿真流程。"""
    expected_states = [0, 1]  # 对应segments中的s0和s1

    # 1. 提交仿真请求
    task_id = _submit_simulation_request(client, expected_states)

    # 2. 轮询任务状态，直到完成
    task_status = _poll_task_status(client, task_id)

    # 3. 获取生成的文件路径
    file_path = _get_file_path(task_status)

    # 4. 分析文件内容
    data_points, file_header = _parse_file(file_path)
    _validate_file_content(data_points, file_header)

    # 5. 使用StatePredictor验证生成的状态
    _predict_and_validate_states(data_points, expected_states, state_predictor)

    # 6. 清理测试生成的文件
    _cleanup_file(file_path)

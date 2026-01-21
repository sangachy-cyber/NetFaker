# 📡 NetFaker REST API 文档（v1）

> 所有接口路径前缀：`/api/v1`

---

## 1. 提交仿真任务

### 请求  
`POST /simulate`

### 请求头  
`Content-Type: application/json`

### 请求体（JSON）

| 字段 | 类型 | 必选 | 说明                                   |
|------|------|------|--------------------------------------|
| `strategy` | string | ✅ | 生成策略，固定值：`"model"` 或 `"rule"`        |
| `target_ip` | string | ✅ | 目标设备 IPv4 地址（格式如 `"172.30.153.236"`） |
| `description` | string | ❌ | 任务描述（可选）                             |
| `segments` | array | ✅ | 仿真阶段列表                               |

#### `segments` 数组元素结构：

| 字段 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `type` | string | ✅ | 阶段类型标识（如 `"s0"`） |
| `duration` | integer | ✅ | 持续时间（秒），必须是 **10 的正整数倍**（10, 20, 30...） |
| `condition` | string | ❌ | **预留字段，当前可选且无作用** |

### 示例请求
```json
{
  "strategy": "rule",
  "target_ip": "172.30.153.236",
  "description": "Xicen yuanshen(12-04 00:12:04)",
  "segments": [
    {"type": "s0", "duration": 60}
  ]
}
```

### 成功响应（HTTP 202 Accepted）
```json
{
  "task_id": "task_20251204_xyz789",
  "status": "queued",
  "message": "任务已提交"
}
```

---

## 2. 查询任务状态

### 请求  
`GET /tasks/{task_id}`

### 成功响应（仅展示 completed 状态）

```json
{
  "task_id": "task_20251204_xyz789",
  "status": "completed",
  "created_at": "2025-12-04T00:12:00Z",
  "updated_at": "2025-12-04T00:12:10Z",
  "result": {
    "config_file_url": "/api/v1/outputs/sim_task_20251204_xyz789.txt"
  }
}
```

---

## 3. 下载配置文件

### 请求  
`GET /outputs/{filename}.txt`

### 响应
- **Content-Type**: `text/plain; charset=utf-8`
- **Body**: 符合 **HoloWAN Recorder File 格式** 的纯文本内容

### 示例内容
```txt
HoloWAN Recorder File (www.msytest.com)
NetworkType: "4G"
test_name: "Xicen yuanshen(12-04 00:12:04)"
Destination: "172.30.153.236:8081"
Start Time: 2025-12-04 00:12:04
End Time: 2025-12-04 01:12:09
Interval(sec): 0.1
Packet Size(byte): 300
Enable Reordering: True
Contents: Delay1(ms),Loss1(%),Bandwidth1(Mbps),Delay2(ms),Loss2(%),Bandwidth2(Mbps)
Switch: 1,1,1,1,1,1
------------------------------------------------
17.185000,0.000000,0.025600,17.332750,0.000000,0.051200
16.346500,0.000000,0.025600,15.226750,0.000000,0.076800
17.549000,0.000000,0.025600,17.247000,0.000000,0.051200
14.703500,0.000000,0.025600,16.085500,0.000000,0.051200
19.316000,0.000000,0.051200,15.704250,0.000000,0.076800
...
```

> 🔹 文件由 NetFaker 根据 `segments` 生成，模拟指定时长内的网络行为序列  
> 🔹 每行数据对应一个时间点（间隔由 `Interval(sec)` 定义）  
> 🔹 `Destination` 端口固定为 `8081`（或根据需求配置）

---

## ⚠️ 错误码

| 状态码 | 说明 |
|--------|------|
| `400` | - `duration` 不是 10 的倍数- `target_ip` 无效- `segments` 缺少必要字段 |
| `404` | 任务或文件不存在 |
| `500` | 服务内部错误 |

---

✅ **最终确认**：
- 输入：`segments` 数组，`duration` 为 10 的倍数，`condition` 可选  
- 输出：仅返回 HoloWAN 标准格式 `.txt` 文件 URL  
- 文件内容完全兼容 HoloWAN Recorder 工具
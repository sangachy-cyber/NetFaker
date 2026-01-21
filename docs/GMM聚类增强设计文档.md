# 🧠 HoloWAN 网络状态聚类与可视化增强方案（最终整合版）

> **目标**：  
> 基于训练集窗口数据，通过 GMM 聚类自动发现网络状态，为**每个窗口分配全局唯一的 `state_id`**（含纯净态与混合态），并提供**可解释名称、置信度、可视化支持**，输出结构化结果供标注与建模使用。

---

## 一、核心设计原则

| 原则 | 说明 |
|------|------|
| ✅ **语义完备** | 区分纯净状态与混合状态（如 `"jittery/abnormal"`） |
| ✅ **ID 唯一** | 每个语义状态有独立整数 `state_id`（非仅主簇编号） |
| ✅ **概率驱动** | 基于 GMM 后验概率，支持置信度过滤 |
| ✅ **无数据泄露** | 所有模型（scaler + GMM）仅在训练集拟合 |
| ✅ **测试集对齐** | 测试集通过 GMM 预测获得一致 `state_id` |
| ✅ **可视化友好** | 支持 UMAP + 状态着色 + 混合标记 |

---

## 二、整体流程

```mermaid
graph LR
A[Train Parquet] --> B[提取 16D 特征]
B --> C[RobustScaler.fit_transform]
C --> D[GMM 聚类 (n=3)]
D --> E[计算后验概率]
E --> F[生成 state_name: 'stable' 或 'jittery/abnormal']
F --> G[映射到唯一 state_id]
G --> H[保存 train_with_state.parquet]

I[Test Parquet] --> J[提取 16D 特征]
J --> K[RobustScaler.transform]
K --> L[GMM.predict_proba]
L --> M[同上生成 state_id / state_name]
M --> N[保存 test_with_state.parquet]

C & D --> O[保存 scaler + gmm_model.joblib]
F & G --> P[生成 state_metadata.json]
H & N --> Q[UMAP 可视化]
```

---

## 三、关键输出字段（Parquet）

| 字段 | 类型 | 说明 |
|------|------|------|
| `state_id` | int32 | **全局唯一状态 ID**（0~5，含混合） |
| `state_name` | string | 可读名称（如 `"stable"` 或 `"jittery/abnormal"`） |
| `is_pure` | bool | 是否为高置信纯净状态（`proba ≥ 0.85`） |
| `state_proba` | float32 | 主状态后验概率 |
| `top2_state_ids` | list[int32] | 概率最高的两个基础状态 ID |
| `top2_state_probas` | list[float32] | 对应概率 |
| `base_state_id` | int32 | （可选）主基础状态（用于兼容颜色映射） |

> 🔸 所有字段同时写入 `train_with_state.parquet` 和 `test_with_state.parquet`

---

## 四、状态 ID 与命名体系（默认 K=3）

| `state_id` | `state_name` | 类型 | 基础状态组合 |
|-----------|-------------|------|--------------|
| 0 | `"stable"` | 纯净 | [0] |
| 1 | `"jittery"` | 纯净 | [1] |
| 2 | `"abnormal"` | 纯净 | [2] |
| 3 | `"jittery/abnormal"` | 混合 | [1, 2] |
| 4 | `"stable/abnormal"` | 混合 | [0, 2] |
| 5 | `"stable/jittery"` | 混合 | [0, 1] |

> ✅ 混合状态按字典序生成，确保确定性  
> ✅ 总状态数 = K + C(K,2)

---

## 五、元数据文件（`state_metadata.json`）

```json
{
  "algorithm": "gmm",
  "n_components": 3,
  "confidence_threshold": 0.85,
  "states": [
    {
      "state_id": 0,
      "state_name": "stable",
      "type": "pure",
      "base_states": [0],
      "color": "#4CAF50"
    },
    {
      "state_id": 3,
      "state_name": "jittery/abnormal",
      "type": "mixed",
      "base_states": [1, 2],
      "color": "#FB8C00"
    },
    // ... 其他状态
  ],
  "created_at": "2026-01-22T10:30:00Z"
}
```

> 📌 用于前端渲染、人工审核、API 返回等场景

---

## 六、可视化方案

### 6.1 降维方法
- **算法**：UMAP（保留局部结构）
- **输入**：`X_scaled`（训练集 + 测试集拼接）
- **输出**：2D 嵌入 `(N, 2)`

### 6.2 着色策略

| 元素 | 着色依据 | 说明 |
|------|--------|------|
| **点填充色** | `state_metadata[state_id]["color"]` | 每个 `state_id` 有专属颜色 |
| **点边框** | 若 `is_pure=False`，加灰色细边框 | 视觉区分模糊样本 |
| **透明度** | `alpha = min(1.0, state_proba * 1.2)` | 置信度越高越不透明 |
| **图例** | 按 `state_name` 分组 | 显示 `"stable"`, `"jittery/abnormal"` 等 |

### 6.3 输出格式
- **静态图**：`output/reports/cluster/clustering_gmm_{ts}.png`
- **交互图（可选）**：`output/reports/cluster/clustering_gmm_{ts}.html`（Plotly）

### 6.4 示例图描述
> - 绿色实心点：`"stable"`（高置信）  
> - 橙色带灰边点：`"jittery/abnormal"`（模糊）  
> - 紫色半透明点：`"stable/abnormal"`（低置信混合）

---

## 七、模块实现路径

| 功能 | 路径 |
|------|------|
| GMM 聚类器 | `src/netfaker/simcore/clustering/clusterers/gmm.py` |
| 状态 ID 映射器 | `src/netfaker/simcore/clustering/state_namer.py` |
| 聚类执行器 | `src/netfaker/simcore/clustering/cluster_runner.py` |
| 可视化器 | `src/netfaker/simcore/clustering/visualizer.py` |
| 入口脚本 | `scripts/run_clustering.py` |

---

## 八、使用方式（uv）

```bash
# 默认：GMM 3 类，θ=0.85，生成唯一 state_id + 可视化
uv run python scripts/run_clustering.py --algorithm gmm --assign-test-states

# 自定义阈值
uv run python scripts/run_clustering.py --algorithm gmm --confidence-threshold 0.8
```

---

## 九、下游应用支持

| 场景 | 如何使用 |
|------|--------|
| **初始标注** | 优先使用 `is_pure=True` 的样本 |
| **模型训练** | 以 `state_id` 为分类标签（6 类） |
| **状态监控** | 统计 `state_name` 分布，告警 `"abnormal/*"` |
| **人工审核** | 按 `state_name` 分组查看波形，修正元数据 |
| **API 服务** | 返回 `{ "state_id": 3, "state_name": "jittery/abnormal", "confidence": 0.62 }` |

---

## 十、优势总结

| 优势 | 价值 |
|------|------|
| **语义清晰** | 纯净 vs 混合一目了然 |
| **ID 唯一** | 支持高效存储与模型训练 |
| **概率透明** | 置信度可用于过滤或加权 |
| **可视化直观** | UMAP + 颜色 + 边框 + 透明度多维表达 |
| **工程完备** | 输出 Parquet + JSON + PNG，开箱即用 |

---


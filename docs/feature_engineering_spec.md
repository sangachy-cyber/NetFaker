# 弱网行为聚类：特征提取与归一化全流程规范（推荐版）

> 🎯 目标：保留绝对延迟差异 + 抑制异常值 + 区分 4 类网络模式  
> 🔧 原则：**不对原始延迟序列做任何 per-trace 归一化**，只对最终特征矩阵做全局鲁棒缩放

## 特征清单（共 10 维，全部基于上行）

| 序号 | 特征名 | 来源数据 | 计算方式 | 是否需预处理 | 归一化要求 | 最终类型 | 
|------|--------|--------|--------|------------|----------|--------| 
| 1 | `raw_mean_delay` | 原始上行延迟序列 `delay_up` (ms) | `np.mean(delay_up)` | ❌ 否 | **RobustScaler**（全局） | 连续 | 
| 2 | `p95_up` | 原始 `delay_up` | `np.percentile(delay_up, 95)` | ❌ 否 | **RobustScaler** | 连续 | 
| 3 | `p1_up` | 原始 `delay_up` | `np.percentile(delay_up, 1)` | ❌ 否 | **RobustScaler** | 连续 | 
| 4 | `std_up` | 原始 `delay_up` | `np.std(delay_up)` | ❌ 否 | **RobustScaler** | 连续 | 
| 5 | `trend_slope_up` | 原始 `delay_up` | 线性回归斜率（ms/s） | ❌ 否 | **RobustScaler** | 连续 | 
| 6 | `autocorr_lag5_up` | 原始 `delay_up` | `np.corrcoef(x[:-5], x[5:])[0,1]` | ❌ 否 | **RobustScaler** | 连续（[-1,1]）| 
| 7 | `loss_up` | 上行丢包序列 `loss_up` | `np.mean(loss_up)` | ❌ 否 | **RobustScaler** | 连续（[0,1]）| 
| 8 | `max_consec_loss_up` | `loss_up` | 最长连续丢包窗口数 → 转秒 | ✅ **log1p** | **RobustScaler** | 连续（长尾）| 
| 9 | `max_burst_up` | `delay_up` | 最长连续高延迟段（delay > p90）→ 秒 | ✅ **log1p** | **RobustScaler** | 连续（长尾）| 
|10 | `n_switches_up` | `delay_up` | 滑动窗口方差突变次数 | ✅ **log1p** | **RobustScaler** | 连续（长尾）| 

## 处理流程（代码逻辑）

```python
all_features = []

for delay_seq, loss_seq in zip(all_delay_up, all_loss_up):
    # --- 1. 提取原始特征（不修改原始序列！）--- 
    p95 = np.percentile(delay_seq, 95)
    p1 = np.percentile(delay_seq, 1)
    raw_mean = np.mean(delay_seq)
    std = np.std(delay_seq)
    
    # 趋势
    t = np.arange(len(delay_seq))
    slope = linregress(t, delay_seq).slope
    
    # 自相关
    if len(delay_seq) > 5:
        autocorr = np.corrcoef(delay_seq[:-5], delay_seq[5:])[0, 1]
    else:
        autocorr = 0.0
    
    # 丢包
    mean_loss = np.mean(loss_seq)
    max_consec_loss = compute_max_consecutive(loss_seq > 0.05) / 10.0  # 假设每窗0.1s
    
    # 动态行为
    burst = compute_max_burst_duration(delay_seq, threshold=np.percentile(delay_seq, 90))
    switches = count_regime_switches(delay_seq)  # 基于滑动方差突变检测

    # --- 2. 构建特征向量 --- 
    feat = [
        raw_mean,
        p95,
        p1,
        std,
        slope,
        autocorr,
        mean_loss,
        max_consec_loss,
        burst,
        switches
    ]
    all_features.append(feat)

# --- 3. 转为 NumPy 数组 --- 
X = np.array(all_features)  # shape: (N, 10)

# --- 4. 对长尾特征做 log(1+x) 预处理 --- 
log_indices = [7, 8, 9]  # max_consec_loss, max_burst, n_switches
X[:, log_indices] = np.log1p(X[:, log_indices])

# --- 5. 全局鲁棒归一化（这才是"鲁棒性归一化"）--- 
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)   # ✅ 最终输入聚类模型的特征
```

## 关键原则总结

| 问题 | 正确做法 | 
|------|--------| 
| **要不要对每条 trace 的延迟序列做归一化？** | ❌ **不要！** 会抹平"50ms 用户"和"400ms 用户"的差异 | 
| **如何处理极端 spike（如 2000ms）？** | ✅ 用 `RobustScaler`（基于 median/IQR），天然抗异常值 | 
| **长尾特征（burst, switches）怎么办？** | ✅ 先 `log1p`，再 `RobustScaler` | 
| **上下行都用吗？** | ❌ 推荐只用上行（避免冗余，提升信噪比） | 
| **归一化是在哪一步做的？** | ✅ **仅在最后对特征矩阵做一次 `RobustScaler`** | 

## 可视化建议（验证特征有效性）

```python
# 1. Pairplot（看特征两两关系）
df = pd.DataFrame(X_scaled, columns=[
    'mean', 'p95', 'p1', 'std', 'trend',
    'autocorr', 'loss', 'loss_burst', 'delay_burst', 'switches'
])
sns.pairplot(df, plot_kws={'alpha': 0.3})

# 2. t-SNE + 手动抽样画图
Z = TSNE(n_components=2).fit_transform(X_scaled)
plt.scatter(Z[:,0], Z[:,1], s=1, alpha=0.5)
```

✅ 如果看到：
- `trend` vs `delay_burst` 负相关 → 渐进 vs 突发分离
- `switches` vs `autocorr` 负相关 → 抖动 vs 平稳分离
- `mean` 和 `p95` 高但 `switches` 低 → 稳定弱网簇

→ 说明特征体系成功！

## 代码实现说明

### 特征提取函数

在 `src/preprocessing.py` 和 `scripts/analyze_10d_clustering.py` 中，`_compute_features` 函数实现了上述特征提取逻辑：

```python
def _compute_features(window_df):
    # 只使用上行数据
    del_up = window_df["delay_up"].values
    loss_up = window_df["loss_up"].values
    
    # 提取10维特征
    # ... 特征计算逻辑 ...
    
    # 构建特征向量
    features = np.array([
        raw_mean_delay,
        p95_up,
        p1_up,
        std_up,
        trend_slope_up,
        autocorr_lag5_up,
        loss_up_mean,
        max_consec_loss_up,
        max_burst_up,
        n_switches_up
    ])
    
    # 对长尾特征进行log1p预处理
    features[7:10] = np.log1p(features[7:10])
    
    # 确保所有值都是有限的
    features = np.where(np.isfinite(features), features, 0.0)
    
    return features
```

### 归一化实现

在 `src/preprocessing.py` 中，`stage9_recompute_condition_vectors` 函数实现了全局鲁棒归一化：

```python
def stage9_recompute_condition_vectors(datasets, assets, network_state_map):
    # 收集所有窗口的特征
    all_features = []
    # ... 特征收集逻辑 ...
    
    # 使用RobustScaler进行全局鲁棒归一化
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    all_features_scaled = scaler.fit_transform(all_features)
    
    # ... 聚类和其他处理 ...
```

## 特征分布检查

为了确保特征被有效缩放，建议在归一化后检查各维度的分布：

```python
for i in range(X_scaled.shape[1]):
    col = X_scaled[:, i]
    print(f"特征 {i}: min={col.min():.2f}, max={col.max():.2f}, "
          f"median={np.median(col):.2f}, IQR={np.percentile(col,75)-np.percentile(col,25):.2f}")
```

正常输出应类似：
```
特征 0: min=-2.10, max=3.45, median=0.00, IQR=1.00
特征 1: min=-1.88, max=4.21, median=0.00, IQR=1.00
...
```

如果看到某个维度的最小值或最大值非常大（如 >100），说明该维度没有被有效缩放，可能需要调整预处理方式。

## 版本控制

- 版本：v1.0
- 日期：2025-12-25
- 更新内容：初始版本，定义10维特征工程规范

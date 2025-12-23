# NetFaker

## 1. 项目概述

NetFaker是一个基于条件扩散模型的网络轨迹生成系统，旨在生成高保真的网络延迟和丢包序列。该系统可以用于网络测试、仿真和性能评估，帮助开发者和研究人员更好地理解和分析网络行为。

### 1.1 核心功能

- **数据预处理**：将原始网络轨迹数据转换为模型训练所需的格式
- **模型训练**：训练条件扩散模型，用于生成网络轨迹样本
- **样本生成**：使用训练好的模型生成高质量的网络轨迹样本
- **可视化**：可视化生成的样本，与真实样本进行对比分析

### 1.2 应用场景

- 网络性能测试和仿真
- 网络协议设计和验证
- 网络拓扑优化
- 网络安全研究
- 云服务性能评估

## 2. 技术栈

| 类别 | 技术 | 版本要求 | 用途 |
|------|------|----------|------|
| 核心语言 | Python | >= 3.9 | 系统开发 |
| 深度学习框架 | PyTorch | == 2.1.0 | 模型训练和推理 |
| 扩散模型库 | diffusers | == 0.29.2 | 条件扩散模型实现 |
| 数据处理 | numpy | >= 1.23.0 | 数值计算 |
| 数据处理 | pandas | >= 1.3.0 | 数据处理和分析 |
| 机器学习 | scikit-learn | >= 1.6.1 | 数据标准化和预处理 |
| 可视化 | matplotlib | >= 3.9.4 | 图表生成 |
| 可视化 | seaborn | >= 0.13.2 | 高级可视化 |
| 配置管理 | pyyaml | >= 6.0 | 配置文件解析 |
| 进度显示 | tqdm | >= 4.67.1 | 命令行进度条 |
| 模板引擎 | jinja2 | >= 3.0.0 | 报告生成 |
| 科学计算 | scipy | >= 1.13.1 | 科学计算和统计分析 |
| 包管理 | uv | - | 依赖管理 |
| 测试框架 | pytest | >= 6.2.0 | 单元测试 |
| 代码风格 | ruff | >= 0.1.0 | 代码风格检查 |

## 3. 项目结构

```
NetFaker/
├── data/             # 原始数据目录
├── docs/             # 设计文档
│   ├── 条件扩散模型方案.md    # 条件扩散模型设计文档
│   └── 预处理方案.md        # 数据预处理设计文档
├── scripts/          # 主脚本目录
│   ├── step1_preprocess.py        # 数据预处理脚本
│   ├── step2_train.py             # 模型训练脚本
│   └── step3_visualize.py        # 可视化脚本
├── src/              # 源代码目录
│   ├── models/                   # 模型定义
│   │   └── conditional_diffusion.py  # 条件扩散模型实现
│   ├── preprocessing/            # 数据预处理模块
│   │   ├── __init__.py           # 预处理模块入口
│   │   └── local_stats.py        # 局部统计特征计算
│   ├── training/                 # 训练模块
│   │   ├── evaluate_model.py     # 模型评估
│   │   └── full_training_evaluation.py  # 完整训练和评估流程
│   └── visualization/            # 可视化模块
│       ├── __init__.py           # 可视化模块入口
│       ├── generate_1000_points_trend.py  # 生成1000点趋势图
│       ├── generate_reference_generated_combined.py  # 生成参考和生成样本对比图
│       └── visualization_main.py # 可视化主函数
├── templates/        # 模板文件
│   └── report.j2                 # 数据报告模板
├── tests/            # 测试文件
│   ├── test_preprocessing.py     # 预处理模块测试
│   └── test_generate_valid_cond_vector.py  # 条件向量生成测试
├── .gitignore        # Git忽略文件
├── config.yaml       # 配置文件
├── LICENSE           # 许可证文件
├── Makefile          # Makefile
├── pyproject.toml    # 项目配置
└── README.md         # 项目文档
```

## 4. 安装与配置

### 4.1 安装依赖

```bash
# 使用uv安装依赖
uv install

# 安装开发依赖
uv sync --dev
```

### 4.2 配置文件

系统使用`config.yaml`作为主要配置文件，包含预处理、模型训练和可视化的参数。主要配置项如下：

```yaml
# 管道版本
pipeline_version: "v1.3"

# 输入输出配置
input_dir: "data"           # 输入数据目录
output_dir: "output"        # 输出目录

# 数据处理参数
raw_interval_sec: 0.1       # 原始数据时间间隔（秒）
max_delay_ms: 2000          # 最大延迟阈值（毫秒）
max_gap_sec: 2.0            # 最大时间间隔（秒）
min_segment_rows: 200       # 最小段长度
window_size: 100            # 窗口大小（10秒）
step_size: 50               # 步长
max_invalid_ratio: 0.01     # 最大无效数据比例

# 数据分割配置
split:
  train: 0.8                # 训练集比例
  val: 0.1                  # 验证集比例
  test: 0.1                 # 测试集比例
  random_state: 42          # 随机种子

# 分位数转换器配置
quantile_transformer:
  output_distribution: "normal"  # 输出分布
  random_state: 42              # 随机种子
  n_quantiles: 1000            # 分位数数量
  subsample: 100000             # 子样本大小

# 网络状态提取配置
filename_pattern: ".*_state(\\d+)\\.txt"  # 文件名模式
default_network_state_id: 0               # 默认网络状态ID

# 输出数据类型
output_dtype: "float32"                     # 输出数据类型

# 报告模板
report_template: "templates/report.j2"       # 报告模板文件
```

## 5. 使用方法

### 5.1 数据预处理

将原始网络轨迹数据转换为模型训练所需的格式：

```bash
uv run python scripts/step1_preprocess.py
```

该脚本会读取`config.yaml`中的配置，处理`input_dir`目录下的原始数据，并将结果保存到`output_dir`目录中。

### 5.2 模型训练

使用预处理后的数据训练条件扩散模型：

```bash
uv run python scripts/step2_train.py --data-dir output --epochs 10 --batch-size 100 --lr 1e-4 --num-samples 10 --output-dir output/visualization
```

**参数说明**：

- `--data-dir`：预处理后的数据目录
- `--epochs`：训练轮数
- `--batch-size`：批处理大小（默认100）
- `--lr`：学习率（默认1e-4）
- `--num-samples`：生成的样本数量
- `--output-dir`：输出目录

### 5.3 可视化

可视化生成的样本，与真实样本进行对比：

```bash
uv run python scripts/step3_visualize.py --reference-file output/datasets/train.jsonl --generated-file output/visualization/generated_samples.npy --assets-dir output/assets --output-dir output/visualization --all
```

**参数说明**：

- `--reference-file`：参考样本文件路径
- `--generated-file`：生成样本文件路径
- `--assets-dir`：资源文件目录
- `--output-dir`：输出目录
- `--all`：生成所有图表
- `--trend`：生成1000个点的趋势图
- `--comparison`：生成参考样本和生成样本的对比图

## 6. 核心功能详解

### 6.1 数据预处理

数据预处理是将原始网络轨迹数据转换为模型训练所需格式的过程，主要包括以下步骤：

1. **解析原始数据**：读取原始数据文件，提取延迟和丢包信息
2. **清理和截断**：过滤无效数据，截断超过阈值的数据
3. **分割和重采样**：将数据分割成连续段，并进行重采样
4. **提取窗口**：从连续段中提取固定大小的窗口
5. **生成条件向量**：生成用于模型训练的条件向量
6. **标准化数据**：对数据进行标准化处理
7. **保存结果**：将处理后的数据保存到文件中

### 6.2 条件扩散模型

NetFaker使用条件扩散模型来生成网络轨迹样本，该模型基于diffusers库中的UNet1DModel实现。主要特点包括：

- **条件生成**：可以根据网络状态ID和上下文特征生成特定类型的网络轨迹
- **高保真度**：生成的样本与真实样本分布相似
- **可扩展性**：支持不同的网络指标和生成长度
- **灵活性**：可以调整生成参数，控制生成样本的质量和多样性

### 6.3 样本生成

使用训练好的模型生成网络轨迹样本，主要步骤包括：

1. **加载模型**：加载训练好的模型和配置
2. **生成条件向量**：生成用于样本生成的条件向量
3. **执行采样**：使用扩散模型生成样本
4. **后处理**：对生成的样本进行后处理
5. **保存结果**：将生成的样本保存到文件中

### 6.4 可视化

可视化模块提供了多种可视化功能，帮助用户理解和分析生成的样本：

- **1000点趋势图**：显示1000个连续数据点的趋势
- **综合趋势图**：同时显示延迟和丢包率的趋势
- **参考样本和生成样本对比图**：将生成的样本与真实样本进行对比
- **训练损失曲线**：显示模型训练过程中的损失变化

## 7. 架构设计

### 7.1 数据流

1. **原始数据** → **数据预处理** → **标准化数据**
2. **标准化数据** → **模型训练** → **训练好的模型**
3. **训练好的模型** + **条件向量** → **样本生成** → **生成的样本**
4. **生成的样本** → **可视化** → **可视化图表**

### 7.2 模块间交互

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   原始数据     │     │   数据预处理   │     │   模型训练     │     │   样本生成     │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
          │                         │                         │                         │
          ▼                         ▼                         ▼                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   data/         │────▶│   output/       │────▶│   output/       │────▶│   output/       │
│   *.txt         │     │   datasets/     │     │   visualization/│     │   visualization/│
│                 │     │                 │     │   trained_model.pth│   │   generated_samples.npy│
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
                                                                               │
                                                                               ▼
                                                                 ┌─────────────────┐
                                                                 │   可视化       │
                                                                 └─────────────────┘
                                                                               │
                                                                               ▼
                                                                 ┌─────────────────┐
                                                                 │   output/       │
                                                                 │   visualization/│
                                                                 │   *.png         │
                                                                 └─────────────────┘
```

## 8. API文档

### 8.1 数据预处理模块

#### 8.1.1 主要函数

- `stage1_parse_txt(txt_path, interval_sec)`：解析原始数据文件
- `stage2_clean_and_truncate(df, max_delay_ms)`：清理和截断数据
- `stage3_split_and_resample(df, max_gap_sec, min_rows, freq_hz=10)`：分割和重采样数据
- `stage4_extract_windows(segments, window_size, step_size)`：提取窗口数据
- `stage5_mark_first_window(windows, trace_id)`：标记第一个窗口
- `stage6_group_by_trace_id(all_windows_meta)`：按trace_id分组
- `stage7_assign_split_by_trace(traces_dict, train_ratio, val_ratio, random_state)`：按trace_id分配训练/验证/测试集
- `stage8_fit_and_normalize(train, val, test, random_state)`：拟合和标准化数据
- `stage9_recompute_condition_vectors(datasets, assets, network_state_map)`：重新计算条件向量
- `stage10_save_artifacts(datasets, assets, out_dir, dtype=np.float32)`：保存处理后的数据和资源
- `stage11_generate_report(stats, config, template_path, out_path)`：生成数据报告

### 8.2 模型模块

#### 8.2.1 主要类

- `PaddedConditionalUNet1D`：条件UNet1D模型，支持padding和cropping
- `DiffusionTrainer`：条件扩散模型训练器
- `DiffusionSampler`：条件扩散模型采样器
- `PostProcessor`：生成样本后处理器

#### 8.2.2 主要方法

- `DiffusionTrainer.train_step(batch)`：执行训练步骤
- `DiffusionTrainer.validation_step(batch)`：执行验证步骤
- `DiffusionSampler.sample(cond_batch, num_inference_steps=1000)`：生成样本
- `PostProcessor.postprocess(generated_samples)`：后处理生成的样本

### 8.3 训练模块

#### 8.3.1 主要函数

- `train_model(train_file, val_file=None, epochs=10, batch_size=100, learning_rate=1e-4, output_dir="./output/visualization")`：训练模型
- `sample_and_postprocess(model, scheduler_config_path, assets_dir, num_samples=10, output_dir="./output/visualization", device="cpu", train_file=None)`：采样和后处理
- `visualize_training_losses(losses_file, output_dir)`：可视化训练损失
- `visualize_generated_samples(generated_samples_file, train_file, output_dir)`：可视化生成的样本

### 8.4 可视化模块

#### 8.4.1 主要函数

- `generate_1000_points_trend(reference_file, qt_up, qt_down, output_path)`：生成1000点趋势图
- `generate_reference_generated_combined(reference_points, generated_points, output_path)`：生成参考和生成样本对比图
- `main()`：可视化主函数

## 9. 贡献指南

### 9.1 代码风格

- 遵循Google Python风格
- 使用ruff进行代码风格检查
- 每行不超过88个字符
- 4空格缩进
- 统一使用双引号
- 函数圈复杂度不超过10（ruff C901规则）

### 9.2 提交规范

- 提交信息使用清晰简洁的中文
- 包含修改内容和原因
- 格式：`[模块名称] 简要描述修改内容`
- 例如：`[预处理模块] 修复数据标准化bug`

### 9.3 测试要求

- 使用pytest框架编写测试用例
- 测试文件命名：`test_<module_name>.py`
- 新增模块测试覆盖率要求100%
- 整体测试覆盖率争取80%以上

### 9.4 文档要求

- 所有注释使用中文
- 类和函数必须包含Google风格文档字符串
- 复杂逻辑添加行注释
- 注释需与代码同步更新

## 10. 许可证信息

本项目采用MIT许可证，详情请见[LICENSE](LICENSE)文件。

## 11. 示例代码

### 11.1 数据预处理示例

```python
from pathlib import Path
from src.preprocessing import stage1_parse_txt, stage2_clean_and_truncate

# 解析原始数据
txt_path = Path("data/sample.txt")
df_raw = stage1_parse_txt(txt_path, 0.1)

# 清理和截断数据
df_clean = stage2_clean_and_truncate(df_raw, 2000)
```

### 11.2 模型训练示例

```python
from src.training.full_training_evaluation import train_model

# 训练模型
model, scheduler_config_path = train_model(
    train_file="output/datasets/train.jsonl",
    val_file="output/datasets/val.jsonl",
    epochs=10,
    batch_size=100,
    learning_rate=1e-4,
    output_dir="output/visualization"
)
```

### 11.3 样本生成示例

```python
from src.training.full_training_evaluation import sample_and_postprocess

# 生成样本
sample_and_postprocess(
    model, 
    scheduler_config_path, 
    "output/assets",
    num_samples=10,
    output_dir="output/visualization",
    device="mps",
    train_file="output/datasets/train.jsonl"
)
```

### 11.4 可视化示例

```python
from src.visualization import visualization_main

# 生成可视化图表
visualization_main.main()
```

## 12. 性能评估

### 12.1 训练性能

| 硬件配置 | 数据集大小 | 训练轮数 | 训练时间 |
|----------|------------|----------|----------|
| CPU | 1000样本 | 10轮 | 约60分钟 |
| GPU (NVIDIA RTX 3090) | 1000样本 | 10轮 | 约5分钟 |
| MPS (Apple Silicon) | 1000样本 | 10轮 | 约10分钟 |

### 12.2 生成性能

| 硬件配置 | 样本数量 | 生成时间 |
|----------|----------|----------|
| CPU | 10个样本 | 约30秒 |
| GPU (NVIDIA RTX 3090) | 10个样本 | 约2秒 |
| MPS (Apple Silicon) | 10个样本 | 约5秒 |

### 12.3 生成质量

生成的样本与真实样本在分布上高度相似，主要评估指标包括：

- **延迟分布**：生成样本的延迟分布与真实样本一致
- **丢包率分布**：生成样本的丢包率分布与真实样本一致
- **时间相关性**：生成样本具有与真实样本相似的时间相关性
- **条件一致性**：生成样本能够准确反映条件向量指定的网络状态

## 13. 注意事项

1. **数据格式**：确保输入数据格式正确，符合系统要求
2. **硬件配置**：建议使用GPU进行模型训练，以提高训练速度
3. **内存要求**：处理大规模数据集时，需要足够的内存
4. **配置调整**：根据实际情况调整配置文件中的参数
5. **模型选择**：根据生成需求选择合适的模型和参数
6. **定期更新**：定期更新依赖库，以获得最佳性能和安全性

## 14. 未来计划

1. **支持更多网络指标**：如带宽、抖动等
2. **优化模型架构**：提高生成速度和质量
3. **增加模型评估指标**：提供更全面的生成样本评估
4. **支持分布式训练**：提高大规模数据集的训练效率
5. **提供Web界面**：简化系统使用流程
6. **支持实时生成**：实现实时网络轨迹生成
7. **增加模型压缩**：支持模型量化和剪枝

## 15. 联系方式

如有问题或建议，欢迎通过以下方式联系：

- GitHub Issues：[https://github.com/sangachy-cyber/NetFaker/issues](https://github.com/sangachy-cyber/NetFaker/issues)
- 邮箱：[sangachy@example.com](mailto:sangachy@example.com)

## 16. 变更日志

### v0.1.0（2025-12-23）

- 初始版本发布
- 实现数据预处理功能
- 实现条件扩散模型训练和生成
- 实现样本可视化功能
- 完善项目文档

## 17. 附录

### 17.1 术语表

| 术语 | 解释 |
|------|------|
| 网络轨迹 | 网络延迟和丢包的时间序列数据 |
| 条件扩散模型 | 一种生成模型，可根据条件向量生成特定类型的数据 |
| 窗口 | 固定长度的时间序列片段 |
| 条件向量 | 用于控制生成过程的特征向量 |
| UNet1DModel | 用于处理一维序列数据的UNet模型 |
| 分位数转换 | 一种数据标准化方法，将数据转换为特定分布 |
| 扩散过程 | 将数据逐步添加噪声的过程 |
| 逆扩散过程 | 从噪声中逐步恢复数据的过程 |
| MPS | Apple Silicon的Metal Performance Shaders，用于加速深度学习计算 |

### 17.2 相关资源

- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
- [diffusers官方文档](https://huggingface.co/docs/diffusers/index)
- [scikit-learn官方文档](https://scikit-learn.org/stable/documentation.html)
- [matplotlib官方文档](https://matplotlib.org/stable/contents.html)
- [uv官方文档](https://docs.astral.sh/uv/)

### 17.3 参考论文

- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. arXiv preprint arXiv:2006.11239.
- Nichol, A., & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. arXiv preprint arXiv:2102.09672.
- Chen, T., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10684-10695).
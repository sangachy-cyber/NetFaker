# NetFaker

网络仿真参数生成工具，用于生成各种网络场景下的仿真参数。

## 功能特性

- 🎯 支持多种网络场景仿真
- 📊 基于规则和机器学习的参数生成策略
- 🌐 提供RESTful API接口
- 🛠️ 命令行工具支持
- 📈 可视化仿真结果

## 快速开始

### 安装依赖

```bash
# 安装开发依赖
uv sync --all-extras

# 安装包（使netfaker可导入）
uv pip install -e .
```

### 启动开发服务器

```bash
uv run uvicorn netfaker.app.main:app --reload
```

服务器将在 `http://localhost:8000` 启动。

### 使用命令行工具

```bash
# 列出所有可用策略
uv run python -m scripts.main list

# 生成仿真参数
uv run python -m scripts.main generate --strategy rule_based --scenario video_streaming
```

## 项目结构

```
netfaker/
 ├── .gitignore 
 ├── .env.example 
 ├── .env                      # ← 本地环境变量（.gitignore 中忽略） 
 ├── README.md 
 ├── LICENSE 
 ├── pyproject.toml 
 ├── Makefile 
 │ 
 ├── src/ 
 │   └── netfaker/ 
 │       ├── __init__.py 
 │       │ 
 │       ├── core/                     # 公共核心模块（app + simcore 共用） 
 │       │   ├── __init__.py 
 │       │   ├── config.py             # 配置管理（Pydantic Settings） 
 │       │   ├── logger.py             # 统一日志初始化 
 │       │   ├── exceptions.py         # 自定义异常（如 SimulationError） 
 │       │   └── utils.py              # 通用工具函数（validate_ip, etc.） 
 │       │ 
 │       ├── app/                      # Web 服务层（FastAPI） 
 │       │   ├── __init__.py 
 │       │   ├── main.py               # FastAPI 应用入口 
 │       │   │ 
 │       │   ├── api/ 
 │       │   │   └── v1/ 
 │       │   │       ├── __init__.py 
 │       │   │       ├── endpoints.py  # REST 路由：/simulate, /tasks/{id} 
 │       │   │       └── schemas.py    # Pydantic 模型：请求/响应体 
 │       │   │ 
 │       │   ├── services/             # 业务逻辑协调 
 │       │   │   ├── __init__.py 
 │       │   │   ├── task_manager.py   # 异步任务调度（调用 simcore.generator） 
 │       │   │   └── hrf_generator.py  # 生成 HoloWAN Recorder File (.hrf) 
 │       │   │ 
 │       │   └── utils/                # Web 层专用工具 
 │       │       ├── __init__.py 
 │       │       └── http_client.py    # 调用下游 HoloWAN API 
 │       │ 
 │       └── simcore/                  # 仿真核心逻辑（原 ml/） 
 │           ├── __init__.py 
 │           │ 
 │           ├── generator.py          # ✨ 统一入口：根据 strategy 路由到具体实现 
 │           │ 
 │           ├── strategies/           # 仿真策略实现 
 │           │   ├── __init__.py 
 │           │   ├── base.py           # 抽象基类：ProfileGenerationStrategy 
 │           │   ├── ml_strategy.py    # ML 策略：调用 preprocessing + modeling 
 │           │   └── rule_strategy.py  # 规则策略：基于场景/条件的硬编码逻辑 
 │           │ 
 │           ├── io/                   # 文件 I/O（共用） 
 │           │   ├── __init__.py 
 │           │   ├── data_loader.py    # 读取原始数据（CSV, PCAP, JSON） 
 │           │   └── result_saver.py   # 保存处理结果、模型输出等 
 │           │ 
 │           ├── preprocessing/        # 数据预处理（仅 ML 策略使用） 
 │           │   ├── __init__.py 
 │           │   ├── cleaner.py        # 数据清洗 
 │           │   ├── feature_engineering.py  # 特征提取 
 │           │   └── clustering.py     # 聚类算法（KMeans, DBSCAN...） 
 │           │ 
 │           ├── modeling/             # 模型训练与推理（仅 ML 策略使用） 
 │           │   ├── __init__.py 
 │           │   ├── models.py         # 模型定义（Sklearn/PyTorch） 
 │           │   ├── trainer.py        # 训练逻辑 
 │           │   └── predictor.py      # 推理接口 
 │           │ 
 │           └── visualization/        # 可视化（可被 ML 或 Rule 使用） 
 │               ├── __init__.py 
 │               ├── plot_clusters.py  # 聚类结果图 
 │               ├── plot_profile.py   # 仿真参数分布图 
 │               └── exporters.py      # 导出 PNG/SVG/HTML 报告 
 │ 
 ├── scripts/                          # 命令行入口脚本（非包内模块） 
 │   ├── __init__.py 
 │   ├── train_model.py                # 训练 ML 模型 
 │   ├── generate_profile_cli.py       # 命令行生成仿真参数（支持 --strategy） 
 │   └── visualize_results.py          # 批量生成可视化报告 
 │ 
 ├── tests/                            # 测试代码 
 │   ├── __init__.py 
 │   ├── conftest.py                   # pytest 全局 fixture 
 │   │ 
 │   ├── unit/ 
 │   │   ├── test_core/ 
 │   │   │   └── test_config.py 
 │   │   ├── test_app/ 
 │   │   │   └── test_hrf_generator.py 
 │   │   └── test_simcore/ 
 │   │       ├── test_strategies/ 
 │   │       │   ├── test_ml_strategy.py 
 │   │       │   └── test_rule_strategy.py 
 │   │       ├── test_preprocessing/ 
 │   │       │   └── test_clustering.py 
 │   │       └── test_visualization/ 
 │   │           └── test_plot_clusters.py 
 │   │ 
 │   └── integration/ 
 │       └── test_end_to_end_simulation.py 
 │ 
 └── data/                             # 数据资产（运行时生成内容应 .gitignore） 
     ├── raw/                          # 原始输入数据（用户提供或采集） 
     ├── processed/                    # 预处理中间结果 
     ├── models/                       # 训练好的模型文件（.pkl, .pt） 
     └── outputs/                      # 最终输出（.hrf, 图片, 报告）
```

## 开发规范

请参考 [开发规范](docs/DEVELOPMENT_GUIDE.md) 了解项目的编码标准、工具链配置与协作流程。

## 测试

```bash
# 运行所有测试
uv run pytest

# 运行特定测试文件
uv run pytest tests/test_strategy.py

# 生成测试覆盖率报告
uv run pytest --cov=src --cov-report=html
```

## 代码风格

```bash
# 自动格式化代码
uv run ruff format .

# 运行代码风格检查
uv run ruff check .
```

## 贡献

欢迎提交Issue和Pull Request！请确保您的代码符合项目的开发规范。

## 许可证

MIT License

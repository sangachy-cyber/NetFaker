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
uv run uvicorn src.netfaker.app.main:app --reload
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
NetFaker/
 ├── .gitignore 
 ├── .env.example 
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
 │       │   │   ├── __init__.py 
 │       │   │   ├── endpoints.py      # REST 路由：/simulate, /tasks/{id} 
 │       │   │   └── schemas.py        # Pydantic 模型：请求/响应体 
 │       │   │   └── v1/               # 版本化API目录（预留） 
 │       │   │       └── __init__.py 
 │       │   │ 
 │       │   ├── database/             # 数据库操作 
 │       │   │   ├── __init__.py 
 │       │   │   └── task_db.py        # 任务数据库操作 
 │       │   │ 
 │       │   ├── models/               # 数据模型定义 
 │       │   │   └── __init__.py 
 │       │   │ 
 │       │   ├── services/             # 业务逻辑协调 
 │       │   │   ├── __init__.py 
 │       │   │   └── task_manager.py   # 异步任务调度（调用 simcore.generator） 
 │       │   │ 
 │       │   └── utils/                # Web 层专用工具 
 │       │       └── __init__.py 
 │       │ 
 │       └── simcore/                  # 仿真核心逻辑 
 │           ├── __init__.py 
 │           ├── generator.py          # ✨ 统一入口：根据 strategy 路由到具体实现 
 │           ├── strategy.py           # 策略基类和注册表 
 │           │ 
 │           ├── clustering/           # 聚类算法实现 
 │           │   ├── __init__.py 
 │           │   ├── cluster_runner.py # 聚类运行器 
 │           │   ├── feature_extractor.py # 特征提取器 
 │           │   ├── state_namer.py    # 状态命名器 
 │           │   ├── visualizer.py     # 聚类可视化 
 │           │   └── clusterers/       # 聚类算法实现 
 │           │       ├── __init__.py 
 │           │       ├── base.py       # 聚类基类 
 │           │       ├── dbscan.py     # DBSCAN 算法 
 │           │       └── gmm.py        # GMM 算法 
 │           │ 
 │           ├── dataset/              # 数据集处理 
 │           │   ├── __init__.py 
 │           │   ├── train_test_splitter.py # 训练测试集划分 
 │           │   └── window_extractor.py    # 窗口提取器 
 │           │ 
 │           ├── io/                   # 文件 I/O 
 │           │   ├── __init__.py 
 │           │   ├── dataset_saver.py  # 数据集保存 
 │           │   ├── holowan_loader.py # HoloWAN 文件加载 
 │           │   └── holowan_saver.py  # HoloWAN 文件保存 
 │           │ 
 │           ├── preprocessing/        # 数据预处理 
 │           │   ├── __init__.py 
 │           │   └── holowan_preprocessor.py # HoloWAN 数据预处理 
 │           │ 
 │           ├── strategies/           # 仿真策略实现 
 │           │   ├── __init__.py 
 │           │   ├── base.py           # 策略抽象基类 
 │           │   ├── ml_strategy.py    # ML 策略：占位符实现 
 │           │   └── rule_based.py     # 规则策略：基于场景/条件的硬编码逻辑 
 │           │ 
 │           ├── synthesizer/          # 序列合成器 
 │           │   ├── __init__.py 
 │           │   ├── rule_parser.py    # 规则解析器 
 │           │   ├── rule_synthesizer.py # 规则合成器 
 │           │   └── window_sampler.py # 窗口采样器 
 │           │ 
 │           ├── utils/                # 仿真核心工具 
 │           │   ├── __init__.py 
 │           │   └── holowan.py        # HoloWAN 文件处理工具 
 │           │ 
 │           └── visualization/        # 可视化 
 │               └── __init__.py 
 │ 
 ├── scripts/                          # 命令行入口脚本（非包内模块） 
 │   ├── __init__.py 
 │   ├── main.py                       # 主脚本入口 
 │   └── ... 
 │ 
 ├── tests/                            # 测试代码 
 │   └── ... 
 │ 
 └── data/                             # 数据资产（运行时生成内容应 .gitignore） 
     ├── raw/                          # 原始输入数据（用户提供或采集） 
     ├── processed/                    # 预处理中间结果 
     ├── models/                       # 训练好的模型文件（.pkl, .pt） 
     └── outputs/                      # 最终输出（.txt, 图片, 报告）
```

## 开发规范

请参考 [开发规范](.trae/rules/project_rules.md) 了解项目的编码标准、工具链配置与协作流程。

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
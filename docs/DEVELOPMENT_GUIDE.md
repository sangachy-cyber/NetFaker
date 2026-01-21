📘 NetFaker 项目开发规范 

本规范定义了 NetFaker 项目的工程结构、编码标准、工具链配置与协作流程，确保代码一致性、可维护性与可扩展性。 

一、项目目录结构 

netfaker/ 
├── src/netfaker/             # 所有可导入的 Python 包代码 
│   ├── core/                 # 公共模块（配置、日志、异常、工具） 
│   ├── app/                  # Web 服务层（FastAPI） 
│   └── simcore/              # 仿真核心逻辑（含 ML 与规则策略） 
├── scripts/                  # 命令行入口脚本（非包内） 
├── tests/                    # 单元测试与集成测试 
├── data/                     # 数据资产（原始/处理/输出） 
├── third_party/              # 第三方库归档目录 
├── pyproject.toml            # 项目元数据与工具配置 
├── .gitignore 
├── README.md 
└── Makefile                  # 开发快捷命令 

✅ 关键原则： 
- 只有 src/netfaker/ 是可导入的 Python 包 
- scripts/、tests/、data/、third_party/ 位于项目根目录，不参与包构建 
- 所有业务逻辑封装在 simcore/，app/ 仅做协议转换 
- third_party/ 目录用于归档项目依赖的第三方库，便于离线使用和版本控制 

二、依赖管理：使用 uv 

NetFaker 使用 uv 作为包管理器和虚拟环境工具。 

安装依赖 
安装运行时 + 开发依赖 
uv sync --all-extras 

安装第三方库 
make third-party-install

安装所有依赖（运行时+开发+第三方） 
make all-install

仅安装运行时依赖（生产部署） 
uv sync --only-group=default 

运行命令 
启动开发服务器 
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload 

运行测试 
uv run pytest 

格式化代码 
uv run ruff format . 

pyproject.toml 依赖分组示例 
[project.optional-dependencies] 
dev = [ 
    "ruff>=0.6.0", 
    "pytest>=8.0.0", 
    "pytest-asyncio", 
] 

⚠️ 必须先执行 uv pip install -e . 才能正确导入 netfaker 模块（因使用 src/ 布局）。 

三、代码风格与格式化：Ruff 

配置（pyproject.toml） 
[tool.ruff] 
line-length = 88 
target-version = "py311" 
lint.select = ["E", "W", "F", "I", "B", "C4", "SIM"] 
lint.ignore = ["E501"]  # 由 line-length 控制 

[tool.ruff.format] 
docstring-code-format = true 

开发流程 
自动格式化所有代码 
uv run ruff format . 

检查 lint 错误（CI 会强制通过） 
uv run ruff check . 

✅ 禁止手动调整缩进或空格，全部交由 Ruff 自动处理。 

四、注释规范：中文 + Google 风格 

所有模块、类、函数必须包含 中文 docstring，采用 Google 风格。 

函数注释示例 
def generate_hrf_file( 
    delay_ms: float, 
    loss_percent: float, 
    target_ip: str, 
    output_path: str, 
) -> bool: 
    """根据网络参数生成 HoloWAN Recorder File (.hrf)。 

    Args: 
        delay_ms: 网络延迟（毫秒），范围 [0, 1000]。 
        loss_percent: 丢包率（百分比），范围 [0.0, 100.0]。 
        target_ip: 目标设备 IPv4 地址。 
        output_path: 输出文件的绝对路径。 

    Returns: 
        是否成功生成文件。True 表示成功，False 表示失败。 

    Raises: 
        ValueError: 当输入参数超出有效范围时。 
        OSError: 当无法写入输出路径时。 
    """ 
    if not (0 <= delay_ms <= 1000): 
        raise ValueError("delay_ms 必须在 [0, 1000] 范围内") 
    # ... 实现逻辑 

模块注释示例（文件顶部） 
""" 
基于预设规则的仿真参数生成策略。 

支持常见网络场景（如视频流、在线游戏）的硬编码参数映射。 
适用于无历史数据或需快速响应的仿真任务。 
""" 

✅ 要求： 
- 所有 public 函数必须有 docstring 
- 参数、返回值、异常需明确说明 
- 使用中文，避免中英文混杂（专有名词如 HRF 除外） 

五、测试规范 

- 测试代码位于 tests/ 目录，结构与 src/ 对应 
- 使用 pytest，支持异步测试（pytest-asyncio） 
- 单元测试覆盖核心逻辑（simcore/strategies/, core/） 
- 集成测试验证端到端流程（POST /simulate → GET /tasks/{id}） 

运行测试 
全量测试 
uv run pytest 

仅单元测试 
uv run pytest tests/unit/ 

显示覆盖率（需安装 pytest-cov） 
uv run pytest --cov=src --cov-report=html 

六、开发工作流 

1. 初始化项目 
git clone `https://github.com/your-org/netfaker.git` 
cd netfaker 
uv sync --all-extras 
uv pip install -e .  # ← 关键：使 netfaker 可导入 

2. 日常开发 
编写代码 → 自动格式化 
uv run ruff format . 

运行本地服务 
uv run uvicorn app.main:app --reload 

提交前检查 
uv run ruff check . 
uv run pytest 

3. 添加新功能 
- 新策略？ → 在 simcore/strategies/ 新增 .py 文件并注册到 generator.py 
- 新 API？ → 在 app/api/v1/endpoints.py 添加路由，定义 schemas.py 模型 
- 新脚本？ → 在 scripts/ 添加入口，调用 simcore 或 app 的高层接口 

七、命名约定 
类型   规则   示例 
包/模块   小写 + 下划线   simcore, rule_strategy 

类   大驼峰   RuleBasedProfileStrategy 

函数/变量   小写 + 下划线   generate_hrf_file, loss_percent 

常量   全大写 + 下划线   MAX_DELAY_MS = 1000 

八、禁止事项 

- ❌ 在代码中写 from src.netfaker import ...（应为 from netfaker import ...） 
- ❌ 在 app/ 中直接实现聚类、文件读取等逻辑 
- ❌ 英文注释混用中文（除技术术语外） 
- ❌ 手动修改 uv.lock 文件 

📌 记住：结构服务于功能，规范服务于协作。  
保持简洁、清晰、一致，是 NetFaker 工程文化的基石。 

✅ 此规范自项目初始化起生效，所有贡献者必须遵守。
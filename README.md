# NetFaker - 网络轨迹生成系统

根据预处理方案.md 文档实现的网络轨迹数据预处理系统。

## 项目结构

```
.
├── data/                 # 原始数据文件
├── src/                  # 源代码
├── tests/                # 测试代码
├── output/               # 输出目录
├── templates/            # 模板文件
├── docs/                 # 文档
├── config.yaml           # 配置文件
├── main.py               # 主程序入口
└── pyproject.toml        # 项目配置
```

## 快速开始

1. 安装依赖：
   ```bash
   uv sync
   ```

2. 运行预处理：
   ```bash
   python main.py
   ```

3. 运行测试：
   ```bash
   pytest
   ```

## 配置

主要配置项在 `config.yaml` 文件中定义。

项目已配置使用清华大学 PyPI 镜像源以加快依赖下载速度。

## 代码格式化

使用 ruff 进行代码格式化：
```bash
ruff format .
```
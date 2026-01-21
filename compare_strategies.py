import os
import subprocess
import sys
import re

ANSI_ESCAPE = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-～]")

STRATEGIES = ["kmeans_two_stage", "isolation_forest_two_stage", "gmm_three_class", "auto_gmm_bic", "hierarchical", "dbscan"]
OUTPUT_DIR = "strategy_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_strategy(strategy):
    print(f"\n🚀 Running {strategy}...")
    env = os.environ.copy()
    plot_dir = os.path.join(OUTPUT_DIR, f"plots_{strategy}")
    env["CLUSTER_PLOT_DIR"] = plot_dir

    result = subprocess.run(
        [sys.executable, "build_train_dataset.py", "--strategy", strategy],
        capture_output=True,
        text=True,
        env=env,
    )

    if result.returncode != 0:
        print(f"❌ Build failed for {strategy}")
        print(result.stderr)
        return None, ""

    # === 提取最后一个 Class-wise Statistical Summary 块 ===
    clean_stdout = ANSI_ESCAPE.sub("", result.stdout)
    lines = clean_stdout.splitlines()
    summary_lines = []
    in_summary = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # 匹配包含emoji的统计摘要标题
        if "Class-wise Statistical Summary" in stripped:
            in_summary = True
            summary_lines = []  # 重置，只保留最后一次
            continue

        if not in_summary:
            continue

        # 遇到新的进度条或任务开始时退出
        if any(
            stripped.startswith(p) for p in ["✅", "📁", "🔄", "🔍", "Train:", "Test:", "解析文件", "生成窗口", "计算条件"]
        ):
            break

        # 跳过分隔线（允许带emoji的分隔线）
        if stripped.startswith("=") and all(c in "= " for c in stripped):
            continue

        # 保留有效行：类名或属性行
        if (stripped in ["S0:", "S1_light:", "S1_anomaly:"]) or (":" in stripped):
            summary_lines.append(line)

    summary_text = "\n".join(summary_lines)
    if not summary_text.strip():
        print(f"⚠️ No summary extracted for {strategy}")
    # ======================================================

    vis_result = subprocess.run(
        [sys.executable, "visualize_clusters.py"],
        capture_output=True,
        text=True,
        env=env,
    )
    if vis_result.returncode != 0:
        print(f"⚠️ Visualization failed for {strategy}: {vis_result.stderr}")
    else:
        print(f"✅ Plots saved to: {plot_dir}")

    return strategy, summary_text


def parse_summary(summary_text):
    classes = {}
    current_class = None
    for line in summary_text.splitlines():
        original_line = line
        stripped = line.strip()
        if not stripped:
            continue
        # 匹配类名（允许 S0 / S1_light / S1_anomaly）
        if stripped in ["S0:", "S1_light:", "S1_anomaly:"]:
            current_class = stripped.rstrip(":")
            classes[current_class] = {}
        elif current_class and ":" in stripped:  # 只要有冒号就尝试解析
            try:
                key, val = stripped.split(":", 1)
                key = key.strip()
                val = val.strip()
                # 提取数值（兼容 "23.45 ms" 或 "12 %"）
                parts = val.split()
                if not parts:
                    continue
                num_str = parts[0]
                if "." in num_str:
                    num = float(num_str)
                else:
                    num = int(num_str)
                classes[current_class][key] = num
            except Exception as e:
                continue  # 跳过解析失败的行
    return classes


def main():
    all_data = []
    for strategy in STRATEGIES:
        name, summary = run_strategy(strategy)
        if not summary.strip():
            continue
        parsed = parse_summary(summary)
        for cls_name, metrics in parsed.items():
            row = {"Strategy": name, "Class": cls_name}
            row.update(metrics)
            all_data.append(row)

    if not all_data:
        print("\n❌ No data collected.")
        return

    headers = [
        "Strategy",
        "Class",
        "样本数",
        "平均延迟",
        "P95 延迟",
        "平均丢包",
        "Burst 频率 (>5%)",
    ]
    col_widths = {
        h: max(len(h), max((len(str(r.get(h, ""))) for r in all_data), default=0)) + 2
        for h in headers
    }

    def fmt_row(row):
        return " | ".join(str(row.get(h, "N/A")).ljust(col_widths[h]) for h in headers)

    print("\n" + "=" * 100)
    print("📊 聚类策略对比汇总")
    print("=" * 100)
    print(fmt_row({h: h for h in headers}))
    print("-|-".join("-" * col_widths[h] for h in headers))
    for r in all_data:
        out = {
            k: (f"{v:.2f}" if isinstance(v, float) else str(v)) for k, v in r.items()
        }
        print(fmt_row(out))

    try:
        import csv

        with open(
            os.path.join(OUTPUT_DIR, "comparison.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in all_data:
                writer.writerow(
                    {
                        k: (round(v, 2) if isinstance(v, float) else v)
                        for k, v in r.items()
                    }
                )
        print(f"\n📁 CSV saved to: {OUTPUT_DIR}/comparison.csv")
    except Exception as e:
        print(f"⚠️ CSV save error: {e}")

    print(f"\n🎉 All results in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

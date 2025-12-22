# main.py (v1.3)
from pathlib import Path
import yaml, json, numpy as np, re
from datetime import datetime, timezone
from tqdm import tqdm
from src.preprocessing import *


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    # 自动填充 output_dir 时间戳（UTC）
    if "<auto>" in cfg["output_dir"]:
        auto_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
        cfg["output_dir"] = cfg["output_dir"].replace("<auto>", auto_name)
    
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "assets").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)

    # 构建 network_state_map
    pattern = cfg.get("filename_pattern", ".*_state(\\d+)\\.txt")
    default_id = cfg.get("default_network_state_id", 0)

    all_windows_meta = []
    failed_files = []
    empty_files = []

    txt_files = list(Path(cfg["input_dir"]).glob("*.txt"))
    network_state_map = {}
    for txt_file in txt_files:
        match = re.search(pattern, txt_file.name)
        network_state_map[txt_file.stem] = int(match.group(1)) if match else default_id

    for txt_file in tqdm(txt_files, desc="Processing files"):
        try:
            df_raw = stage1_parse_txt(txt_file, cfg["raw_interval_sec"])
            df_clean = stage2_clean_and_truncate(df_raw, cfg["max_delay_ms"])
            segments = stage3_split_and_resample(
                df_clean,
                max_gap_sec=cfg["max_gap_sec"],
                min_rows=cfg["min_segment_rows"],
                max_invalid_ratio=cfg.get("max_invalid_ratio", 0.01)
            )
            windows = []
            for seg in segments:
                seg_windows = stage4_extract_windows([seg], cfg["window_size"], cfg["step_size"])
                windows.extend(seg_windows)
            
            if not windows:
                empty_files.append(str(txt_file))
                continue

            trace_id = txt_file.stem
            window_metas = stage5_mark_first_window(windows, trace_id)
            all_windows_meta.extend(window_metas)

        except Exception as e:
            failed_files.append(f"{txt_file}: {str(e)}")
            continue

    # 保存失败/空文件
    if failed_files:
        with open(out_dir / "meta" / "failed_files.txt", "w") as f:
            f.write("\n".join(failed_files))
    if empty_files:
        with open(out_dir / "meta" / "empty_files.txt", "w") as f:
            f.write("\n".join(empty_files))

    # 聚合为 trace 级
    traces_dict = stage6_group_by_trace_id(all_windows_meta)
    
    # 按 trace 划分（整 trace 分配）
    train, val, test = stage7_assign_split_by_trace(
        traces_dict,
        train_ratio=cfg["split"]["train"],
        val_ratio=cfg["split"]["val"],
        random_state=cfg["split"]["random_state"]
    )
    
    # 归一化 + 列重命名
    renamed_datasets, assets = stage8_fit_and_normalize(
        train, val, test, random_state=cfg["quantile_transformer"]["random_state"]
    )

    # 重建条件向量（基于归一化数据，标记 keep）
    final_datasets, extra_assets = stage9_recompute_condition_vectors(
        renamed_datasets, assets, network_state_map
    )
    assets.update(extra_assets)

    # 保存（自动过滤 keep=False）
    stage10_save_artifacts(
        final_datasets, assets, out_dir, dtype=getattr(np, cfg["output_dtype"])
    )

    # 写入 schema.json
    schema = {
        "pipeline_version": "v1.3",
        "columns": ["timestamp", "del_up", "del_dn", "loss_up", "loss_dn"],
        "window_size": cfg["window_size"],
        "freq_hz": 10,
        "normalized": True,
        "condition_vector_dim": 23,
        "condition_vector_structure": {
            "global_features": list(range(13)),
            "local_features": list(range(13, 23))
        },
        "network_state_id_source": "filename_regex_or_default",
        "network_state_id_encoding": "integer category stored as float (e.g., 2.0)",
        "level2_normalization": "Z-score normalization applied to 22 float dimensions (indices 0–10 and 12–22), excluding the integer-encoded network_state_id at index 11.",
        "split_strategy": "Entire traces are assigned to a single split to prevent data leakage.",
        "normalization": {
            "del_up/del_dn": "QuantileTransformer(output_distribution='normal'), fitted on train set",
            "loss_up/loss_dn": "Clipped to [0.0, 1.0], no transformation applied"
        }
    }
    with open(out_dir / "meta" / "schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    # 报告统计（仅 keep=True）
    total_train = sum(1 for w in final_datasets["train"] if w["keep"])
    total_val = sum(1 for w in final_datasets["val"] if w["keep"])
    total_test = sum(1 for w in final_datasets["test"] if w["keep"])

    stats = {
        "total_windows": total_train + total_val + total_test,
        "train_count": total_train,
        "val_count": total_val,
        "test_count": total_test,
        "processed_files": len(txt_files),
        "failed_files": len(failed_files),
        "empty_files": len(empty_files),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    stage11_generate_report(stats, cfg, Path(cfg["report_template"]), out_dir / "data_report.md")


if __name__ == "__main__":
    main()
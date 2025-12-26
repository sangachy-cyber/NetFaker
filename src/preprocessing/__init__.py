"""预处理模块（旧接口，保留向后兼容性）"""

# 导出所有类和函数
__all__ = [
    # 数据类型
    "WindowMetaRaw",
    "WindowMetaRenamed",
    "WindowMetaNormed",
    
    # 主要类
    "PreprocessingPipeline",
    "DataSaver",
    "ReportGenerator",
    
    # 旧接口（保留向后兼容性）
    "stage1_parse_txt",
    "stage2_clean_and_truncate",
    "stage3_split_and_resample",
    "stage4_extract_windows",
    "stage5_mark_first_window",
    "stage6_group_by_trace_id",
    "stage7_assign_split_by_trace",
    "stage8_fit_and_normalize",
    "stage9_recompute_condition_vectors",
    "stage10_save_artifacts",
    "stage11_generate_report",
    
    # 新接口
    "compute_window_features",
    "compute_local_features",
    "merge_features",
    "normalize_condition_vector",
]

# 从preprocessing子模块导入所有功能
from .pipeline import PreprocessingPipeline
from .processing import (
    parse_txt as stage1_parse_txt,
    clean_and_truncate as stage2_clean_and_truncate,
    split_and_resample as stage3_split_and_resample,
    extract_windows as stage4_extract_windows,
    mark_first_window as stage5_mark_first_window,
    group_by_trace_id as stage6_group_by_trace_id,
    assign_split_by_trace as stage7_assign_split_by_trace,
    fit_and_normalize as stage8_fit_and_normalize,
)
from .features import (
    compute_window_features,
    compute_local_features,
    merge_features,
    normalize_condition_vector,
)
from .io import DataSaver, ReportGenerator
from .data import WindowMetaRaw, WindowMetaRenamed, WindowMetaNormed


# 保留旧接口的兼容性
def stage9_recompute_condition_vectors(*args, **kwargs):
    """旧接口兼容函数，已迁移到pipeline模块"""
    raise NotImplementedError("stage9_recompute_condition_vectors已迁移到PreprocessingPipeline类")


def stage10_save_artifacts(*args, **kwargs):
    """旧接口兼容函数，已迁移到DataSaver类"""
    raise NotImplementedError("stage10_save_artifacts已迁移到DataSaver类")


def stage11_generate_report(*args, **kwargs):
    """旧接口兼容函数，已迁移到ReportGenerator类"""
    raise NotImplementedError("stage11_generate_report已迁移到ReportGenerator类")

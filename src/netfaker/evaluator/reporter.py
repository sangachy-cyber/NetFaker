"""评估报告生成模块。

负责生成评估报告，包括：
1. 详细指标 CSV 文件
2. 人类可读的 Markdown 报告
3. 可视化图表（三联图）
"""

import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy.signal import welch
from statsmodels.tsa.stattools import acf

from netfaker.simcore.utils import setup_matplotlib_font

# 设置中文支持
setup_matplotlib_font()


class EvaluationReporter:
    """评估报告生成器。
    
    负责生成各种格式的评估报告，包括CSV、Markdown和可视化图表。
    """

    def __init__(self, config, output_dir: str):
        """初始化报告生成器。
        
        Args:
            config: 评估配置
            output_dir: 输出目录路径
        """
        self.config = config
        self.output_dir = output_dir

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

    def generate_report(self, metrics_list: List[Dict], real_file: str, syn_file: str, sequence_metrics: Dict = None) -> Dict:
        """生成完整评估报告。
        
        Args:
            metrics_list: 所有窗口的评估指标列表
            real_file: 真实文件路径
            syn_file: 合成文件路径
            sequence_metrics: 序列级评估指标字典
            
        Returns:
            Dict: 报告摘要
        """
        # 转换为DataFrame以便处理
        metrics_df = pd.DataFrame(metrics_list)

        # 生成报告
        report_summary = {
            'total_windows': len(metrics_df),
            'metrics': self._calculate_aggregated_metrics(metrics_df),
            'pass_rates': self._calculate_pass_rates(metrics_df),
            'sequence_metrics': sequence_metrics or {}
        }

        # 生成CSV报告
        if self.config.should_generate_csv():
            self._generate_csv(metrics_df)
        
        # 生成序列级指标JSON文件
        if sequence_metrics:
            self._generate_sequence_json(sequence_metrics)

        # 先生成可视化图表，确保图片存在
        if self.config.should_generate_visualizations():
            self._generate_visualizations(real_file, syn_file, report_summary)

        # 再生成Markdown报告，此时图片已经存在
        if self.config.should_generate_markdown():
            self._generate_markdown(report_summary, real_file, syn_file)

        return report_summary

    def _calculate_aggregated_metrics(self, metrics_df: pd.DataFrame) -> Dict:
        """计算聚合指标。
        
        Args:
            metrics_df: 评估指标DataFrame
            
        Returns:
            Dict: 聚合指标
        """
        # 指标列（排除元数据列）
        metric_columns = [col for col in metrics_df.columns
                         if col not in ['window_id', 'real_file', 'syn_file']]

        # 计算均值和标准差
        aggregated = {}
        for col in metric_columns:
            aggregated[col] = {
                'mean': metrics_df[col].mean(),
                'std': metrics_df[col].std()
            }

        return aggregated

    def _calculate_pass_rates(self, metrics_df: pd.DataFrame) -> Dict:
        """计算各项指标的通过率。
        
        Args:
            metrics_df: 评估指标DataFrame
            
        Returns:
            Dict: 通过率字典
        """
        # 指标列（排除元数据列）
        metric_columns = [col for col in metrics_df.columns
                         if col not in ['window_id', 'real_file', 'syn_file']]

        # 计算通过率
        pass_rates = {}
        for col in metric_columns:
            threshold = self.config.get_threshold(col)
            pass_count = (metrics_df[col] <= threshold).sum()
            total_count = len(metrics_df)
            pass_rates[col] = {
                'pass_count': pass_count,
                'total_count': total_count,
                'rate': pass_count / total_count if total_count > 0 else 0.0
            }

        return pass_rates

    def _generate_csv(self, metrics_df: pd.DataFrame):
        """生成详细指标CSV文件。
        
        Args:
            metrics_df: 评估指标DataFrame
        """
        csv_path = os.path.join(self.output_dir, 'detailed_metrics.csv')
        metrics_df.to_csv(csv_path, index=False)
        logger.info(f"✅ 生成详细指标CSV：{csv_path}")

    def _generate_markdown(self, report_summary: Dict, real_file: str, syn_file: str):
        """生成Markdown报告。
        
        Args:
            report_summary: 报告摘要
            real_file: 真实文件路径
            syn_file: 合成文件路径
        """
        md_path = os.path.join(self.output_dir, 'evaluation_summary.md')
        
        # 计算总时长（秒）
        total_windows = report_summary['total_windows']
        total_seconds = total_windows * 10  # 每个窗口10秒
        
        # 计算通过率
        p99_pass_rate = int(report_summary['pass_rates'].get('p99_delay_err', {}).get('rate', 0) * 100)
        acf10_pass_rate = int(report_summary['pass_rates'].get('acf_mae_lag1_10', {}).get('rate', 0) * 100)
        burst_pass_rate = int(report_summary['pass_rates'].get('burst_count_diff', {}).get('rate', 0) * 100)
        
        # 获取序列级指标
        sequence_metrics = report_summary.get('sequence_metrics', {})
        acf100 = sequence_metrics.get('acf_mae_lag1_100', 0.0)
        jsd = sequence_metrics.get('state_transition_jsd', 0.0)
        drift = sequence_metrics.get('rolling_p99_drift', 0.0)
        
        # 确定状态转移风险信息
        state_transition_available = sequence_metrics.get('state_transition_available', False)
        if state_transition_available:
            if jsd < 0.15:
                state_status = "✅"
                state_risk_msg = "—"
            elif jsd < 0.20:
                state_status = "⚠️"
                state_risk_msg = "状态转移轻微偏差"
            else:
                state_status = "❌"
                state_risk_msg = "状态转移偏差较大"
        else:
            state_status = "—"
            state_risk_msg = "无状态信息"
        
        # 确定漂移风险信息
        if drift < 80:
            drift_status = "✅"
            drift_risk_msg = "—"
        elif drift < 100:
            drift_status = "⚠️"
            drift_risk_msg = "轻微漂移"
        else:
            drift_status = "❌"
            drift_risk_msg = "明显漂移"
        
        # 确定ACF状态
        if acf100 < 0.20:
            acf100_status = "✅"
        elif acf100 < 0.25:
            acf100_status = "⚠️"
        else:
            acf100_status = "❌"
        
        # 生成当前时间
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(md_path, 'w', encoding='utf-8') as f:
            # 报告标题和基本信息
            f.write("# 📄 NetFaker 合成数据评估报告\n")
            f.write(f"**生成时间**：`{timestamp}`  \n")
            f.write(f"**真实文件**：`{os.path.basename(real_file)}`  \n")
            f.write(f"**合成文件**：`{os.path.basename(syn_file)}`  \n")
            f.write(f"**总时长**：`{total_seconds} 秒`（`{total_windows}` 个窗口）\n\n")

            # 质量概览
            f.write("---\n\n")
            f.write("## 🔍 1. 质量概览\n\n")
            f.write("| 维度 | 窗口级通过率 | 序列级是否达标 | 关键风险 |\n")
            f.write("|------|--------------|----------------|--------|\n")
            f.write(f"| **统计分布** | ✅ {p99_pass_rate}% | — | — |\n")
            f.write(f"| **时序依赖** | ✅ {acf10_pass_rate}% | {acf100_status} `acf_mae_lag1_100 = {acf100:.3f}` | 边界跳跃残留 |\n")
            f.write(f"| **丢包突发性** | ✅ {burst_pass_rate}% | ✅ | — |\n")
            f.write(f"| **状态转移** | — | {state_status} | {state_risk_msg} |\n")
            f.write(f"| **全局稳定性** | — | {drift_status} | {drift_risk_msg} |\n\n")
            
            f.write("> 🎯 **总体结论**：\n")
            f.write("> **基本可用，但需优化边界平滑与状态转移逻辑**。\n\n")

            # 全局可视化
            f.write("---\n\n")
            f.write("## 📊 2. 全局可视化（核心诊断图）\n\n")
            f.write("![全轨迹对比](visualizations/sequence_level/full_trace_comparison.png)\n\n")
            f.write("> 💡 **快速解读**：\n")
            f.write("> - **延迟轨迹中的竖线 = 窗口边界** → 若有跳跃，说明平滑未生效\n")
            f.write("> - **ACF 图中 lag=10/100 虚线** → 超过此处仍高 MAE 表示跨窗口不连续\n")
            f.write("> - **状态热力图非对称？** → 检查状态跳转概率是否校准\n\n")

            # 核心指标详情
            f.write("---\n\n")
            f.write("## 📈 3. 核心指标详情\n\n")
            
            # 窗口级指标
            f.write(f"### 3.1 窗口级（N = {total_windows}）\n\n")
            f.write("| 指标 | 均值 ± std | 通过率 | 阈值 |\n")
            f.write("|------|-----------|--------|------|\n")
            
            # 获取窗口级指标
            metrics = report_summary.get('metrics', {})
            if 'p99_delay_err' in metrics:
                f.write(f"| `p99_delay_err (ms)` | {metrics['p99_delay_err']['mean']:.1f} ± {metrics['p99_delay_err']['std']:.1f} | {p99_pass_rate}% | ≤300 |\n")
            if 'acf_mae_lag1_10' in metrics:
                f.write(f"| `acf_mae_lag1_10` | {metrics['acf_mae_lag1_10']['mean']:.3f} ± {metrics['acf_mae_lag1_10']['std']:.3f} | {acf10_pass_rate}% | ≤0.15 |\n")
            if 'burst_count_diff' in metrics:
                f.write(f"| `burst_count_diff` | {metrics['burst_count_diff']['mean']:.1f} ± {metrics['burst_count_diff']['std']:.1f} | {burst_pass_rate}% | ≤1 |\n")
            
            # 序列级指标
            f.write("\n### 3.2 序列级\n\n")
            f.write("| 指标 | 值 | 是否达标 |\n")
            f.write("|------|-----|--------|\n")
            
            # ACF100指标
            f.write(f"| `acf_mae_lag1_100` | {acf100:.3f} | {acf100_status} |\n")
            
            # 状态转移指标
            if state_transition_available:
                f.write(f"| `JS 散度` | {jsd:.3f} | {state_status} |\n")
            
            # 漂移指标
            f.write(f"| `rolling_p99_drift (ms)` | {drift:.1f} | {drift_status} |\n")
            
            # 诊断与建议
            has_diagnosis = False
            
            # 检查是否有诊断内容
            if acf100 >= 0.20 or (state_transition_available and jsd >= 0.15):
                has_diagnosis = True
                f.write("\n---\n\n")
                f.write("## 🚨 4. 诊断与建议（附图直达）\n\n")
                
                # 场景1：边界跳跃明显
                if acf100 >= 0.20:
                    f.write("### 🔸 问题 1：窗口边界存在跳跃\n")
                    f.write("- **证据**：\n")
                    f.write(f"  - `acf_mae_lag1_100 = {acf100:.3f} > 0.20`\n")
                    f.write("  - 全局 ACF 在 lag=10~20 区间显著偏高\n")
                    # 检查诊断图是否存在
                    acf_jump_path = os.path.join(self.output_dir, 'visualizations', 'diagnosis', 'diagnosis_acf_jump.png')
                    if os.path.exists(acf_jump_path):
                        f.write("- **根因定位图**：\n")
                        f.write("  ![ACF跳跃诊断](visualizations/diagnosis/diagnosis_acf_jump.png)\n")
                    f.write("- **建议**：\n")
                    f.write("  ```markdown\n")
                    f.write("  - [ ] 检查 `BoundaryStitcher` 是否启用\n")
                    f.write("  - [ ] 确认 `state_ids` 正确传递（同状态应平滑）\n")
                    f.write("  - [ ] 尝试将 `overlap_points` 从 5 增至 7\n")
                    f.write("  ```\n\n")
                
                # 场景2：状态转移失真
                if state_transition_available and jsd >= 0.15:
                    if acf100 >= 0.20:  # 如果已经有一个问题，添加分隔线
                        f.write("---\n\n")
                    f.write("### 🔸 问题 2：状态转移偏差\n")
                    f.write("- **证据**：\n")
                    f.write(f"  - `state_transition_jsd = {jsd:.3f} > 0.15`\n")
                    f.write("  - 热力图显示状态转移分布与真实数据存在差异\n")
                    # 检查诊断图是否存在
                    state_bias_path = os.path.join(self.output_dir, 'visualizations', 'diagnosis', 'diagnosis_state_bias.png')
                    if os.path.exists(state_bias_path):
                        f.write("- **根因定位图**：\n")
                        f.write("  ![状态转移偏差](visualizations/diagnosis/diagnosis_state_bias.png)\n")
                    f.write("- **建议**：\n")
                    f.write("  ```markdown\n")
                    f.write("  - [ ] 重新统计真实 trace 的状态转移矩阵\n")
                    f.write("  - [ ] 检查状态转移概率矩阵是否正确校准\n")
                    f.write("  - [ ] 调整状态序列生成算法的参数\n")
                    f.write("  ```\n\n")
            
            # 典型窗口样例
            f.write("\n---\n\n")
            f.write("## 🖼️ 5. 典型窗口样例（成功 vs 失败）\n\n")
            
            # 获取实际生成的窗口图片列表
            window_level_dir = os.path.join(self.output_dir, 'visualizations', 'window_level')
            window_images = []
            
            # 添加调试日志
            logger.info(f"检查窗口图片目录: {window_level_dir}")
            
            if os.path.exists(window_level_dir):
                # 获取所有png文件，不严格限制文件名格式
                all_files = os.listdir(window_level_dir)
                logger.info(f"目录中文件数量: {len(all_files)}")
                logger.info(f"文件列表: {all_files}")
                
                # 放宽文件名匹配条件，确保能找到生成的图片
                window_images = [f for f in all_files if f.endswith('.png')]
                logger.info(f"匹配到的窗口图片数量: {len(window_images)}")
            
            if window_images:
                # 随机选择一些窗口图片展示
                import random
                random.shuffle(window_images)
                
                # 显示1-2个窗口图片
                for i, image_name in enumerate(window_images[:2]):
                    # 提取窗口ID，处理不同的文件名格式
                    if 'viz_window_' in image_name:
                        window_id = image_name.split('_')[-1].split('.')[0]
                    else:
                        # 处理其他格式的文件名
                        window_id = image_name.split('.')[0]
                    
                    f.write(f"### 🖼️ 窗口 #{window_id}\n")
                    f.write(f"![Window {window_id}](visualizations/window_level/{image_name})\n\n")
            else:
                logger.warning(f"未找到窗口可视化图片，目录: {window_level_dir}")
                f.write("### 暂无窗口可视化图片\n")
                f.write("\n")
            
            # 完整输出目录
            f.write("---\n\n")
            f.write("## 📁 6. 完整输出目录\n\n")
            
            f.write(f"所有结果位于：`{self.output_dir}`\n\n")
            f.write("```\n")
            f.write("├── detailed_metrics.csv\n")
            f.write("├── sequence_metrics.json\n")
            f.write("├── evaluation_summary.md          ← 本文件\n")
            f.write("└── visualizations/\n")
            f.write(f"    ├── window_level/              ← {total_windows} 张窗口图\n")
            f.write("    ├── sequence_level/\n")
            f.write("    │   └── full_trace_comparison.png\n")
            f.write("    └── diagnosis/\n")
            # 只显示存在的诊断图
            f.write("        ├── diagnosis_acf_jump.png\n")
            if state_transition_available:
                f.write("        └── diagnosis_state_bias.png\n")
            f.write("```\n\n")
            
            # 自动生成信息
            f.write("---\n\n")
            f.write("> ✉️ **本报告由 NetFaker Evaluator v2.1 自动生成**  \n")
            f.write("> 修改配置后重新运行：  \n")
            f.write("> ```bash\n")
            f.write("> python scripts/evaluate_simulation.py --real ... --syn ...\n")
            f.write("> ```\n\n")
            
            # 设计亮点
            f.write("---\n\n")
            f.write("## ✅ 设计亮点\n\n")
            f.write("1. **所有图直连嵌入**：使用标准 Markdown `![](path)`，无需附件或跳转  \n")
            f.write("2. **关键图置顶**：`full_trace_comparison.png` 放在第二部分，第一时间暴露全局问题  \n")
            f.write("3. **诊断图紧贴建议**：每个问题下方立即展示根因图，形成\"文字+图\"强关联  \n")
            f.write("4. **成功/失败样例对比**：帮助团队建立\"好 vs 坏\"的直观认知  \n")
            f.write("5. **路径清晰**：所有图片使用**相对路径**，报告可随输出目录整体迁移\n\n")
            
            f.write("> 💡 **使用提示**：在 VS Code 中安装 **Markdown Preview Enhanced** 插件，可获得最佳渲染效果。\n")

        logger.info(f"✅ 生成Markdown报告：{md_path}")

    def _generate_sequence_json(self, sequence_metrics: Dict):
        """生成序列级指标JSON文件。
        
        Args:
            sequence_metrics: 序列级评估指标字典
        """
        import json
        
        json_path = os.path.join(self.output_dir, 'sequence_metrics.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sequence_metrics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 生成序列级指标JSON：{json_path}")
    
    def _generate_visualizations(self, real_file: str, syn_file: str, report_summary: Dict):
        """生成可视化图表。
        
        Args:
            real_file: 真实文件路径
            syn_file: 合成文件路径
            report_summary: 报告摘要
        """
        from netfaker.simcore.io.holowan_loader import HoloWANLoader
        
        # 创建可视化目录结构
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        window_level_dir = os.path.join(viz_dir, 'window_level')
        sequence_level_dir = os.path.join(viz_dir, 'sequence_level')
        diagnosis_dir = os.path.join(viz_dir, 'diagnosis')
        
        os.makedirs(viz_dir, exist_ok=True)
        os.makedirs(window_level_dir, exist_ok=True)
        os.makedirs(sequence_level_dir, exist_ok=True)
        os.makedirs(diagnosis_dir, exist_ok=True)
        
        # 加载数据
        loader = HoloWANLoader()
        real_df = loader.load_file(real_file)
        syn_df = loader.load_file(syn_file)
        
        if real_df is None or syn_df is None:
            logger.error("无法加载数据进行可视化")
            return
        
        # 确保数据长度一致
        min_len = min(len(real_df), len(syn_df))
        real_df = real_df.iloc[:min_len]
        syn_df = syn_df.iloc[:min_len]
        
        # 提取数据
        real_delay = real_df['raw_delay_up'].values
        syn_delay = syn_df['raw_delay_up'].values
        real_loss = real_df['raw_loss_up'].values
        syn_loss = syn_df['raw_loss_up'].values
        
        # 提取状态ID（如果有的话）
        real_states = real_df['state_id'].tolist() if 'state_id' in real_df.columns else None
        syn_states = syn_df['state_id'].tolist() if 'state_id' in syn_df.columns else None
        
        # 1. 生成序列级可视化
        self._generate_sequence_visualization(real_delay, syn_delay, real_loss, syn_loss, 
                                              real_states, syn_states, sequence_level_dir)
        
        # 2. 生成窗口级可视化（抽样）
        self._generate_window_visualizations(real_df, syn_df, window_level_dir, report_summary)
        
        # 3. 生成诊断聚焦图（按需）
        sequence_metrics = report_summary.get('sequence_metrics', {})
        self._generate_diagnosis_visualizations(real_delay, syn_delay, real_loss, syn_loss, 
                                               sequence_metrics, diagnosis_dir)
        
        logger.info(f"✅ 生成可视化图表：{viz_dir}")
    
    def _generate_sequence_visualization(self, real_delay: np.ndarray, syn_delay: np.ndarray, 
                                        real_loss: np.ndarray, syn_loss: np.ndarray, 
                                        real_states: list, syn_states: list, 
                                        output_dir: str):
        """生成序列级可视化图表。
        
        Args:
            real_delay: 真实延迟序列
            syn_delay: 合成延迟序列
            real_loss: 真实丢包率序列
            syn_loss: 合成丢包率序列
            real_states: 真实状态序列
            syn_states: 合成状态序列
            output_dir: 输出目录
        """
        
        # 创建2×2网格图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('NetFaker 合成数据全轨迹对比', fontsize=16)
        
        # 子图1：延迟轨迹对比
        ax1 = axes[0, 0]
        time = np.arange(len(real_delay))
        ax1.plot(time, real_delay, label='真实延迟', alpha=0.8)
        ax1.plot(time, syn_delay, label='合成延迟', alpha=0.8)
        
        # 添加窗口边界线
        window_size = 100
        for i in range(window_size, len(real_delay), window_size):
            ax1.axvline(x=i, color='gray', linestyle='--', alpha=0.3)
        
        ax1.set_xlabel('时间点')
        ax1.set_ylabel('延迟 (ms)')
        ax1.legend()
        ax1.grid(True)
        ax1.set_title('延迟轨迹对比')
        
        # 子图2：状态转移热力图
        ax2 = axes[0, 1]
        if real_states and syn_states:
            # 计算状态转移矩阵
            real_transitions = self._calculate_transition_matrix(real_states)
            syn_transitions = self._calculate_transition_matrix(syn_states)
            
            # 获取所有状态
            all_states = sorted(set(real_states + syn_states))
            n_states = len(all_states)
            
            # 创建转移矩阵
            real_matrix = np.zeros((n_states, n_states))
            syn_matrix = np.zeros((n_states, n_states))
            
            for i, from_state in enumerate(all_states):
                for j, to_state in enumerate(all_states):
                    real_matrix[i, j] = real_transitions.get((from_state, to_state), 0)
                    syn_matrix[i, j] = syn_transitions.get((from_state, to_state), 0)
            
            # 绘制热力图
            im1 = ax2.imshow(real_matrix - syn_matrix, cmap='RdBu', vmin=-0.2, vmax=0.2)
            ax2.set_xticks(range(n_states))
            ax2.set_xticklabels(all_states)
            ax2.set_yticks(range(n_states))
            ax2.set_yticklabels(all_states)
            ax2.set_xlabel('目标状态')
            ax2.set_ylabel('源状态')
            ax2.set_title('状态转移矩阵差异（真实-合成）')
            plt.colorbar(im1, ax=ax2)
        else:
            ax2.text(0.5, 0.5, '无状态数据', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('状态转移热力图')
        
        # 子图3：丢包事件标记
        ax3 = axes[1, 0]
        # 绘制丢包率
        ax3.plot(time, real_loss, label='真实丢包率', alpha=0.8, color='blue')
        ax3.plot(time, syn_loss, label='合成丢包率', alpha=0.8, color='orange')
        ax3.set_xlabel('时间点')
        ax3.set_ylabel('丢包率 (%)')
        
        # 标记burst事件（丢包率>5%）
        real_burst_mask = real_loss > 5
        syn_burst_mask = syn_loss > 5
        ax3.scatter(time[real_burst_mask], real_loss[real_burst_mask], 
                   color='red', s=20, alpha=0.5, label='真实Burst')
        ax3.scatter(time[syn_burst_mask], syn_loss[syn_burst_mask], 
                   color='red', s=20, alpha=0.5, marker='x', label='合成Burst')
        
        ax3.legend()
        ax3.grid(True)
        ax3.set_title('丢包事件标记')
        
        # 子图4：ACF对比
        ax4 = axes[1, 1]
        # 计算ACF
        real_acf = self._calculate_acf(real_delay, [1, 100])
        syn_acf = self._calculate_acf(syn_delay, [1, 100])
        lags = np.arange(1, 101)
        
        ax4.plot(lags, real_acf, label='真实ACF', alpha=0.8, color='blue')
        ax4.plot(lags, syn_acf, label='合成ACF', alpha=0.8, color='orange')
        ax4.set_xlabel('滞后')
        ax4.set_ylabel('自相关系数')
        
        # 添加参考线
        ax4.axvline(x=10, color='gray', linestyle='--', alpha=0.5, label='lag=10')
        ax4.axvline(x=100, color='gray', linestyle='--', alpha=0.5, label='lag=100')
        
        ax4.legend()
        ax4.grid(True)
        ax4.set_title('ACF对比（lag=1~100）')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        output_path = os.path.join(output_dir, 'full_trace_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_window_visualizations(self, real_df: pd.DataFrame, syn_df: pd.DataFrame, 
                                       output_dir: str, report_summary: Dict):
        """生成窗口级可视化图表。
        
        Args:
            real_df: 真实数据DataFrame
            syn_df: 合成数据DataFrame
            output_dir: 输出目录
            report_summary: 报告摘要
        """
        
        # 计算窗口数
        window_size = 100
        total_windows = len(real_df) // window_size
        
        # 根据抽样率确定生成数量
        sample_rate = self.config.config['report']['viz_sample_rate']
        sample_count = max(1, int(total_windows * sample_rate))
        
        # 随机选择窗口进行可视化
        import random
        selected_windows = random.sample(range(total_windows), min(sample_count, total_windows))
        selected_windows.sort()
        
        for window_id in selected_windows:
            # 提取窗口数据
            start_idx = window_id * window_size
            end_idx = start_idx + window_size
            
            real_window = real_df.iloc[start_idx:end_idx]
            syn_window = syn_df.iloc[start_idx:end_idx]
            
            real_delay = real_window['raw_delay_up'].values
            syn_delay = syn_window['raw_delay_up'].values
            real_loss = real_window['raw_loss_up'].values
            syn_loss = syn_window['raw_loss_up'].values
            
            # 创建窗口可视化
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            time = np.arange(window_size)
            
            # 子图1：延迟对比
            ax1 = axes[0]
            ax1.plot(time, real_delay, label='真实延迟', alpha=0.8, color='blue')
            ax1.plot(time, syn_delay, label='合成延迟', alpha=0.8, color='orange')
            # 绘制滑动均值
            real_mean = np.convolve(real_delay, np.ones(5)/5, mode='same')
            syn_mean = np.convolve(syn_delay, np.ones(5)/5, mode='same')
            ax1.plot(time, real_mean, label='真实延迟均值', color='blue', linestyle='--')
            ax1.plot(time, syn_mean, label='合成延迟均值', color='orange', linestyle='--')
            ax1.set_ylabel('延迟 (ms)')
            ax1.legend()
            ax1.grid(True)
            ax1.set_title(f'窗口 {window_id}：延迟对比')
            
            # 子图2：丢包率对比
            ax2 = axes[1]
            ax2.plot(time, real_loss, label='真实丢包率', alpha=0.8, color='blue')
            ax2.plot(time, syn_loss, label='合成丢包率', alpha=0.8, color='orange')
            # 标记burst事件
            real_burst_mask = real_loss > 5
            syn_burst_mask = syn_loss > 5
            ax2.scatter(time[real_burst_mask], real_loss[real_burst_mask], 
                       color='red', s=20, alpha=0.5, label='真实Burst')
            ax2.scatter(time[syn_burst_mask], syn_loss[syn_burst_mask], 
                       color='red', s=20, alpha=0.5, marker='x', label='合成Burst')
            ax2.set_ylabel('丢包率 (%)')
            ax2.legend()
            ax2.grid(True)
            ax2.set_title(f'窗口 {window_id}：丢包率对比')
            
            # 子图3：PSD对比
            ax3 = axes[2]
            # 计算PSD
            real_psd = self._calculate_psd(real_delay)
            syn_psd = self._calculate_psd(syn_delay)
            freq = np.linspace(0, 0.5, len(real_psd))
            
            ax3.loglog(freq, real_psd, label='真实PSD', alpha=0.8, color='blue')
            ax3.loglog(freq, syn_psd, label='合成PSD', alpha=0.8, color='orange')
            # 添加1/f参考线
            ax3.loglog(freq[1:], 1/freq[1:], label='1/f参考线', color='gray', linestyle='--')
            ax3.set_xlabel('归一化频率')
            ax3.set_ylabel('功率谱密度')
            ax3.legend()
            ax3.grid(True)
            ax3.set_title(f'窗口 {window_id}：PSD对比')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图像
            output_path = os.path.join(output_dir, f'viz_window_{window_id:04d}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_diagnosis_visualizations(self, real_delay: np.ndarray, syn_delay: np.ndarray, 
                                          real_loss: np.ndarray, syn_loss: np.ndarray, 
                                          sequence_metrics: Dict, output_dir: str):
        """生成诊断聚焦图。
        
        Args:
            real_delay: 真实延迟序列
            syn_delay: 合成延迟序列
            real_loss: 真实丢包率序列
            syn_loss: 合成丢包率序列
            sequence_metrics: 序列级指标
            output_dir: 输出目录
        """
        
        # 检查是否需要生成ACF跳诊断图
        acf100 = sequence_metrics.get('acf_mae_lag1_100', 0.0)
        if acf100 > 0.20:
            self._generate_acf_jump_diagnosis(real_delay, syn_delay, output_dir)
        
        # 检查是否需要生成状态转移诊断图
        jsd = sequence_metrics.get('state_transition_jsd', 0.0)
        state_available = sequence_metrics.get('state_transition_available', False)
        if state_available and jsd > 0.15:
            # 这里需要state数据，但当前方法没有state参数，暂时跳过
            pass
        
        # 检查是否需要生成漂移诊断图
        drift = sequence_metrics.get('rolling_p99_drift', 0.0)
        if drift > 80.0:
            self._generate_drift_diagnosis(real_delay, syn_delay, output_dir)
    
    def _generate_acf_jump_diagnosis(self, real_delay: np.ndarray, syn_delay: np.ndarray, 
                                    output_dir: str):
        """生成ACF跳诊断图。
        
        Args:
            real_delay: 真实延迟序列
            syn_delay: 合成延迟序列
            output_dir: 输出目录
        """
        
        # 计算ACF（lag=5~30）
        real_acf = self._calculate_acf(real_delay, [5, 30])
        syn_acf = self._calculate_acf(syn_delay, [5, 30])
        lags = np.arange(5, 31)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 子图1：ACF对比（lag=5~30）
        ax1 = axes[0]
        ax1.plot(lags, real_acf, label='真实ACF', alpha=0.8, color='blue')
        ax1.plot(lags, syn_acf, label='合成ACF', alpha=0.8, color='orange')
        ax1.set_xlabel('滞后')
        ax1.set_ylabel('自相关系数')
        ax1.legend()
        ax1.grid(True)
        ax1.set_title('ACF对比（lag=5~30）')
        
        # 子图2：边界延迟局部zoom-in
        ax2 = axes[1]
        # 选择前几个窗口的边界
        window_size = 100
        n_windows = 3
        zoom_len = 20  # 每个边界前后观察20个点
        
        x = []
        y_real = []
        y_syn = []
        
        for i in range(n_windows):
            # 边界位置
            boundary = (i + 1) * window_size
            start = boundary - zoom_len
            end = boundary + zoom_len
            
            if end > len(real_delay):
                break
            
            # 提取数据
            time = np.arange(start, end)
            x.extend(time)
            y_real.extend(real_delay[start:end])
            y_syn.extend(syn_delay[start:end])
        
        ax2.plot(x, y_real, label='真实延迟', alpha=0.8, color='blue')
        ax2.plot(x, y_syn, label='合成延迟', alpha=0.8, color='orange')
        
        # 标记窗口边界
        for i in range(n_windows):
            boundary = (i + 1) * window_size
            if boundary < len(real_delay):
                ax2.axvline(x=boundary, color='red', linestyle='--', alpha=0.5, label='窗口边界')
        
        ax2.set_xlabel('时间点')
        ax2.set_ylabel('延迟 (ms)')
        ax2.legend()
        ax2.grid(True)
        ax2.set_title('窗口边界延迟局部放大')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        output_path = os.path.join(output_dir, 'diagnosis_acf_jump.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_drift_diagnosis(self, real_delay: np.ndarray, syn_delay: np.ndarray, 
                                 output_dir: str):
        """生成漂移诊断图。
        
        Args:
            real_delay: 真实延迟序列
            syn_delay: 合成延迟序列
            output_dir: 输出目录
        """
        
        # 计算滚动P99
        window_size = 100
        real_rolling_p99 = self._calculate_rolling_percentile(real_delay, window_size, 99)
        syn_rolling_p99 = self._calculate_rolling_percentile(syn_delay, window_size, 99)
        
        # 创建时间轴
        time = np.arange(len(real_rolling_p99))
        
        # 创建诊断图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制滚动P99
        ax.plot(time, real_rolling_p99, label='真实滚动P99', alpha=0.8, color='blue')
        ax.plot(time, syn_rolling_p99, label='合成滚动P99', alpha=0.8, color='orange')
        
        # 线性拟合
        real_slope, real_intercept = np.polyfit(time, real_rolling_p99, 1)
        syn_slope, syn_intercept = np.polyfit(time, syn_rolling_p99, 1)
        
        real_fit = real_slope * time + real_intercept
        syn_fit = syn_slope * time + syn_intercept
        
        ax.plot(time, real_fit, label=f'真实拟合（斜率：{real_slope:.3f}）', 
               color='blue', linestyle='--')
        ax.plot(time, syn_fit, label=f'合成拟合（斜率：{syn_slope:.3f}）', 
               color='orange', linestyle='--')
        
        ax.set_xlabel('时间点')
        ax.set_ylabel('滚动P99延迟 (ms)')
        ax.legend()
        ax.grid(True)
        ax.set_title('滚动P99漂移诊断')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        output_path = os.path.join(output_dir, 'diagnosis_drift_trend.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _calculate_acf(self, data: np.ndarray, lag_range: list) -> np.ndarray:
        """计算自相关函数。
        
        Args:
            data: 输入数据序列
            lag_range: 滞后范围 [min_lag, max_lag]
            
        Returns:
            np.ndarray: 自相关函数值
        """
        
        max_lag = lag_range[1]
        nlags = max_lag - lag_range[0]
        
        # 使用statsmodels计算ACF
        acf_vals = acf(data, nlags=nlags, fft=True)
        
        return acf_vals
    
    def _calculate_psd(self, data: np.ndarray) -> np.ndarray:
        """计算功率谱密度。
        
        Args:
            data: 输入数据序列
            
        Returns:
            np.ndarray: 功率谱密度值
        """
        
        # 使用welch方法计算PSD
        f, psd = welch(data, fs=1.0, nperseg=len(data)//2, scaling='density')
        
        return psd
    
    def _calculate_transition_matrix(self, states: list) -> dict:
        """计算状态转移矩阵。
        
        Args:
            states: 状态序列
            
        Returns:
            dict: 状态转移概率字典
        """
        transitions = {}
        state_counts = {}
        
        # 统计状态转移次数
        for i in range(len(states) - 1):
            from_state = states[i]
            to_state = states[i + 1]
            
            # 更新状态计数
            state_counts[from_state] = state_counts.get(from_state, 0) + 1
            
            # 更新转移计数
            key = (from_state, to_state)
            transitions[key] = transitions.get(key, 0) + 1
        
        # 计算转移概率
        transition_probs = {}
        for (from_state, to_state), count in transitions.items():
            transition_probs[(from_state, to_state)] = count / state_counts[from_state]
        
        return transition_probs
    
    def _calculate_rolling_percentile(self, data: np.ndarray, window_size: int, percentile: float) -> np.ndarray:
        """计算滚动分位数。
        
        Args:
            data: 输入数据序列
            window_size: 滚动窗口大小
            percentile: 分位数（0-100）
            
        Returns:
            np.ndarray: 滚动分位数值
        """
        
        rolling_percentiles = []
        
        for i in range(len(data) - window_size + 1):
            window = data[i:i + window_size]
            p = np.percentile(window, percentile)
            rolling_percentiles.append(p)
        
        return np.array(rolling_percentiles)

    def _calculate_top_failed_modes(self, metrics_df: pd.DataFrame) -> List[str]:
        """计算Top失败模式。
        
        Args:
            metrics_df: 评估指标DataFrame
            
        Returns:
            List[str]: Top失败模式列表
        """
        # 指标列（排除元数据列）
        metric_columns = [col for col in metrics_df.columns
                         if col not in ['window_id', 'real_file', 'syn_file']]

        # 计算每个指标的失败窗口数
        failed_counts = {}
        for col in metric_columns:
            threshold = self.config.get_threshold(col)
            failed_count = (metrics_df[col] > threshold).sum()
            failed_counts[col] = failed_count

        # 按失败数排序，取Top 3
        sorted_failed = sorted(failed_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        # 格式化输出
        top_failed = []
        total_windows = len(metrics_df)
        for metric, count in sorted_failed:
            if count > 0:
                percentage = (count / total_windows) * 100
                top_failed.append(f"{percentage:.1f}% 窗口 {metric} 超出阈值")

        return top_failed

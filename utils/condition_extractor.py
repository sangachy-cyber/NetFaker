import numpy as np


def extract_continuous_condition(window, mode="anomaly"):
    """提取连续条件特征"""
    log_d1, p1 = window[:, 0], window[:, 1]
    d1 = np.exp(log_d1) - 1.0

    if mode == "light":
        median_delay = np.median(d1)
        burst_freq = np.sum(p1 > 0.05) / len(p1)
        return np.array([median_delay, burst_freq, 0.0])  # 补0对齐维度
    else:  # anomaly
        p95_delay = np.percentile(d1, 95)
        high_delay_ratio = np.mean(d1 > 100)

        burst_mask = (p1 > 0.05).astype(int)
        if burst_mask.sum() == 0:
            mean_burst_len = 0.0
        else:
            bursts = []
            curr = 0
            for b in burst_mask:
                if b:
                    curr += 1
                else:
                    if curr:
                        bursts.append(curr)
                        curr = 0
            if curr:
                bursts.append(curr)
            mean_burst_len = float(np.mean(bursts)) if bursts else 0.0

        return np.array([p95_delay, high_delay_ratio, mean_burst_len])


def assign_hybrid_label(window, scaler):
    """分配混合离散标签"""
    d1_log, p1, d2_log, p2 = window.T
    d1_orig = np.exp(scaler.inverse_transform(d1_log.reshape(-1, 1))).flatten() - 1
    d2_orig = np.exp(scaler.inverse_transform(d2_log.reshape(-1, 1))).flatten() - 1

    max_delay = max(d1_orig.max(), d2_orig.max())
    mean_loss = (p1.mean() + p2.mean()) / 2

    if max_delay < 100 and mean_loss < 0.01:
        return "S0"
    else:
        # 区分 light vs anomaly
        p95_d1 = np.percentile(d1_orig, 95)
        p95_d2 = np.percentile(d2_orig, 95)
        avg_p95 = (p95_d1 + p95_d2) / 2

        if avg_p95 < 80:  # 阈值可调
            return "S1_light"
        else:
            return "S1_anomaly"

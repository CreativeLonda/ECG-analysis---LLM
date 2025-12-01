import numpy as np
from scipy.signal import find_peaks

def extract_features(filtered_ecg, rr_intervals, r_peaks, p_peaks, q_peaks, s_peaks, t_peaks, fs=500):
    # ---------- HEART RATE ----------
    heart_rate = 60 / rr_intervals if len(rr_intervals) > 0 else [0]
    hr_avg = np.mean(heart_rate)

    # ---------- HRV METRICS ----------
    SDNN = np.std(rr_intervals) * 1000 if len(rr_intervals) > 1 else 0
    RMSSD = np.sqrt(np.mean(np.diff(rr_intervals)**2)) * 1000 if len(rr_intervals) > 2 else 0
    pNN50 = np.sum(np.abs(np.diff(rr_intervals)) > 0.05) / len(rr_intervals) * 100 if len(rr_intervals) > 1 else 0

    # ---------- QRS WIDTH ----------
    qrs_widths = []
    ecg_signal = np.array(filtered_ecg)

    for r in r_peaks:
        left = max(r - int(0.12*fs), 0)
        right = min(r + int(0.12*fs), len(ecg_signal)-1)
        seg = ecg_signal[left:right]

        d = np.diff(seg)
        d = np.convolve(d, np.ones(5)/5, mode='same')  # smoothing

        maxd = np.max(np.abs(d))
        if maxd < 1e-6:
            qrs_widths.append(0.08)
            continue

        thresh = 0.1 * maxd
        onset_idx = None
        offset_idx = None

        for i in range(len(d)-3):
            if abs(d[i]) > thresh and abs(d[i+1]) > thresh:
                onset_idx = i
                break

        for i in range(len(d)-3, 0, -1):
            if abs(d[i]) > thresh and abs(d[i-1]) > thresh:
                offset_idx = i
                break

        if onset_idx is None or offset_idx is None or offset_idx <= onset_idx:
            qrs_widths.append(0.08)
            continue

        width = (offset_idx - onset_idx) / fs
        qrs_widths.append(width)

    qrs_widths = [float(x) for x in qrs_widths]

    # ---------- ST INTERVAL ----------
    st_intervals = []
    for i in range(len(r_peaks)-1):
        qrs_end = r_peaks[i] + int(qrs_widths[i] * fs)
        start = qrs_end
        end = min(qrs_end + int(0.25*fs), r_peaks[i+1] - 1)

        seg = ecg_signal[start:end]

        if len(seg) < 20:
            st_intervals.append(np.nan)
            continue

        peaks, _ = find_peaks(seg, distance=10)
        if len(peaks) == 0:
            st_intervals.append(np.nan)
            continue

        t_peak = start + peaks[np.argmax(seg[peaks])]
        t_onset = t_peak - int(0.02 * fs)
        st = (t_onset - qrs_end) / fs
        if st < 0:
            st = 0.0

        st_intervals.append(st)

    st_intervals = [float(x) for x in st_intervals]

    # ---------- RETURN FEATURES ----------
    return {
        "Heart_Rate_BPM": float(np.mean(heart_rate)),
        "RR_intervals_s": rr_intervals.tolist(),
        "SDNN_ms": float(SDNN),
        "RMSSD_ms": float(RMSSD),
        "pNN50_percent": float(pNN50),
        "QRS_widths_s": qrs_widths,
        "ST_intervals_s": st_intervals,
        "R_peak_count": len(r_peaks),
        "P_peak_count": len(p_peaks),
        "Q_peak_count": len(q_peaks),
        "S_peak_count": len(s_peaks),
        "T_peak_count": len(t_peaks),
    }

def summarize_features(features: dict) -> dict:
    """
    Summarize ECG features for LLM, including irregularity / arrhythmia detection.
    """
    rr_intervals = np.array(features["RR_intervals_s"])
    qrs_widths = np.array(features["QRS_widths_s"])
    st_intervals = np.array(features["ST_intervals_s"])

    # Basic stats
    rr_mean = float(np.mean(rr_intervals)) if len(rr_intervals) > 0 else 0.0
    rr_std = float(np.std(rr_intervals)) if len(rr_intervals) > 1 else 0.0
    qrs_mean = float(np.mean(qrs_widths)) if len(qrs_widths) > 0 else 0.0
    st_mean = float(np.nanmean(st_intervals)) if len(st_intervals) > 0 else 0.0

    # ---------------- ARRHYTHMIA DETECTION ----------------
    # 1. RR interval irregularity
    rr_diff = np.abs(np.diff(rr_intervals))
    irregularity = np.sum(rr_diff > 0.1) / len(rr_intervals) if len(rr_intervals) > 1 else 0

    # 2. QRS abnormality (example: > 0.12 s)
    qrs_abnormal = np.any(qrs_widths > 0.12)

    # 3. Combine criteria
    arrhythmia_flag = (rr_std > 0.05) or (irregularity > 0.05) or qrs_abnormal

    # -------------------------------------------------------

    return {
        "Heart_Rate_BPM": round(float(features["Heart_Rate_BPM"]), 1),
        "RR_mean_s": round(rr_mean, 3),
        "RR_std_s": round(rr_std, 3),
        "SDNN_ms": round(float(features["SDNN_ms"]), 1),
        "RMSSD_ms": round(float(features["RMSSD_ms"]), 1),
        "pNN50_percent": round(float(features["pNN50_percent"]), 1),
        "QRS_mean_s": round(qrs_mean, 3),
        "ST_mean_s": round(st_mean, 3),
        "R_peak_count": int(features["R_peak_count"]),
        "Arrhythmia_detected": arrhythmia_flag,
        "RR_irregularity_flag": irregularity > 0.05,
        "QRS_abnormal_flag": qrs_abnormal
    }
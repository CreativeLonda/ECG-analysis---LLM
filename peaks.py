import numpy as np
from scipy.signal import find_peaks

def extract_all_peaks(filtered_ecg, fs=500):
    # -------- R PEAKS ----------
    diff_ecg = np.diff(filtered_ecg)
    squared = diff_ecg ** 2
    window = int(0.1 * fs)
    ma = np.convolve(squared, np.ones(window)/window, mode="same")

    height = 0.5 * np.max(ma)
    distance = int(0.25 * fs)

    rough_peaks, _ = find_peaks(ma, height=height, distance=distance)

    r_peaks = []
    search = int(0.08 * fs)

    for p in rough_peaks:
        start = max(p - search, 0)
        end = min(p + search, len(filtered_ecg))
        real_r = start + np.argmax(filtered_ecg[start:end])
        r_peaks.append(real_r)

    r_peaks = np.array(r_peaks)

    # RR intervals (seconds)
    rr_intervals = np.diff(r_peaks) / fs

    # -------- P/Q/S/T PEAKS ----------
    p_peaks = []
    q_peaks = []
    s_peaks = []
    t_peaks = []

    p_w = int(0.2 * fs)
    q_w = int(0.05 * fs)
    s_w = int(0.05 * fs)
    t_w = int(0.4 * fs)

    for r in r_peaks:
        if r - p_w > 0:
            p_peaks.append(r - p_w + np.argmax(filtered_ecg[r-p_w:r]))

        if r - q_w > 0:
            q_peaks.append(r - q_w + np.argmin(filtered_ecg[r-q_w:r]))

        if r + s_w < len(filtered_ecg):
            s_peaks.append(r + np.argmin(filtered_ecg[r:r+s_w]))

        if r + t_w < len(filtered_ecg):
            t_peaks.append(r + s_w + np.argmax(filtered_ecg[r+s_w:r+t_w]))

    return (
        np.array(r_peaks),
        np.array(p_peaks),
        np.array(q_peaks),
        np.array(s_peaks),
        np.array(t_peaks),
        rr_intervals
    )

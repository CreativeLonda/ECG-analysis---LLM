# main.py
import numpy as np
import os

from filter import bandpass_filter
from peaks import extract_all_peaks
from features import extract_features, summarize_features
from llm_report import generate_ecg_report_with_ollama
from pdf_report import create_ecg_report

fs = 500  # ECG sampling rate

def main():
    print("📥 Loading ECG.dat ...")
    ecg_signal = np.loadtxt("ECG.dat")

    print("🔧 Filtering ECG ...")
    filtered = bandpass_filter(ecg_signal, fs=fs)

    print("📌 Detecting peaks ...")
    r_peaks, p_peaks, q_peaks, s_peaks, t_peaks, rr_intervals = extract_all_peaks(filtered, fs=fs)

    print("📊 Extracting raw features ...")
    raw_features = extract_features(
        filtered_ecg=filtered,
        rr_intervals=rr_intervals,
        r_peaks=r_peaks,
        p_peaks=p_peaks,
        q_peaks=q_peaks,
        s_peaks=s_peaks,
        t_peaks=t_peaks,
        fs=fs
    )

    print("📝 Summarizing features ...")
    summary_features = summarize_features(raw_features)

    print("🧠 Generating AI interpretation (Ollama) ...")
    interpretation = generate_ecg_report_with_ollama(summary_features)
    print("\n--- ECG INTERPRETATION ---\n")
    print(interpretation)

    print("📄 Creating PDF report ...")
    plot_path = "ecg_plot.png"
    if not os.path.exists(plot_path):
        plot_path = None

    create_ecg_report(raw_features, interpretation, plot_path=plot_path)

    print("\n✅ DONE! ECG_Report.pdf generated successfully.\n")


if __name__ == "__main__":
    main()

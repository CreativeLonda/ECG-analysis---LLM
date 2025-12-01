import subprocess

def generate_ecg_report_with_ollama(features):
    """
    Generate a human-readable ECG report using Ollama LLaMA 3.2 model.

    Args:
        features (dict): Dictionary containing extracted ECG features, e.g.,
            {
                "heart_rate": 120,
                "rr_mean": 0.5,
                "rr_std": 0.04,
                "sdnn": 35,
                "rmssd": 30,
                "pnn50": 12,
                "qrs_mean": 0.1,
                "st_mean": 0.05,
                "r_peak_count": 800,
                "arrhythmia": "Yes"
            }

    Returns:
        str: Generated ECG interpretation.
    """

    # Construct the prompt dynamically from features
    prompt = f"""
    Patient ECG Analysis:
    - Heart Rate: {features.get('Heart_Rate_BPM', 'N/A')} bpm
    - RR Interval Mean: {features.get('RR_mean_s', 'N/A')} s
    - RR Interval Std: {features.get('RR_std_s', 'N/A')} s
    - SDNN: {features.get('SDNN_ms', 'N/A')} ms
    - RMSSD: {features.get('RMSSD_ms', 'N/A')} ms
    - pNN50: {features.get('pNN50_percent', 'N/A')}%
    - QRS Duration Mean: {features.get('QRS_mean_s', 'N/A')} s
    - ST Segment Mean: {features.get('ST_mean_s', 'N/A')} s
    - R Peak Count: {features.get('R_peak_count', 'N/A')}
    - Arrhythmia Detected: {features.get('Arrhythmia_detected', 'Unknown')}
    """

    # Run Ollama subprocess
    try:
        result = subprocess.run(
        ["ollama", "run", "llama3.2"],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True
    )

        return result.stdout.strip()

    except subprocess.CalledProcessError as e:
        return f"Error running Ollama: {e.stderr}"

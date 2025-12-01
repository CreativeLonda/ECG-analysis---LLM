from scipy.signal import butter, filtfilt

fs = 500
lowcut = 0.5
highcut = 40.0

def bandpass_filter(data, lowcut=lowcut, highcut=highcut, fs=fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

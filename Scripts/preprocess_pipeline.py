import os
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt
from wfdb.processing import xqrs_detect
from typing import Tuple, Dict, List


# --------------------------------------------------------
# 1. ECG Filtering Utilities
# --------------------------------------------------------

def butter_bandpass(lowcut=0.5, highcut=40.0, fs=360, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    return butter(order, [low, high], btype='band')


def apply_bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=360, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, signal)


# --------------------------------------------------------
# 2. R-Peak Detection (WFDB xqrs)
# --------------------------------------------------------

def detect_r_peaks(signal, fs=360):

    try:
        # Newer WFDB versions
        from wfdb.processing import xqrs_detect
        r_peaks = xqrs_detect(sig=signal, fs=fs)
        return r_peaks

    except Exception:
        pass

    try:
        # Older & stable (most common) WFDB versions
        from wfdb.processing import gqrs_detect
        r_peaks = gqrs_detect(sig=signal, fs=fs)
        return r_peaks

    except Exception as e:
        print("R-peak detection failed:", e)
        return []



# --------------------------------------------------------
# 3. Beat Segmentation
# --------------------------------------------------------

def segment_beats(signal, r_peaks, window_before=100, window_after=150):
    """
    Extracts fixed-size heartbeat windows.
    Total window = 100 samples before + 150 samples after = 250 samples.
    """
    beats = []
    valid_peaks = []
    length = len(signal)

    for r in r_peaks:
        start = r - window_before
        end = r + window_after

        if start >= 0 and end <= length:
            beats.append(signal[start:end])
            valid_peaks.append(r)

    return np.array(beats), valid_peaks


# --------------------------------------------------------
# 4. Label Extraction + Mapping
# --------------------------------------------------------

# AAMI standard class mapping
AAMI_MAP = {
    'N': 'N',   # Normal
    'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',  # LBBB, RBBB, etc.
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',  # Supraventricular
    'V': 'V', 'E': 'V',                      # Ventricular
    'F': 'F',                                # Fusion
}


def map_label(original_label):
    """
    Converts MIT-BIH beat annotation to AAMI heartbeat class.
    Unknown labels are ignored.
    """
    return AAMI_MAP.get(original_label, None)


def get_labels_for_beats(annotation_symbols, valid_r_peaks):
    labels = []

    for r in valid_r_peaks:
        # Each annotation corresponds exactly to r-peaks in MIT-BIH
        labels.append(annotation_symbols[valid_r_peaks.index(r)])

    # Map to AAMI classes
    mapped = [map_label(lb) for lb in labels]

    # Filter None
    final_labels = [m for m in mapped if m is not None]

    return final_labels


# --------------------------------------------------------
# 5. Normalization
# --------------------------------------------------------

def zscore_normalize(beats: np.ndarray):
    mean = np.mean(beats, axis=1, keepdims=True)
    std = np.std(beats, axis=1, keepdims=True) + 1e-8
    return (beats - mean) / std


# --------------------------------------------------------
# 6. Complete Preprocessing Class
# --------------------------------------------------------

class ECGPreprocessor:
    def __init__(self, data_path: str, save_path: str, fs=360):
        self.data_path = data_path
        self.save_path = save_path
        self.fs = fs

        os.makedirs(save_path, exist_ok=True)

    def load_record(self, record_name: str):
        record_path = os.path.join(self.data_path, record_name)
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')

        signal = record.p_signal[:, 0]  # MLII lead
        ann_symbols = annotation.symbol

        return signal, ann_symbols

    def preprocess_record(self, record_name: str):
        print(f"\nProcessing Record: {record_name}")

        signal, labels = self.load_record(record_name)

        # 1. Filter
        filtered = apply_bandpass_filter(signal)

        # 2. R peak detection
        r_peaks = detect_r_peaks(filtered, fs=self.fs)

        # 3. Segment beats
        beats, valid_r = segment_beats(filtered, r_peaks)

        # 4. Map labels
        labels = get_labels_for_beats(labels, valid_r)

        # 5. Normalize
        beats = zscore_normalize(beats)

        # Save
        save_file = os.path.join(self.save_path, f"{record_name}.npz")
        np.savez(save_file, beats=beats, labels=labels)

        print(f"Saved preprocessed file: {save_file}")

        return beats, labels

    def preprocess_all_records(self, record_list: List[str]):
        for rec in record_list:
            try:
                self.preprocess_record(rec)
            except Exception as e:
                print(f"Error processing {rec}: {e}")


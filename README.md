# ECG Arrhythmia Classification Using 1D CNNs with Attention

## Project Overview
Electrocardiograms (ECGs) capture the electrical activity of the heart, and analyzing these signals allows clinicians to detect arrhythmias—abnormal heart rhythms that may indicate underlying medical conditions.

This project focuses on building a deep learning pipeline for **automatic arrhythmia detection** using:
- **(1D CNNs)** for feature extraction from raw ECG waveforms.
- **Attention mechanisms** to highlight the most relevant segments of the heartbeat, improving interpretability and potentially classification performance.

The main objective is to classify individual heartbeats into clinically relevant arrhythmia categories.

---

## Dataset: MIT-BIH Arrhythmia Database
We use the **MIT-BIH Arrhythmia Database**, a widely recognized benchmark dataset in ECG research.

### Key Characteristics
- **Number of Records:** 48 patient recordings (`100`, `101`, ... `234`)
- **ECG Leads:** Two channels per record
  - **MLII** (primarily used for heartbeat annotations)
  - **V5**
- **Sampling Frequency:** 360 Hz
- **Files Included:**
  - `.dat` – raw ECG signal data
  - `.hea` – metadata (sampling rate, patient info, lead labels)
  - `.atr` – annotations including heartbeat locations and types
  - `.xws` – additional data (typically unused)

Although the dataset contains only 48 files, each recording includes thousands of annotated beats. In total, the dataset provides **~100,000 labeled heartbeats**, making it large enough for deep learning applications.

REST INFO WILL FOLLOW UP 

---




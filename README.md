# ECG Arrhythmia Classification Using 1D CNNs with Attention

## Project Overview

Electrocardiograms (ECGs) record the electrical activity of the heart
and are critical for diagnosing cardiac arrhythmias---abnormal rhythms
that may indicate serious health conditions. Manual interpretation of
ECG signals is time-consuming and requires clinical expertise,
motivating the use of automated classification methods.

This project presents a comparative machine learning and deep learning
framework for automatic ECG arrhythmia classification, with a primary
focus on an attention-based 1D Convolutional Neural Network (CNN). The
system classifies individual ECG heartbeats into clinically relevant
arrhythmia categories and provides model interpretability through
attention visualisation.

In addition to offline model training and evaluation, the project
includes an interactive Streamlit web application that allows users to
upload ECG data and obtain real-time classification results.

## Models Implemented

-   Random Forest -- classical baseline using handcrafted statistical
    features\
-   LSTM -- sequence-based deep learning model for temporal
    dependencies\
-   1D CNN with Attention -- primary model for high accuracy and
    interpretability

## Dataset: MIT-BIH Arrhythmia Database

The project uses the MIT-BIH Arrhythmia Database, accessed via
PhysioNet.

### Dataset Characteristics

-   Records: 48 annotated ECG recordings (100--234)\
-   Sampling Frequency: 360 Hz\
-   ECG Leads: MLII (primary), V5\
-   Files Used:
    -   .dat -- raw ECG signals\
    -   .hea -- metadata\
    -   .atr -- beat annotations

Approximately 100,000 annotated heartbeats are available for supervised
learning.

## Data Processing Pipeline

1.  Noise removal and baseline wander correction\
2.  R-peak detection and heartbeat segmentation\
3.  Normalisation and AAMI class labelling\
4.  Saving processed beats for efficient training

## Model Training and Evaluation

-   Stratified train--validation split\
-   Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC\
-   Attention visualisation for interpretability

## Streamlit Application

The project includes an interactive Streamlit web application (app.py).

### Features

-   Upload ECG data in CSV format\
-   Automatic preprocessing\
-   Arrhythmia classification using the trained model\
-   Visualisation of predictions and attention overlays

### Run the Application

    streamlit run app.py

## Project Structure

    ├── Data/
    │   ├── Raw/
    │   └── Processed/
    ├── Models/
    ├── Notebooks/
    ├── Result/
    ├── Scripts/
    ├── app.py
    ├── requirements.txt
    ├── 101ecg_sample.csv
    ├── 102ecg_sample.csv
    ├── 103ecg_sample.csv
    ├── 104ecg_sample.csv
    ├── 105ecg_sample.csv
    ├── ecg_sample.csv
    └── README.md
    

## Key Contributions

-   Comparative evaluation of classical and deep learning models\
-   Attention-based ECG interpretability\
-   End-to-end deployment using Streamlit

## Disclaimer

This project is for academic and research purposes only and is not
intended for clinical diagnosis.

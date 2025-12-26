import wfdb
import pandas as pd

records = ["101", "102", "103", "104", "105"]

for rec in records:
    record = wfdb.rdrecord(f".../Data/Raw/{rec}", pn_dir="mitdb")
    signal = record.p_signal[:, 0]  # Lead II

    pd.DataFrame(signal, columns=["ecg"]).to_csv(
        f"{rec}ecg_sample.csv",
        index=False
    )

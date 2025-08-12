from tqdm import tqdm
from glob import glob
from scipy.io import loadmat
from braindecode.preprocessing.preprocess import exponential_moving_standardize
import os
import sqlite3
import traceback
import mne
import numpy as np
import warnings
from moabb.datasets import Zhou2016

warnings.filterwarnings("ignore")

WORKING_DIR = os.getcwd()
BASE_PATH = f"{WORKING_DIR}/../Datasets/zhou"
os.makedirs(BASE_PATH, exist_ok=True)
SAVE_PATH = f"{WORKING_DIR}/../Datasets/zhou_preprocessed"
os.makedirs(SAVE_PATH, exist_ok=True)
DB_PATH = f"{WORKING_DIR}/databases/zhou.db"

origin_ival = [125, 1125]
bandpass_filter = [0.0, 38.0]

con = sqlite3.connect(DB_PATH)
print("LOG >>> Successfully connected to the database")

cur = con.cursor()
print("LOG >>> Successfully created Table")

cur.execute(
    """CREATE TABLE MetaData(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Sub Integer,
    Path text
    );"""
)

file_idx = 0
dataset = Zhou2016()
session = dataset.get_data()
print("LOG >>> Successfully loaded the dataset")

try:
    for target_subject in tqdm(range(4)):
        print(f"\nLOG >>> Subject: {target_subject} \n")
        for sess in [0, 1, 2]:
            for r in [0, 1]:
                print(f"\nLOG >>> Session: {sess}, Run: {r} \n")

                raw = session[target_subject + 1][str(sess)][str(r)]
                events, annot = mne.events_from_annotations(raw)

                raw.load_data()
                raw.filter(
                    bandpass_filter[0],
                    bandpass_filter[1],
                    fir_design="firwin",
                    verbose=False,
                )

                raw.info["bads"] = ["VEOU", "VEOL", "Fp1", "Fp2", "O1", "O2", "Oz"]

                picks = mne.pick_types(
                    raw.info, meg=False, eeg=True, eog=False, stim=False, exclude="bads"
                )

                tmin, tmax = (
                    origin_ival[0] / raw.info["sfreq"],
                    (origin_ival[1] - 1) / raw.info["sfreq"],
                )

                event_id = {
                    "feet": annot["feet"],
                    "left_hand": annot["left_hand"],
                    "right_hand": annot["right_hand"],
                }

                epochs = mne.Epochs(
                    raw,
                    events,
                    event_id,
                    tmin,
                    tmax,
                    proj=True,
                    picks=picks,
                    baseline=None,
                    preload=True,
                )

                epochs_data = epochs.get_data()

                preprocessed_data = []
                for epoch in epochs_data:
                    normalized_data = exponential_moving_standardize(
                        epoch, init_block_size=int(epochs.info["sfreq"] * 4)
                    )
                    preprocessed_data.append(normalized_data)
                preprocessed_data = np.stack(preprocessed_data)
                preprocessed_data = (preprocessed_data - preprocessed_data.min()) / (
                    preprocessed_data.max() - preprocessed_data.min()
                )

                label_list = epochs.events[:, -1] - epochs.events[:, -1].min()
                print("Shape :", preprocessed_data.shape)
                for i in range(preprocessed_data.shape[0]):
                    save_filename = os.path.join(SAVE_PATH, f"{file_idx:06d}.npz")
                    np.savez(
                        save_filename, data=preprocessed_data[i], label=label_list[i]
                    )

                    cur.execute(
                        "INSERT INTO MetaData (Sub, Path) Values(:Sub, :Path)",
                        {
                            "Sub": target_subject,
                            "Path": save_filename,
                        },
                    )

                    file_idx += 1
except Exception as e:
    print(target_subject)
    print(e)
    print(traceback.format_exc())

con.commit()
con.close()

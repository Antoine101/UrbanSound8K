# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:04:04 2022

@author: APU
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as transforms

dataset_path = "dataset"

metadata = pd.read_csv(os.path.join(dataset_path,"UrbanSound8K.csv"))
unique_classes = metadata["class"].unique()

target_sample_rate = 22050
target_length = 4
n_samples = target_length * target_sample_rate 
n_fft = 1048

for index, row in metadata.iterrows():
    print(index+1)
    label = row["class"]
    file_name = row["slice_file_name"]
    fold_nbr = row["fold"]
    file_path = os.path.join(dataset_path, f"fold{fold_nbr}", file_name)
    signal, sr = torchaudio.load(file_path)
    # Mix down if necessary
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    # Resample if necessary
    if sr != target_sample_rate:
        resample_transform = torchaudio.transforms.Resample(sr, target_sample_rate) 
        signal = resample_transform(signal)
    # Cut if necessary
    if signal.shape[1] > n_samples:
        signal = signal[:, :n_samples]
    signal_length = signal.shape[1]
    # Right-pad if necessary
    if signal_length < n_samples:
        num_missing_samples = n_samples - signal_length
        last_dim_padding = (0, num_missing_samples)
        signal = nn.functional.pad(signal, last_dim_padding)
    # Compute spectrogram
    spectrogram_transform = transforms.Spectrogram(
                                                    n_fft = n_fft,
                                                    pad = 0,
                                                    window_fn = torch.hann_window,
                                                    power = 2,
                                                    normalized = True,
                                                    wkwargs = None,
                                                    center = False,
                                                    pad_mode = "reflect",
                                                    onesided = True,
                                                    return_complex = False
                                                    )      
    spectrogram = spectrogram_transform(signal)
    # Convert amplitude to dB
    db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")
    spectrogram_db = db_transform(spectrogram)
    spectrogram_db = torch.squeeze(spectrogram_db)
    plt.figure()
    plt.imshow(spectrogram_db, extent=[0, target_length, 0, target_sample_rate/2], origin="lower", aspect="auto")
    plt.title(f"{file_name} - Sample rate={target_sample_rate} - n_fft={n_fft}.png", fontweight="bold")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    save_path = os.path.join("features", "spectrograms", label)
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
    save_name = f"{file_name} - Sample rate={target_sample_rate} - n_fft={n_fft}.png"
    plt.savefig(fname=os.path.join(save_path, save_name))
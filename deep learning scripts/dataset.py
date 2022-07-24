import os
import math
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as transforms
from torch_audiomentations import Compose, Gain, PitchShift, Shift


class UrbanSound8KDataset(Dataset):
    
    def __init__(self, dataset_path, train, validation_fold, feature_name, feature_processing_parameters, signal_augmentation, feature_augmentation, augmentation_parameters):
        self.dataset_path = dataset_path
        metadata = pd.read_csv(os.path.join(dataset_path, "UrbanSound8K.csv"))
        if train:
            self.metadata = metadata[metadata["fold"] != validation_fold].reset_index(drop=True)
        else:
            self.metadata = metadata[metadata["fold"] == validation_fold].reset_index(drop=True)
        self.signal_augmentation = signal_augmentation
        self.feature_augmentation = feature_augmentation
        self.feature_name = feature_name
        self.feature_processing_parameters = feature_processing_parameters
        self.augmentation_parameters = augmentation_parameters
        

    def __len__(self):
        return len(self.metadata)

    
    def __getitem__(self, index):
        audio_name = self._get_event_audio_name(index)
        class_id = torch.tensor(self._get_event_class_id(index), dtype=torch.long)
        signal, sr = self._get_event_signal(index)
        signal = self._mix_down_if_necessary(signal)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        if self.signal_augmentation:
            signal = self._signal_augmentation(signal)
        if self.feature_name == "spectrogram":
            feature = self._spectrogram_transform(signal)
            if self.feature_augmentation:
                feature = self._feature_augmentation(feature)
            feature = self._db_transform(feature)
        elif self.feature_name == "mel-spectrogram":
            feature = self._mel_spectrogram_transform(signal)
            if self.feature_augmentation:
                feature = self._feature_augmentation(feature)
            feature = self._db_transform(feature)
        elif self.feature_name == "mfcc":
            feature = self._mfcc_transform(signal)
            if self.feature_augmentation:
                feature = self._feature_augmentation(feature)            
        return index, audio_name, class_id, feature

    
    def _get_event_class_id(self, index):
        return self.metadata.iloc[index]["classID"]

    
    def _get_event_audio_name(self, index):
        return self.metadata.iloc[index]["slice_file_name"]

    
    def _get_event_signal(self, index):
        event_fold = f"fold{self.metadata.iloc[index]['fold']}"
        event_filename = self.metadata.iloc[index]["slice_file_name"]
        audio_path = os.path.join(self.dataset_path, event_fold, event_filename)
        signal, sr = torchaudio.load(audio_path, normalize=True)
        return signal, sr

    
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
        

    def _resample_if_necessary(self, signal, sr):
        if sr != self.feature_processing_parameters["target_sample_rate"]:
            resample_transform = transforms.Resample(sr, self.feature_processing_parameters["target_sample_rate"])
            signal = resample_transform(signal)
        return signal

    
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.feature_processing_parameters["n_samples"]:
            signal = signal[:, :self.feature_processing_parameters["n_samples"]]
        return signal

        
    def _right_pad_if_necessary(self, signal):
        signal_length = signal.shape[1]
        if signal_length < self.feature_processing_parameters["n_samples"]:
            num_missing_samples = self.feature_processing_parameters["n_samples"] - signal_length
            last_dim_padding = (0, num_missing_samples)
            signal = nn.functional.pad(signal, last_dim_padding)
        return signal

    
    def _signal_augmentation(self, signal):
        signal = signal.unsqueeze(0)
        signal_transforms = Compose([
                                    Gain(
                                        min_gain_in_db=self.augmentation_parameters["min_gain_in_db"], 
                                        max_gain_in_db=self.augmentation_parameters["max_gain_in_db"], 
                                        p=self.augmentation_parameters["p_gain"]
                                        ),
                                    PitchShift(
                                        min_transpose_semitones=self.augmentation_parameters["min_transpose_semitones"], 
                                        max_transpose_semitones=self.augmentation_parameters["max_transpose_semitones"], 
                                        p=self.augmentation_parameters["p_pitch_shift"], 
                                        sample_rate=self.feature_processing_parameters["target_sample_rate"], 
                                        target_rate=self.feature_processing_parameters["target_sample_rate"]
                                        ),
                                    Shift(
                                        min_shift=self.augmentation_parameters["min_shift"], 
                                        max_shift=self.augmentation_parameters["max_shift"], 
                                        p=self.augmentation_parameters["p_shift"], 
                                        sample_rate=self.feature_processing_parameters["target_sample_rate"], 
                                        target_rate=self.feature_processing_parameters["target_sample_rate"]
                                        )
                                    ], 
                                    p=self.augmentation_parameters["p_compose"], 
                                    shuffle=False
                                    )
        signal = signal_transforms(samples=signal, sample_rate=self.feature_processing_parameters["target_sample_rate"])
        signal = signal.squeeze(0)
        return signal

    
    def _spectrogram_transform(self, signal):
        spectrogram_transform = transforms.Spectrogram(
                                                        n_fft = self.feature_processing_parameters["n_fft"],
                                                        win_length = self.feature_processing_parameters["n_fft"],
                                                        hop_length = self.feature_processing_parameters["n_fft"] // self.feature_processing_parameters["hop_denominator"],
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
        return spectrogram

    
    def _mel_spectrogram_transform(self, signal):
        mel_spectrogram_transform = transforms.MelSpectrogram(
                                                        sample_rate = self.feature_processing_parameters["target_sample_rate"],
                                                        n_fft = self.feature_processing_parameters["n_fft"],
                                                        n_mels = self.feature_processing_parameters["n_mels"],
                                                        window_fn = torch.hann_window,
                                                        power = 2,
                                                        normalized = True,
                                                        wkwargs = None,
                                                        center = True,
                                                        pad_mode = "reflect",
                                                        onesided = True,
                                                        norm = None,
                                                        mel_scale = "htk"
                                                        )
        mel_spectrogram = mel_spectrogram_transform(signal)
        return mel_spectrogram


    def _mfcc_transform(self, signal):
        mfcc_transform = transforms.MFCC(
                                        sample_rate = self.feature_processing_parameters["target_sample_rate"],
                                        n_mfcc = self.feature_processing_parameters["n_mfcc"],
                                        dct_type = 2,
                                        norm = "ortho",
                                        log_mels = False 
                                        )
        mfcc = mfcc_transform(signal)
        return mfcc

    
    def _db_transform(self, spectrogram):
        db_transform = transforms.AmplitudeToDB(stype="power")
        spectrogram_db = db_transform(spectrogram)
        return spectrogram_db


    def _feature_augmentation(self, feature):
        feature_height = feature.size(dim=1)
        feature_width = feature.size(dim=2)
        freq_mask_len = math.ceil(self.augmentation_parameters["percentage_freq_mask_len"]*feature_height)
        time_mask_len = math.ceil(self.augmentation_parameters["percentage_time_mask_len"]*feature_width)
        frequency_masking = transforms.FrequencyMasking(freq_mask_param=freq_mask_len)
        time_masking = transforms.TimeMasking(time_mask_param=time_mask_len, p=self.augmentation_parameters["p_time_masking"])
        feature = frequency_masking(feature)
        feature = time_masking(feature)
        return feature
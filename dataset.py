import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as transforms


class UrbanSound8KDataset(Dataset):
    
    def __init__(self, dataset_path, transforms_params):
        if torch.cuda.is_available():
            self.accelerator = "cuda"
        else:
            self.accelerator = "cpu"
        self.dataset_path = dataset_path
        self.metadata = pd.read_csv(os.path.join(dataset_path, "UrbanSound8K.csv"))
        self.n_folds = max(self.metadata["fold"])
        self.n_classes = len(self.metadata["class"].unique())
        self.classes_map = pd.Series(self.metadata["class"].values,index=self.metadata["classID"]).sort_index().to_dict()
        self.target_sample_rate = transforms_params["target_sample_rate"]
        self.target_length = transforms_params["target_length"]
        self.n_samples = transforms_params["n_samples"]
        self.n_fft = transforms_params["n_fft"]
        self.hop_denominator = transforms_params["hop_denominator"]
        self.n_mels = transforms_params["n_mels"]
        self.n_mfcc = transforms_params["n_mfcc"]
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        audio_name = self._get_event_audio_name(index)
        class_id = torch.tensor(self._get_event_class_id(index), dtype=torch.long)
        signal, sr = self._get_event_signal(index)
        #signal = signal.to(self.accelerator)
        signal = self._mix_down_if_necessary(signal)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        #spectrogram = self._spectrogram_transform(signal)
        #feature = self._db_transform(spectrogram)
        mel_spectrogram = self._mel_spectrogram_transform(signal)
        feature = self._db_transform(mel_spectrogram)
        #feature = self._mfcc_transform(signal)
        feature = self._augmentation(feature)
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
        if sr != self.target_sample_rate:
            resample_transform = transforms.Resample(sr, self.target_sample_rate)
            #resample_transform = resample_transform.to(self.accelerator)
            signal = resample_transform(signal)
        return signal
    
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.n_samples:
            signal = signal[:, :self.n_samples]
        return signal
        
    def _right_pad_if_necessary(self, signal):
        signal_length = signal.shape[1]
        if signal_length < self.n_samples:
            num_missing_samples = self.n_samples - signal_length
            last_dim_padding = (0, num_missing_samples)
            signal = nn.functional.pad(signal, last_dim_padding)
        return signal
    
    def _spectrogram_transform(self, signal):
        spectrogram_transform = transforms.Spectrogram(
                                                        n_fft = self.n_fft,
                                                        win_length = self.n_fft,
                                                        hop_length = self.n_fft // self.hop_denominator,
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
        #spectrogram_transform = spectrogram_transform.to(self.accelerator)
        spectrogram = spectrogram_transform(signal)
        return spectrogram
    
    def _mel_spectrogram_transform(self, signal):
        mel_spectrogram_transform = transforms.MelSpectrogram(
                                                        sample_rate = self.target_sample_rate,
                                                        n_fft = self.n_fft,
                                                        n_mels = self.n_mels,
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
        #mel_spectrogram_transform = mel_spectrogram_transform.to(self.accelerator)
        mel_spectrogram = mel_spectrogram_transform(signal)
        return mel_spectrogram

    def _mfcc_transform(self, signal):
        mfcc_transform = transforms.MFCC(
                                                    sample_rate = self.target_sample_rate,
                                                    n_mfcc = self.n_mfcc,
                                                    dct_type = 2,
                                                    norm = "ortho",
                                                    log_mels = False 
                                                    )
        #mfcc_transform = mfcc_transform.to(self.accelerator)
        mfcc = mfcc_transform(signal)
        return mfcc
    
    def _db_transform(self, spectrogram):
        db_transform = transforms.AmplitudeToDB(stype="power")
        #db_transform = db_transform.to(self.accelerator)
        spectrogram_db = db_transform(spectrogram)
        return spectrogram_db

    def _augmentation(self, feature):
        frequency_masking = transforms.FrequencyMasking(freq_mask_param=80)
        time_masking = transforms.TimeMasking(time_mask_param=80, p=1.0)
        #frequency_masking = frequency_masking.to(self.accelerator)
        #time_masking = time_masking.to(self.accelerator)
        feature = frequency_masking(feature)
        feature = time_masking(feature)
        return feature

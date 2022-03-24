import os
import math
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torchmetrics import Accuracy, Precision, Recall, F1, ConfusionMatrix
import torchaudio
import torchaudio.transforms as transforms
from captum.attr import IntegratedGradients
import IPython
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


########################################################################
# PARAMETERS
########################################################################

# Dataset path
dataset_path = "dataset"

# Device to operate on
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Used device: {device}")

# Transforms parameters
target_sample_rate = 22050
target_length = 4
n_samples = target_length * target_sample_rate
n_fft = 512
n_mels = 64

transforms_params = {
    "target_sample_rate": target_sample_rate,
    "target_length": target_length,
    "n_samples": n_samples,
    "n_fft": n_fft,
    "n_mels": n_mels
}


########################################################################
# IMPORT OF THE METADATA FILE
########################################################################
metadata = pd.read_csv("dataset/UrbanSound8K.csv")


########################################################################
# FITLERING ON CLASSES (OPTIONAL)
########################################################################
classes_filter = ["dog_bark", "siren"]

metadata = metadata[metadata["class"].isin(classes_filter)].reset_index(drop=True)
metadata
for id, classe in enumerate(classes_filter):
    metadata.loc[metadata["class"]==classe, "classID"] = id
metadata


########################################################################
# CREATION OF THE DATASET CLASS
########################################################################
class UrbanSound8K(Dataset):
    
    def __init__(self, metadata, dataset_path, transforms_params, device):
        self.device = device
        self.metadata = metadata
        self.dataset_path = dataset_path
        self.n_folds = max(metadata["fold"])
        self.n_classes = len(metadata["class"].unique())
        self.classes_map = pd.Series(metadata["class"].values,index=metadata["classID"]).sort_index().to_dict()
        self.target_sample_rate = transforms_params["target_sample_rate"]
        self.target_length = transforms_params["target_length"]
        self.n_samples = transforms_params["n_samples"]
        self.n_fft = transforms_params["n_fft"]
        self.n_mels = transforms_params["n_mels"]
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        audio_name = self._get_event_audio_name(index)
        class_id = torch.tensor(self._get_event_class_id(index), dtype=torch.long)
        signal, sr = self._get_event_signal(index)
        signal = signal.to(self.device)
        signal = self._mix_down_if_necessary(signal)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        #spectrogram = self._spectrogram_transform(signal)
        #spectrogram_db = self._db_transform(spectrogram)
        mel_spectrogram = self._mel_spectrogram_transform(signal)
        mel_spectrogram_db = self._db_transform(mel_spectrogram)
        return mel_spectrogram_db, class_id, audio_name
    
    def _get_event_class_id(self, index):
        return self.metadata.iloc[index]["classID"]
    
    def _get_event_audio_name(self, index):
        return self.metadata.iloc[index]["slice_file_name"]
    
    def _get_event_signal(self, index):
        event_fold = f"fold{self.metadata.iloc[index]['fold']}"
        event_filename = self.metadata.iloc[index]["slice_file_name"]
        audio_path = os.path.join(self.dataset_path, event_fold, event_filename)
        signal, sr = torchaudio.load(audio_path)
        return signal, sr
    
    def _mix_down_if_necessary(self, signal):
        # If signal has multiple channels, mix down to mono
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
        
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resample_transform = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            resample_transform = resample_transform.to(self.device)
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
                                                        hop_length = self.n_fft // 2,
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
        spectrogram_transform = spectrogram_transform.to(self.device)
        spectrogram = spectrogram_transform(signal)
        return spectrogram
    
    def _mel_spectrogram_transform(self, signal):
        mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
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
        mel_spectrogram_transform = mel_spectrogram_transform.to(self.device)
        mel_spectrogram = mel_spectrogram_transform(signal)
        return mel_spectrogram
    
    def _db_transform(self, spectrogram):
        db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")
        db_transform = db_transform.to(self.device)
        spectrogram_db = db_transform(spectrogram)
        return spectrogram_db


########################################################################
# INSTANTIATION OF THE DATASET
########################################################################
dataset = UrbanSound8K(
    metadata=metadata,
    dataset_path=dataset_path, 
    transforms_params=transforms_params,
    device=device
)


########################################################################
# CREATION OF THE LIGHTNING PIPELINE
########################################################################
class Pipeline(pl.LightningModule):
    def __init__(self, out_dim, classes_map, learning_rate, batch_size):
        super().__init__()
        
        # Save hyperparameters to the checkpoint
        self.save_hyperparameters()        

        # Definition of the model
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.3)
            )
        self.flatten = nn.Sequential(nn.Flatten()) 
        self.dense = nn.Sequential(
            nn.Linear(128 * 16 * 86, 512),
            nn.Linear(512, 256),
            nn.Linear(256, out_dim)
            )
        
        # Instantiation of the metrics
        self.accuracy = Accuracy(num_classes=len(classes_map), average="weighted")
        self.recall = Recall(num_classes=len(classes_map), average="weighted")
        self.f1_score = F1(num_classes=len(classes_map), average="weighted")
        self.confmat = ConfusionMatrix(num_classes=len(classes_map))           
        
        # Instantiation of the classes map
        self.classes_map = classes_map
        
        # Instantiation of the number of classes
        self.n_classes = len(classes_map)
        
        # Instatiation of the learning rate
        self.learning_rate = learning_rate
        
        # Instantiation of the batch size (needed to avoid batch size inference error caused by text returned by the dataset)
        self.batch_size = batch_size
        
    def configure_optimizers(self):
        
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
            
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "validation_loss",
                        "frequency": 1
                        }
                }
    
    def forward(self, x):

        x = self.conv_block(x)
        x = self.flatten(x)
        logits = self.dense(x)
        
        return logits 
    
        
    def training_step(self, train_batch, batch_idx): 
        
        # Unpack the training batch
        inputs, targets, _ = train_batch
        # Pass the inputs to the model to get the logits
        logits = self(inputs)
        # Compute the loss
        loss = F.cross_entropy(logits, targets)
        # Get the probabilities for each class by applying softmax
        probs = F.softmax(logits, dim=1)
        # Get the prediction for each batch sample
        _, preds = torch.max(probs, 1)
        # Compute the accuracy
        accuracy = self.accuracy(logits, targets)
        # Log the loss
        self.log("training_loss", loss, on_step=True, on_epoch=True, batch_size=self.batch_size)
        
        return {"inputs":inputs, "targets":targets, "predictions":preds, "loss":loss}
    
    
    def training_epoch_end(self, outputs):

        # Log weights and biases for all layers of the model
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params,self.current_epoch)
        
        # Only after the first training epoch, log one of the training inputs as a figure and log the model graph
        if self.current_epoch == 0:
            input_sample = outputs[0]["inputs"][0]
            input_sample_target = outputs[0]["targets"][0].item()
            input_sample_class = self.classes_map[input_sample_target]
            fig = plt.figure(figsize=(20,20))
            ax = fig.add_subplot(111)
            ax.imshow(torch.squeeze(input_sample).cpu(), cmap="viridis", origin="lower", aspect="auto")
            ax.set_title(f"Class: {input_sample_class}")
            ax.set_xlabel("Time Frames")
            ax.set_ylabel("Mel Bands")
            self.logger.experiment.add_figure(f"Training sample input", fig)
            input_sample = torch.unsqueeze(input_sample, 3)
            input_sample = torch.permute(input_sample, (0,3,1,2))
            self.logger.experiment.add_graph(self, input_sample)

            
    def validation_step(self, validation_batch, batch_idx):
        
        # Unpack the validation batch
        inputs, targets, audios_name = validation_batch
        # Pass the inputs to the model to get the logits
        logits = self(inputs)
        # Compute the loss and log it for early stopping monitoring
        loss = F.cross_entropy(logits, targets)
        # Get the probabilities for each class by applying softmax
        probs = F.softmax(logits, dim=1)
        # Get the prediction for each batch sample
        _, preds = torch.max(probs, 1)
        # Compute the accuracy for this batch
        accuracy = self.accuracy(preds, targets)
        # Log the loss and the accuracy
        self.log("validation_loss", loss, on_step=True, on_epoch=True, batch_size=self.batch_size)
        self.log("validation_accuracy", accuracy, on_step=True, on_epoch=True, batch_size=self.batch_size)
        
        return {"inputs":inputs, "targets":targets, "predictions":preds, "loss":loss, "audios_name":audios_name}
    
    
    def validation_epoch_end(self, outputs):
        
        # Concatenate the predictions of all batches
        preds = torch.cat([output["predictions"] for output in outputs])
        # Concatenate the targets of all batches
        targets = torch.cat([output["targets"] for output in outputs])
        # Concatenate the audios name of all batches
        audios_name_tuples_list = [output["audios_name"] for output in outputs]
        audios_name = [audio_name for audios_name_tuples in audios_name_tuples_list for audio_name in audios_name_tuples]
        
        for i in range(len(outputs)):
            self.logger.experiment.add_text("Predictions on validation set", f"Audio: {audios_name[i]} - Class: {targets[i]} - Predicted: {preds[i]}")
        
        # Compute the confusion matrix, turn it into a DataFrame, generate the plot and log it
        cm = self.confmat(preds, targets)
        cm = cm.cpu()
        
        for class_id in range(self.n_classes):
                precision = cm[class_id, class_id] / torch.sum(cm[:,class_id])
                precision = round(precision.item()*100,1)
                self.log(f"validation_precision/{class_id}", precision)
                recall = cm[class_id, class_id] / torch.sum(cm[class_id,:])
                recall = round(recall.item()*100,1)
                self.log(f"validation_recall/{class_id}", recall)
      
        df_cm = pd.DataFrame(cm.numpy(), index = range(self.n_classes), columns=range(self.n_classes))
        plt.figure()
        fig = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.yticks(rotation=0)
        self.logger.experiment.add_figure("Confusion matrix", fig, self.current_epoch)
        
    def on_save_checkpoint(self, checkpoint):
        # Get the state_dict from self.model to get rid of the "model." prefix
        checkpoint["state_dict"] = self.state_dict()


########################################################################
# MODEL TRAINING AND VALIDATION
########################################################################

# Batch size
batch_size = 10
# Number of epochs
n_epochs = 30
# Learning rate
learning_rate = 2e-4

for i in range(1,dataset.n_folds+1):
    
    print(f"========== Cross-validation {i} on {dataset.n_folds} ==========")
    
    # Get the train and validation sets
    train_metadata = dataset.metadata.drop(dataset.metadata[dataset.metadata["fold"]==i].index)
    validation_metadata = dataset.metadata[dataset.metadata["fold"]==i]
    train_indices = train_metadata.index 
    train_sampler = SubsetRandomSampler(train_indices)
    
    # Create the train and validation dataloaders
    train_dataloader = DataLoader(
                            dataset, 
                            batch_size=batch_size, 
                            sampler=train_sampler,
                            num_workers=12
                            )
    
    validation_dataloader = DataLoader(
                            dataset, 
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=12
                            )
    
    # Instantiate the pipeline
    pipeline = Pipeline(out_dim=dataset.n_classes, classes_map=dataset.classes_map, learning_rate=learning_rate, batch_size=batch_size)
    
    # Instantiate the logger
    run_name = f"{dataset.n_folds} folds CV - Val. on fold {i}"
    tensorboard_logger = TensorBoardLogger(save_dir="logs", name=run_name)
    
    # Instantiate a learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Instantiate early stopping based on epoch validation loss
    early_stopping = EarlyStopping("validation_loss", patience=20, verbose=True)
    
    # Instantiate a checkpoint callback
    checkpoint = ModelCheckpoint(
                            dirpath=f"./checkpoints/{dataset.n_folds} folds cross-validation - Validation on fold {i}",
                            filename="{epoch}-{validation_loss:.2f}",
                            verbose=True,
                            monitor="validation_loss",
                            save_last = False,
                            save_top_k=1,      
                            mode="min",
                            save_weights_only=True
                            )
    
    # Instantiate the trainer and train the model
    trainer = Trainer(
                    gpus=-1,
                    max_epochs=n_epochs, 
                    logger=tensorboard_logger,
                    log_every_n_steps = 1,
                    callbacks=[early_stopping, lr_monitor, checkpoint]
                    )   
    
    trainer.fit(pipeline, train_dataloader, validation_dataloader)
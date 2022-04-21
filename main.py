import os
import pandas as pd
import warnings
import dataset
import datamodule
import pipeline
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


if __name__ == "__main__":

    # Filter harmless warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", ".*Your `val_dataloader` has `shuffle=True`.*")
    warnings.filterwarnings("ignore", ".*Checkpoint directory.*")

    # Device to operate on
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Device used: {torch.cuda.get_device_name(device=0)}")
    else:
        device = "cpu"
        print(f"Device used: CPU")

    # Set number of workers (for dataloaders)
    num_workers = int(os.cpu_count() / 3)
    print(f"Number of workers used: {num_workers}")

    # Set maximum number of epochs to train for
    max_epochs = 2
    print(f"Maximum number of epochs: {max_epochs}")

    # Set the batch size
    batch_size = 256 if device=="cuda" else 64
    print(f"Batch size: {batch_size}")

    # Set the initial learning rate
    learning_rate = 0.1
    print(f"Initial learning rate: {learning_rate}")    

    # Instantiate the dataset
    dataset_dir = "dataset"
    metadata = pd.read_csv(os.path.join(dataset_dir, "UrbanSound8K.csv"))
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
    ds = dataset.UrbanSound8KDataset(metadata, dataset_dir, transforms_params, device)
    dm = datamodule.UrbanSound8KDataModule(dataset_dir, batch_size, num_workers)

    # Get the train and validation sets
    train_metadata = dataset.metadata.drop(dataset.metadata[dataset.metadata["fold"]==i].index)
    train_indices = train_metadata.index.to_list() 
    train_sampler = SubsetRandomSampler(train_indices)
    validation_indices = dataset.metadata[dataset.metadata["fold"]==i].index.to_list()
    
    # Create the train and validation dataloaders
    train_dataloader = DataLoader(
                            ds, 
                            batch_size=batch_size, 
                            sampler=train_sampler,
                            num_workers=0
                            )
    
    validation_dataloader = DataLoader(
                            ds, 
                            sampler=validation_indices,
                            batch_size=batch_size,
                            num_workers=0
                            )

    # Instantiate the logger
    tensorboard_logger = TensorBoardLogger(save_dir="logs")

    # Instantiate early stopping based on epoch validation loss
    early_stopping = EarlyStopping("validation_loss", patience=20, verbose=True)

    # Instantiate a learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Instantiate a checkpoint callback
    checkpoint = ModelCheckpoint(
                            dirpath=f"./checkpoints/",
                            filename="{epoch}-{validation_loss:.2f}",
                            verbose=True,
                            monitor="validation_loss",
                            save_last = False,
                            save_top_k=1,      
                            mode="min",
                            save_weights_only=True
                            )

    # Instantiate the trainer
    trainer = Trainer(
                    gpus=-1,
                    max_epochs=max_epochs, 
                    logger=tensorboard_logger,
                    log_every_n_steps = 1,
                    callbacks=[early_stopping, lr_monitor, checkpoint]
                    ) 

    # Instantiate the pipeline
    pipeline = pipeline.CIFAR100ResNet(learning_rate=learning_rate, batch_size=batch_size)  
    
    # Fit the trainer on the training set
    trainer.fit(pipeline, train_dataloader, validation_dataloader)
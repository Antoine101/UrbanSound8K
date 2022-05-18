import os
from argparse import ArgumentParser
import utils
import pandas as pd
import warnings
import dataset
import datamodule
import lightning_module
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

    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default="gpu", help="Type of accelerator: 'gpu', 'cpu', 'auto'")
    parser.add_argument("--devices", default="auto", help="Number of devices (GPUs or CPU cores) to use: integer starting from 1 or 'auto'")
    parser.add_argument("--workers", type=int, default=4, help="Number of CPU cores to use as as workers for the dataloarders: integer starting from 1 to maximum number of cores on this machine")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum number of epochs to run for")
    parser.add_argument("--bs", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    args = parser.parse_args()

    # Print summary of selected arguments and adjust them if needed
    args = utils.args_interpreter(args)

    # Instantiate the dataset
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

    ds = dataset.UrbanSound8KDataset("dataset", transforms_params, args.device)
    #####dm = datamodule.UrbanSound8KDataModule(args.bs, args.workers)

    for i in range(1, dataset.n_folds+1):
    
        print(f"========== Cross-validation {i} on {dataset.n_folds} ==========")

        # Get the train and validation sets
        train_metadata = dataset.metadata.drop(dataset.metadata[dataset.metadata["fold"]==i].index)
        train_indices = train_metadata.index.to_list() 
        train_sampler = SubsetRandomSampler(train_indices)
        validation_indices = dataset.metadata[dataset.metadata["fold"]==i].index.to_list()
    
        # Create the train and validation dataloaders
        train_dataloader = DataLoader(
                                        ds, 
                                        batch_size=args.bs, 
                                        sampler=train_sampler,
                                        num_workers=args.workers
                                    )
    
        validation_dataloader = DataLoader(
                                            ds, 
                                            sampler=validation_indices,
                                            batch_size=args.bs,
                                            num_workers=args.workers
                                        )

        ###dm = datamodule.UrbanSound8KDataModule(dataset_dir="dataset", validation_fold=i, batch_size=args.bs, num_workers=args.workers)

        # Instantiate the logger
        tensorboard_logger = TensorBoardLogger(save_dir="logs")

        # Instantiate early stopping based on epoch validation loss
        early_stopping = EarlyStopping("validation_loss", patience=20, verbose=True)

        # Instantiate a learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

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
                            accelerator=args.accelerator,
                            devices=args.devices,
                            max_epochs=args.epochs, 
                            logger=tensorboard_logger,
                            log_every_n_steps = 1,
                            callbacks=[early_stopping, lr_monitor, checkpoint]
                        ) 

    # Instantiate the pipeline
    lm = lightning_module.UrbanSound8KNet(learning_rate=args.lr, batch_size=args.bs)  
    
    # Fit the trainer on the training set
    trainer.fit(lm, train_dataloader, validation_dataloader)
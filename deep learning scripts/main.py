from datetime import datetime
from argparse import ArgumentParser
import utils
import warnings
import pandas as pd
from model import UrbanSound8KModel
import lightning_module
import lightning_data_module
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


if __name__ == "__main__":

    # Filtering of the harmless warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", ".*Your `val_dataloader` has `shuffle=True`.*")
    warnings.filterwarnings("ignore", ".*Checkpoint directory.*")

    # Parsing of the command line arguments
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default="auto", help="Type of accelerator: 'gpu', 'cpu', 'auto'")
    parser.add_argument("--devices", default="auto", help="Number of devices (GPUs or CPU cores) to use: integer starting from 1 or 'auto'")
    parser.add_argument("--workers", type=int, default=4, help="Number of CPU cores to use as as workers for the dataloarders: integer starting from 1 to maximum number of cores on this machine")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum number of epochs to run for")
    parser.add_argument("--bs", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
    args = parser.parse_args()

    # Printing of the selected arguments summary and adju
    # stments if needed
    args = utils.args_interpreter(args)

    # Loading of the dataset metadata to retrieve the number of classes and the {label_id: label_name} mapping dictionnary
    metadata = pd.read_csv("dataset/UrbanSound8K.csv")
    n_classes = len(metadata["class"].unique())
    classes_map = pd.Series(metadata["class"].values, index=metadata["classID"]).sort_index().to_dict()

    # Feature selection ("spectrogram", "mel-spectrogram" or "mfcc")
    feature_name = "mfcc"

    # Feature processing parameters
    target_sample_rate = 22050
    target_length = 4
    n_samples = target_length * target_sample_rate
    feature_processing_parameters = {
        "target_sample_rate": target_sample_rate,
        "target_length": target_length,
        "n_samples": n_samples,
        "n_fft": 1024,
        "hop_denominator": 2,
        "n_mels": 128,
        "n_mfcc": 40
    }

    # Calculation of the input height and width to pass to the model for adjustment of fc1 in_features
    input_height, input_width = utils.calculate_input_shape(feature_name, feature_processing_parameters)

    # Augmentation
    signal_augmentation = True
    feature_augmentation = True

    # Data augmentation parameters
    augmentation_parameters = {
        "min_gain_in_db":-15.0,
        "max_gain_in_db":15.0,
        "p_gain":0.5,
        "min_transpose_semitones":-1,
        "max_transpose_semitones":1,
        "p_pitch_shift":0.5,
        "min_shift":-0.25,
        "max_shift":0.25,
        "p_shift":0.5,
        "p_compose": 1.0,
        "percentage_freq_mask_len": 0.15,
        "percentage_time_mask_len": 0.15,
        "p_time_masking": 1.0
    }

    # Optimizer selection
    optimizer = "Adam"
    optimizer_parameters = {}

    # Learning rate scheduler selection
    lr_scheduler = "ReduceLROnPlateau"
    lr_scheduler_parameters = {
        "patience": 3
    }


    for i in range(1, 11):
        
        print("\n")
        print(f"========== Cross-validation {i} on {10} ==========")

        # Instiation of the lightning data module
        dm = lightning_data_module.UrbanSound8KDataModule(
                                                            dataset_path="dataset",
                                                            batch_size=args.bs, 
                                                            num_workers=args.workers, 
                                                            feature_processing_parameters=feature_processing_parameters, 
                                                            validation_fold=i, 
                                                            feature_name = feature_name,
                                                            signal_augmentation = signal_augmentation, 
                                                            feature_augmentation = feature_augmentation,
                                                            augmentation_parameters=augmentation_parameters,
                                                            to_gpus=False
                                                        )

        # Instantiation of the model
        model = UrbanSound8KModel(input_height=input_height, input_width=input_width)

        # Instantiation of the lightning module
        lm = lightning_module.UrbanSound8KModule(
                                                n_classes=n_classes, 
                                                classes_map=classes_map, 
                                                optimizer=optimizer,
                                                optimizer_parameters=optimizer_parameters,
                                                lr_scheduler=lr_scheduler,
                                                lr_scheduler_parameters=lr_scheduler_parameters,
                                                learning_rate=args.lr, 
                                                batch_size=args.bs, 
                                                model=model
                                                ) 

        # Instantiation of the logger
        timestamp = datetime.today().strftime("%Y-%m-%d - %Hh%Mm%Ss")
        tensorboard_logger = TensorBoardLogger(
                                                save_dir=".", 
                                                name="logs", 
                                                version=f"{timestamp} - Validation on fold {i}",
                                                log_graph=True,
                                                default_hp_metric=True,
                                                )

        # Instantiation of the early stopping callback
        early_stopping = EarlyStopping(
                                        monitor = "validation_loss",
                                        min_delta = 0.01,
                                        patience=6, 
                                        verbose=True,
                                        mode="min"
                                        )

        # Instantiation of the learning rate monitor callback
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        # Instantiation of the checkpoint callback
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

        # Instantiation of the trainer
        trainer = Trainer(
                            accelerator=args.accelerator,
                            devices=args.devices,
                            max_epochs=args.epochs, 
                            logger=tensorboard_logger,
                            log_every_n_steps = 1,
                            callbacks=[early_stopping, lr_monitor, checkpoint]
                        ) 
 

        # Fit the trainer on the training set
        trainer.fit(lm, dm)
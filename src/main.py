from argparse import ArgumentParser
import utils
import warnings
import dataset
import model
import lightning_module
from torch.utils.data import DataLoader, SubsetRandomSampler
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
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    args = parser.parse_args()

    # Printing of the selected arguments summary and adju
    # stments if needed
    args = utils.args_interpreter(args)

    # Audio pre-processing parameters
    target_sample_rate = 22050
    target_length = 4
    n_samples = target_length * target_sample_rate
    n_fft = 512
    hop_denominator = 2
    n_mels = 64
    n_mfcc = 40
    transforms_params = {
        "target_sample_rate": target_sample_rate,
        "target_length": target_length,
        "n_samples": n_samples,
        "n_fft": n_fft,
        "hop_denominator": hop_denominator,
        "n_mels": n_mels,
        "n_mfcc": n_mfcc
    }

    # Instantiation of the dataset
    ds = dataset.UrbanSound8KDataset(dataset_path="dataset", transforms_params=transforms_params)

    # Instantiation of the model
    model = model.UrbanSound8KModel(input_height=ds[0][3].size(1), input_width=ds[0][3].size(2), output_neurons=len(ds.classes_map))

    # Instantiation of the lightning module
    lm = lightning_module.UrbanSound8KModule(n_classes=ds.n_classes, classes_map=ds.classes_map, learning_rate=args.lr, batch_size=args.bs, model=model) 

    for i in range(1, ds.n_folds+1):
        
        print("\n")
        print(f"========== Cross-validation {i} on {ds.n_folds} ==========")

        # Determination of the train and validation sets indices
        train_metadata = ds.metadata.drop(ds.metadata[ds.metadata["fold"]==i].index)
        train_indices = train_metadata.index.to_list() 
        train_sampler = SubsetRandomSampler(train_indices)
        validation_indices = ds.metadata[ds.metadata["fold"]==i].index.to_list()
    
        # Creation of the train and validation dataloaders
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

        # Instantiation of the logger
        tensorboard_logger = TensorBoardLogger(save_dir=".")

        # Instantiation of the early stopping callback
        early_stopping = EarlyStopping("validation_loss", patience=20, verbose=True)

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
        trainer.fit(lm, train_dataloader, validation_dataloader)
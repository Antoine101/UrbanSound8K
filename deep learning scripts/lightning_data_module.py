import pytorch_lightning as pl
from dataset import UrbanSound8KDataset
from torch.utils.data import DataLoader

class UrbanSound8KDataModule(pl.LightningDataModule):

    def __init__(self, dataset_path, batch_size, num_workers, feature_name, feature_processing_parameters, validation_fold, signal_augmentation, feature_augmentation, augmentation_parameters):
        super().__init__()
        self.save_hyperparameters(ignore=["dataset_path"])
        self.prepare_data_per_node = True
        self.dataset_path


    def prepare_data(self) -> None:
        return super().prepare_data()
    

    def setup(self, stage=None):

        if stage == "fit" or stage is None:
            self.train_ds = UrbanSound8KDataset(
                                                dataset_path=self.dataset_path, 
                                                validation_fold=self.hparams.validation_fold, 
                                                feature_name=self.hparams.feature_name, 
                                                feature_processing_parameters=self.hparams.feature_processing_parameters, 
                                                train=True,
                                                signal_augmentation=self.hparams.signal_augmentation,
                                                feature_augmentation=self.hparams.feature_augmentation,
                                                augmentation_parameters=self.hparams.augmentation_parameters
                                                )
                                                
            self.validation_ds = UrbanSound8KDataset(
                                                dataset_path=self.dataset_path, 
                                                validation_fold=self.hparams.validation_fold, 
                                                feature_name=self.hparams.feature_name, 
                                                feature_processing_parameters=self.hparams.feature_processing_parameters, 
                                                train=False,
                                                signal_augmentation=False,
                                                feature_augmentation=False,
                                                augmentation_parameters=self.hparams.augmentation_parameters
                                                )


    def train_dataloader(self):
        train_dataloader = DataLoader(
                                    dataset=self.train_ds, 
                                    batch_size=self.hparams.batch_size, 
                                    shuffle=True,
                                    num_workers=self.hparams.num_workers, 
                                    pin_memory=True
                                    )
        return train_dataloader


    def val_dataloader(self):
        validation_dataloader = DataLoader(
                                    dataset=self.validation_ds, 
                                    batch_size=self.hparams.batch_size,
                                    shuffle=False, 
                                    num_workers=self.hparams.num_workers, 
                                    pin_memory=True
                                    )
        return validation_dataloader
    
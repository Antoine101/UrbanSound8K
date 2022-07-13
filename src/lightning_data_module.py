import pytorch_lightning as pl
from dataset_v2 import UrbanSound8KDataset
from torch.utils.data import DataLoader, SubsetRandomSampler

class UrbanSound8KDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, num_workers, transforms_params, validation_fold):
        super().__init__()
        self.save_hyperparameters()
        self.prepare_data_per_node = True


    def prepare_data(self) -> None:
        return super().prepare_data()
    

    def setup(self, stage=None):

        if stage == "fit" or stage is None:
            self.train_ds = UrbanSound8KDataset(dataset_path="dataset", validation_fold=self.hparams.validation_fold, transforms_params=self.hparams.transforms_params, train=True)
            self.validation_ds = UrbanSound8KDataset(dataset_path="dataset", validation_fold=self.hparams.validation_fold, transforms_params=self.hparams.transforms_params, train=False)


    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True)
        return train_dataloader


    def val_dataloader(self):
        validation_dataloader = DataLoader(self.validation_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True)
        return validation_dataloader
    
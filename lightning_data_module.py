import pytorch_lightning as pl
from dataset import UrbanSound8KDataset
from torch.utils.data import DataLoader, SubsetRandomSampler

class UrbanSound8KDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, transforms_params, validation_fold):
        self.batch_size = batch_size
        self.transforms_params = transforms_params
        self.validation_fold = validation_fold

        
    def setup(self):
        self.train_ds = UrbanSound8KDataset(dataset_path="dataset", validation_fold=self.validation_fold, transforms_params=self.transforms_params, train=True)
        self.validation_ds = UrbanSound8KDataset(dataset_path="dataset", validation_fold=self.validation_fold, transforms_params=self.transforms_params, train=False)


    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_ds, batch_size=self.batch_size)
        return train_dataloader

    
    def val_dataloader(self):
        validation_dataloader = DataLoader(self.validation_ds, batch_size=self.batch_size)
        return validation_dataloader
    
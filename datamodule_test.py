import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule

class CIFAR100DataModule(LightningDataModule):
    def __init__(self, dataset_dir, batch_size=64, num_workers=2):
        super().__init__()
        self.metadata = pd.read_csv(os.path.join(dataset_dir, "UrbanSound8K.csv"))
        self.batch_size = batch_size
        self.num_workers = num_workers
    

    def prepare_data(self):
        self.train = 
        self.validation =


    def setup(self, stage=None):
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
        train_set_transforms = transforms.Compose([
                                            transforms.ToTensor()
                                            ])

        validation_set_transforms = transforms.Compose([
                                            transforms.ToTensor()
                                            ])                                            


        # Create the train, validation and test sets
        self.cifar100_train = datasets.CIFAR100(root="./data", train=True, transform=train_set_transforms)
        self.cifar100_validation = datasets.CIFAR100(root="./data", train=True, transform=validation_set_transforms)

        # Retrieve classes from the train set
        self.classes = self.cifar100_train.classes


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, sampler=self.train_sampler, num_workers=self.num_workers)


    def val_dataloader(self):
        return DataLoader(self.validation, batch_size=self.batch_size, sampler=self.validation_indices, num_workers=self.num_workers)
import copy
import os

import random
import pytorch_lightning as pl
import torch
from torchvision import transforms, datasets

import numpy as np
from torch.utils.data import SubsetRandomSampler, DataLoader, Dataset, Subset


class CombinedCifarDataModule(pl.LightningDataModule):
    """
    This class is used to combine the CIFAR10 dataset and the corrupted CIFAR10 dataset.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cifar_val = CifarValDataModule(args)
        self.cifar_corrupted = CifarCorruptedDataModule(args)

    def setup(self, stage=None):

        self.cifar_val.setup(stage)
        self.cifar_corrupted.setup(stage)
        if self.args.train_size is not None:
            indx = np.arange(len(self.cifar_val.dataset_train))
            np.random.shuffle(indx)
            # Take a Subset of the dataset
            self.cifar_val.dataset_train = Subset(self.cifar_val.dataset_train, indx[:self.args.train_size])
            indx = np.arange(len(self.cifar_corrupted.dataset_train))
            np.random.shuffle(indx)
            self.cifar_corrupted.dataset_train = Subset(self.cifar_corrupted.dataset_train,
                                                        indx[:self.args.train_size])

    def train_dataloader(self):
        val_data_loader = self.cifar_val.train_dataloader()
        corrupted_data_loader = self.cifar_corrupted.train_dataloader()
        return {"cifar": val_data_loader,
                self.args.corruption: corrupted_data_loader}

    def val_dataloader(self):
        val_data_loader = self.cifar_val.test_dataloader()
        corrupted_data_loader = self.cifar_corrupted.test_dataloader()
        return val_data_loader, corrupted_data_loader


class CifarValDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input_shape = (3, 32, 32)
        self.num_classes = 10
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(
            #    mean=[0.5, 0.5, 0.5],
            #    std=[0.5, 0.5, 0.5])
        ])

    def setup(self, stage=None):
        self.dataset_train = datasets.CIFAR10(self.args.data_dir + "/CIFAR10/",
                                                   train=False,
                                                   download=False, transform=self.transform)

    def train_dataloader(self):
        train_loader = DataLoader(self.dataset_train,
                                  batch_size=self.args.batch_size,
                                  num_workers=self.args.num_workers,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
        return train_loader

    def test_dataloader(self):
        train_loader = DataLoader(self.dataset_train,
                                  batch_size=self.args.batch_size,
                                  num_workers=self.args.num_workers,
                                  shuffle=False,
                                  pin_memory=True,
                                  drop_last=True)
        return train_loader


# write a class to read corrupted CIFAR-10 dataset from the .npy files with the specified corruption and severity
class CifarCorruptedDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input_shape = (3, 32, 32)
        self.num_classes = 10
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(
            #    mean=[0.4914, 0.4822, 0.4465],
            #    std=[0.2023, 0.1994, 0.2010])
        ])
        self.corruption_types = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                                 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                                 'snow', 'frost', 'fog', 'brightness',
                                 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

    def setup(self, stage=None):
        if self.args.corruption_index is not None:
            self.args.corruption = self.corruption_types[self.args.corruption_index]
            self.args.corruption_type = self.args.corruption

        self.dataset_train = CifarCorruptedDataset(self.args.data_dir + "/CIFAR-10-C/",
                                                   self.args.corruption,
                                                   self.args.severity,
                                                   transform=self.transform)

    def train_dataloader(self):
        train_loader = DataLoader(self.dataset_train,
                                  batch_size=self.args.batch_size,
                                  num_workers=self.args.num_workers,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
        self.train_loader = train_loader
        return train_loader

    def val_dataloader(self):
        loader = DataLoader(self.dataset_train,
                                  batch_size=self.args.batch_size,
                                  num_workers=self.args.num_workers,
                                  shuffle=False,
                                  pin_memory=True,
                                  drop_last=True)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.dataset_train,
                            batch_size=self.args.batch_size,
                            num_workers=self.args.num_workers,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=True)
        return loader


class CifarCorruptedDataset(Dataset):
    def __init__(self, base_c_path, corruption, severity,
                 transform=None):
        self.images = np.load(os.path.join(base_c_path, corruption + '.npy'))
        self.labels = torch.LongTensor(np.load(os.path.join(base_c_path, 'labels.npy')))

        if severity == 1:
            self.images = self.images[:10000]
            self.labels = self.labels[:10000]
        elif severity == 2:
            self.images = self.images[10000:20000]
            self.labels = self.labels[10000:20000]
        elif severity == 3:
            self.images = self.images[20000:30000]
            self.labels = self.labels[20000:30000]
        elif severity == 4:
            self.images = self.images[30000:40000]
            self.labels = self.labels[30000:40000]
        elif severity == 5:
            self.images = self.images[40000:]
            self.labels = self.labels[40000:]
        else:
            raise ValueError('Severity level must be between 1 and 5')
        # read only the appropriate severity
        self.transform = transform

    def __getitem__(self, index):
        # imgs = torch.from_numpy(self.images[index]).float()
        imgs = self.images[index]
        labels = self.labels[index]
        if self.transform:
            imgs = self.transform(imgs)
        return imgs, labels

    def __len__(self):
        return len(self.labels)
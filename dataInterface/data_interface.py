# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import yaml
import inspect
import importlib
import pickle as pkl
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, ConcatDataset, Sampler, DistributedSampler


class DInterface(pl.LightningDataModule):

    def __init__(self, batch_size=16, num_workers=8, dataset=None, dataset_dict=None):
        super().__init__()
        
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.load_data_module(dataset, dataset_dict)

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_set = self.generate_dataset('train')
            self.val_set = self.generate_dataset('test')
        elif stage == 'test':
            self.test_set = self.generate_dataset('test')
        else: 
            raise ValueError(f"Invalid stage: {stage}")

    def generate_dataset(self, stage):
        return self.Dataset(phase=stage, **self.dataset_dict)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def load_data_module(self, dataset, dataset_dict):
        self.Dataset = dataset
        self.dataset_dict = dataset_dict
        

class SubsetSampler(Sampler):
    def __init__(self, dataset_sizes, shuffle=False):
        self.dataset_sizes = dataset_sizes
        self.indices = []
        current_indice = 0
        for dataset_size in dataset_sizes:
            if shuffle:
                self.indices.extend(torch.randperm(dataset_size) + current_indice)
            else:
                self.indices.extend(torch.arange(dataset_size) + current_indice)
            current_indice += dataset_size

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
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

    def __init__(self, dataset_cfg=None, data_cfg=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.dataset_cfg.update(data_cfg)
        
        self.num_workers = self.dataset_cfg['num_workers']
        self.dataset = self.dataset_cfg['dataset']
        self.batch_size = self.dataset_cfg['batch_size']

        self.load_data_module()

    def setup(self, stage=None):
        # load empty file list config
        if 'skip_empty_rd' in self.dataset_cfg and self.dataset_cfg['skip_empty_rd']:
            with open('configs/data/non_empty_files.yaml', 'r') as f:
                self.dataset_cfg['train_data_if_skip'] = yaml.load(f, Loader=yaml.FullLoader)
        else:
            self.dataset_cfg['train_data_if_skip'] = None

        if stage == 'fit':
            self.trainset = self.generate_dataset('train')
            self.valset = self.generate_dataset('test')
        elif stage == 'test':
            self.testset = self.generate_dataset('test')
        else: 
            raise ValueError(f"Invalid stage: {stage}")

    def generate_dataset(self, set_type):
        sets = self.dataset_cfg['path'][set_type + "_sets"]

        datasets = []
        for subset_data in sets:
            subsets = sets[subset_data]
            for subset in subsets:
                datasets.append(self.Dataset(set_type=set_type, subset=subset, use_camera=True, dataset_cfg=self.dataset_cfg))
        return datasets

    def train_dataloader(self):
        dataset = ConcatDataset(self.trainset)
        sampler = SubsetSampler([len(subset) for subset in self.trainset], shuffle=True)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        dataset = ConcatDataset(self.valset)
        sampler = SubsetSampler([len(subset) for subset in self.valset], shuffle=False)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def test_dataloader(self):
        dataset = ConcatDataset(self.testset)
        sampler = SubsetSampler([len(subset) for subset in self.testset], shuffle=False)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def load_data_module(self):
        name = self.dataset
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            self.Dataset = getattr(importlib.import_module('.'+name, package=__package__), camel_name)
        except:
            raise ValueError(f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')


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
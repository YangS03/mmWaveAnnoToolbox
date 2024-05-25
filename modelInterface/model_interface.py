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

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import importlib
import pytorch_lightning as pl


class MInterface(pl.LightningModule):
    def __init__(self, network_cfg, model_cfg, loss_cfg, lr_cfg, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.load_model()
        self.configure_loss()

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):
        return 

    def validation_step(self, batch, batch_idx):
        return 

    def test_step(self, batch, batch_idx):
        return
    
    def on_train_end(self):
        return 
    
    def on_validation_epoch_end(self):
        return

    def on_test_epoch_end(self):
        return 

    # def configure_optimizers(self):
    #     # Optimizer
    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=0)
    #     # Scheduler
    #     scheduler_name = lr_cfg['lr_scheduler']
    #     Scheduler = getattr(importlib.import_module('.lr_scheduler', package='torch.optim'), scheduler_name)
    #     scheduler = Scheduler(optimizer, **lr_cfg['lrs_args'])
    #     return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": monitor, "frequency": self.trainer.check_val_every_n_epoch}}

    def configure_loss(self):
        return

    def load_model(self):
        return
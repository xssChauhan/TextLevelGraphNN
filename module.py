import enum
from pytorch_lightning import LightningModule
from torch import batch_norm

from torch.utils.data import DataLoader

import torch
import utils
import dataset

from typing import List
from typing import Union

from model import GNNModel

class BaseModule(LightningModule):

    def __init__(self, hparams):

        super().__init__()
        self.hparams = hparams
        self.model = self.init_model()
    
    def configure_optimizers(self):

        return torch.optim.Adam(lr=1e-4, params=self.parameters())

    def _get_dataloader(self, data_path):

        ds =  dataset.R8Dataset(
            data_path
        )

        dataloader = DataLoader(
            ds, batch_size=self.hparams.batch_size,
            shuffle=True, num_workers=1,
            pin_memory=True, collate_fn=utils.collate_fn

        )
        return dataloader


    def train_dataloader(self) -> DataLoader:

        loader = self._get_dataloader(self.hparams.train_path) 

        self.vocab = loader.dataset.vocab
    
        return loader
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:

        loader = self._get_dataloader(self.hparams.test_path)

        return loader


class TextGNNModule(BaseModule):

    def init_model(self):

        model = GNNModel(
            num_node=19975,
            embedding_dim=100,
            num_cls=8,
            pre_embed=None #TODO load the embedding
        )
        return model

    def training_step(self, batch, batch_idx):     
        # Convert the batch to graph batch
        # For each batch we need

        x, nx, ew, y= utils.make_graph(batch, self.vocab, self.hparams.n_neighbours, device=self.device)

        x = x.long()
        nx = nx.long()
        ew = ew.long()
        pred = self.model(x,nx,ew)

        loss = torch.nn.functional.nll_loss(
            pred, y
        ) 
        return loss
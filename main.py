from module import BaseModule, TextGNNModule

from argparse import Namespace


hparams = Namespace(**{
    "train_path": "data/train.txt",
    "test_path": "data/test.txt",
    "batch_size": 1,
    "n_neighbours": 2
})

model = TextGNNModule(hparams)

from pytorch_lightning.trainer import Trainer

trainer = Trainer()
trainer.fit(model)



from typing import Any, Dict, Optional, Tuple

import torch
import torchvision
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from torchvision.transforms import transforms
from torchvision import transforms as T

class pfDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (320, 320, 1245),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        resize: int = 256,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.resize = 224
        # data transformations
        self.transforms = T.Compose([
            T.ToTensor(),
            T.RandomRotation(15),
            T.CenterCrop(self.resize),
            T.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None


    @property
    def num_classes(self):
        return 8

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        # MNIST(self.hparams.data_dir, train=True, download=True)
        # MNIST(self.hparams.data_dir, train=False, download=True)
        # download
        print("------------------prepare_data--------------")


    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:

            dataset = torchvision.datasets.ImageFolder(self.hparams.data_dir, transform=self.transforms)
            # l = len(dataset)
            # print("---------------",l)
            # l1 = int(0.1 * l)
            # l2 = int(0.1 * l)
            print("------------------setup data--------------")
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                # lengths= [l - l1-l2, l1,l2],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data = pfDataModule(data_dir="../../../../data/pf")
    # data = PokemanDataModule(data_dir="../../data/pokeman")
    data.setup()
    print(len(data.data_train),len(data.data_test),len(data.data_val))
    # T.ToPILImage()(data.data_train[0][0]).show()


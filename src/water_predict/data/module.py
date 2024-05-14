from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from water_predict.data.dataset import OverallDataset


class WaterPredict(LightningDataModule):
    def __init__(self, batch_size: int, file_path: str, x_length: int, share_length: int, y_length: int):
        super().__init__()
        self.batch_size = batch_size
        self.file_path = file_path
        self.x_length = x_length
        self.share_length = share_length
        self.y_length = y_length

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: [str] = None):
        if stage == "fit":
            self.train_dataset = OverallDataset(file_path=self.file_path, flag="train",
                                                x_length=self.x_length, share_length=self.share_length,
                                                y_length=self.y_length)
            self.val_dataset = OverallDataset(file_path=self.file_path, flag="val",
                                              x_length=self.x_length, share_length=self.share_length,
                                              y_length=self.y_length)
        if stage == "test":
            self.val_dataset = OverallDataset(file_path=self.file_path, flag="val",
                                              x_length=self.x_length, share_length=self.share_length,
                                              y_length=self.y_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=8, shuffle=False)

    def test_dataloader(self):
        return self.val_dataloader()

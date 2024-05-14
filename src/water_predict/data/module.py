from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from water_predict.data.dataset import FitDataset, TestDataset


class WaterPredict(LightningDataModule):
    def __init__(self, batch_size: int, file_path: str, watershed_ids: list[int],
                 x_length: int, y_length: int):
        super().__init__()
        self.batch_size = batch_size
        self.file_path = file_path
        self.watershed_ids = watershed_ids
        self.x_length = x_length
        self.y_length = y_length

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: [str] = None):
        if stage == "fit":
            self.train_dataset = FitDataset(file_path=self.file_path, flag="train", watershed_ids=self.watershed_ids,
                                            x_length=self.x_length, y_length=self.y_length)
            self.val_dataset = FitDataset(file_path=self.file_path, flag="val", watershed_ids=self.watershed_ids,
                                          x_length=self.x_length, y_length=self.y_length)
        if stage == "test":
            self.test_dataset = TestDataset(file_path=self.file_path, watershed_ids=self.watershed_ids)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, pin_memory=True, shuffle=False)

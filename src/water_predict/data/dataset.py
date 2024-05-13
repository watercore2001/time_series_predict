from torch.util.data import Dataset
from typing import Literal
import pandas as pd

class StationDataset(Dataset):
    TRAIN_PRECENT = 0.8
    def __init__(self, flag: Literal["train", "val"], data: pd.DataFrame, scaler,
                 x_length: int, share_length: int, y_length: int):
        super().__init__()
        train_border = int(df.shape[0] * TRAIN_PRECENT)
        if flag == "train":
            border1 = 0
            border2 = train_border
        if flag == "val":
            border1 = train_border - x_length
            border2 = df.shape[0]


    def __len__(self):
        return 0

    def __getitem__(self, index: int):
        return {"x": x, "x_weeks": [1,2,3], "x_years": [2022, 2022, 2023],
                "y": y, "y_weeks": [1,2,3], "y_years": [2022, 2022, 2023],
                "station_id": 0, "lat": 0., "lng": 0., "watershed_id": 1}

class OverallDataset(Dataset):
    def __init__(self, flag: Literal["train", "val"], x_length: int, share_length: int, y_length: int):
        super().__init__()
        self.station_datasets = []
        # init
        # norm

        self.cumulative_lengths = [0]
        for dataset in station_datasets:
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(dataset))

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx: int):
        dataset_index = 0
        while self.cumulative_lengths[dataset_index + 1] <= idx:
            dataset_index += 1
        inner_idx = idx - self.cumulative_lengths[dataset_index]
        return self.station_datasets[dataset_index][inner_idx]


